"""Benchmark datasets: pinned Hugging Face revisions, SHA-256 checked, normalised to one JSONL item format.

Item format::

    {"id": "medqa_usmle-0001", "dataset": "medqa_usmle", "lang": "en", "kind": "mcq",
     "question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "context": null,
     "answer": "C", "meta": {...}}

``kind`` is ``mcq`` (answer is an option letter) or ``yesno`` (answer is yes / no / maybe).
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

HF = "https://huggingface.co/datasets"


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    repo: str
    revision: str  # git commit of the HF dataset repo
    path: str
    sha256: str
    n: int
    lang: str
    kind: str
    license: str
    citation: str
    normalize: Callable[[List[Dict[str, Any]], "Source"], List[Dict[str, Any]]]

    @property
    def url(self) -> str:
        return "%s/%s/resolve/%s/%s" % (HF, self.repo, self.revision, self.path)


def _mcq(rows: List[Dict[str, Any]], src: Source) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        opts = {o["key"]: o["value"] for o in r["options"]}
        q = r["question"].strip()
        meta: Dict[str, Any] = {"meta_info": r.get("meta_info")}
        if src.lang == "zh-TW":
            # TWMLE continues some cases across questions ("承上題" = "following the previous question"); the dataset does not
            # keep the link, so these items are scored as given and also reported separately.
            meta["depends_on_previous"] = bool(re.match(r"^\s*(承上題|承上|續上題|同上題)", q))
        out.append({"id": "%s-%04d" % (src.name, i), "dataset": src.name, "lang": src.lang, "kind": "mcq", "question": q,
                    "options": opts, "context": None, "answer": r["answer_idx"], "meta": meta})
    return out


def _pubmedqa(rows: List[Dict[str, Any]], src: Source) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        ctx = r["context"]
        parts = []
        for label, text in zip(ctx.get("labels") or [], ctx.get("contexts") or []):
            parts.append("%s: %s" % (label, text) if label else text)
        out.append({"id": "%s-%04d" % (src.name, i), "dataset": src.name, "lang": src.lang, "kind": "yesno",
                    "question": r["question"].strip(), "options": None, "context": "\n".join(parts), "answer": r["final_decision"],
                    "meta": {"pubid": r["pubid"]}})
    return out


SOURCES: Dict[str, Source] = {s.name: s for s in [
    Source("medqa_usmle", "MedQA-USMLE (4 options), test", "bigbio/med_qa", "484a6c066fe8e75c83edea0c88b5169316714fcd",
           "med_qa_en_4options_source/test-00000-of-00001.parquet",
           "b50b49fad2642c10ed343c6174d5ab0cb8efad434e7b66465ac471094eb57deb", 1273, "en", "mcq", "MIT (MedQA)",
           "Jin et al., What Disease does this Patient Have? (MedQA), Applied Sciences 2021", _mcq),
    Source("medqa_twmle", "MedQA-TWMLE (Taiwan medical licensing exam, Traditional Chinese), test", "bigbio/med_qa",
           "484a6c066fe8e75c83edea0c88b5169316714fcd", "med_qa_tw_source/test-00000-of-00001.parquet",
           "9756921fcff39ee482d8e06488b4ff32c26c9b966ac4c67c07f854da583f34b0", 1413, "zh-TW", "mcq", "MIT (MedQA)",
           "Jin et al., What Disease does this Patient Have? (MedQA), Applied Sciences 2021", _mcq),
    Source("pubmedqa", "PubMedQA (expert-labeled, reasoning-required)", "qiaojin/PubMedQA",
           "9001f2853fb87cab8d220904e0de81ac6973b318", "pqa_labeled/train-00000-of-00001.parquet",
           "3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c", 1000, "en", "yesno", "MIT",
           "Jin et al., PubMedQA: A Dataset for Biomedical Research Question Answering, EMNLP 2019", _pubmedqa),
]}
DEFAULT_SUITE = ["medqa_usmle", "medqa_twmle", "pubmedqa"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(data_dir: Path, names: Iterable[str] = DEFAULT_SUITE) -> Dict[str, Dict[str, Any]]:
    """Download (if needed), verify and normalise; writes <data_dir>/<name>.jsonl. Returns {name: info}."""
    import pyarrow.parquet as pq

    data_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for name in names:
        src = SOURCES[name]
        raw = data_dir / "raw" / ("%s.parquet" % name)
        raw.parent.mkdir(parents=True, exist_ok=True)
        if not raw.exists() or sha256(raw) != src.sha256:
            tmp = raw.with_suffix(".part")
            urllib.request.urlretrieve(src.url, tmp)
            tmp.rename(raw)
        digest = sha256(raw)
        if digest != src.sha256:
            raise RuntimeError("%s: sha256 %s does not match the pinned %s" % (name, digest, src.sha256))
        items = src.normalize(pq.read_table(raw).to_pylist(), src)
        if len(items) != src.n:
            raise RuntimeError("%s: %d items, expected %d" % (name, len(items), src.n))
        out = data_dir / ("%s.jsonl" % name)
        with open(out, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        info[name] = {"title": src.title, "repo": src.repo, "revision": src.revision, "path": src.path, "sha256": digest,
                      "n": len(items), "lang": src.lang, "kind": src.kind, "license": src.license, "items_sha256": sha256(out)}
    return info


def load(data_dir: Path, name: str) -> List[Dict[str, Any]]:
    with open(data_dir / ("%s.jsonl" % name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

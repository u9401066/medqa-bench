"""Scores a run (accuracy with 95% Wilson CI, failure modes, speed, resources, energy) and compares runs pairwise."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .datasets import load


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _pct(xs: List[float], q: float) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = (len(xs) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def load_predictions(run: Path) -> Dict[str, Dict[str, Any]]:
    """Last record per item (a resumed run re-tries items that failed)."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(run / "predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def macro_f1(golds: List[str], preds: List[Optional[str]], labels: List[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(golds, preds) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(golds, preds) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def _acc(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if r.get("correct"))
    lo, hi = wilson(k, n)
    return {"n": n, "correct": k, "acc": k / n if n else None, "ci95": [lo, hi]}


def _subsets(name: str, rows: List[Dict[str, Any]], items: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    subs: Dict[str, Any] = {}
    if name == "medqa_usmle":
        for tag in ("step1", "step2&3"):
            sel = [r for r in rows if items[r["id"]]["meta"].get("meta_info") == tag]
            if sel:
                subs[tag] = _acc(sel)
    if name == "medqa_twmle":
        dep = [r for r in rows if items[r["id"]]["meta"].get("depends_on_previous")]
        subs["standalone"] = _acc([r for r in rows if not items[r["id"]]["meta"].get("depends_on_previous")])
        subs["depends_on_previous (承上題)"] = _acc(dep)
    if name == "pubmedqa":
        for lab in ("yes", "no", "maybe"):
            subs["gold=%s" % lab] = _acc([r for r in rows if r["gold"] == lab])
        subs["macro_f1"] = macro_f1([r["gold"] for r in rows], [r.get("pred") for r in rows], ["yes", "no", "maybe"])
    return subs


def _resources(run: Path, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Resource statistics over the active part of each session; idle samples taken just before each session form the baseline."""
    path = run / "resources.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("t")]

    def col(name: str, sel: List[Dict[str, str]]) -> List[float]:
        return [float(r[name]) for r in sel if r.get(name) not in (None, "")]

    base: List[Dict[str, str]] = []
    act: List[Dict[str, str]] = []
    energy_j, covered_s = 0.0, 0.0
    for s in sessions:
        if "end" not in s:
            continue
        b0 = s.get("baseline_start", s["start"])
        base += [r for r in rows if b0 <= float(r["t"]) < s["start"]]
        win = [r for r in rows if s["start"] <= float(r["t"]) <= s["end"] and r.get("gpu_power_w") not in (None, "")]
        act += win
        for r0, r1 in zip(win, win[1:]):
            dt = float(r1["t"]) - float(r0["t"])
            if 0 < dt <= 5:
                energy_j += dt * (float(r0["gpu_power_w"]) + float(r1["gpu_power_w"])) / 2
                covered_s += dt
    if not act:
        return {}
    pw = col("gpu_power_w", act)
    bw = col("gpu_power_w", base)
    base_w = statistics.mean(bw) if bw else None
    return {
        "samples": len(act),
        "baseline_samples": len(bw),
        "gpu_mem_used_mib": {"mean": statistics.mean(col("gpu_mem_used_mib", act)), "peak": max(col("gpu_mem_used_mib", act)),
                             "baseline": statistics.mean(col("gpu_mem_used_mib", base)) if base else None},
        "gpu_util_pct_mean": statistics.mean(col("gpu_util_pct", act)),
        "gpu_power_w": {"mean": statistics.mean(pw), "peak": max(pw), "baseline_idle": base_w},
        "gpu_energy_wh": energy_j / 3600.0,
        "gpu_energy_wh_above_idle": (energy_j - base_w * covered_s) / 3600.0 if base_w is not None else None,
        "server_rss_mib": {"mean": statistics.mean(col("server_rss_mib", act)), "peak": max(col("server_rss_mib", act))},
        "ram_used_mib": {"mean": statistics.mean(col("ram_used_mib", act)), "peak": max(col("ram_used_mib", act)),
                         "baseline": statistics.mean(col("ram_used_mib", base)) if base else None},
        "cpu_pct_mean": statistics.mean(col("cpu_pct", act)),
        "gpu_temp_c_peak": max(col("gpu_temp_c", act)) if col("gpu_temp_c", act) else None,
        "external_load": _external(act),
    }


def _external(act: List[Dict[str, str]]) -> Optional[Dict[str, float]]:
    """Samples where the server was busy with more requests than ours (other users of a shared server)."""
    pairs = [(float(r["server_busy"]), float(r["inflight"])) for r in act if r.get("server_busy") not in (None, "") and r.get("inflight") not in (None, "")]
    if not pairs:
        return None
    extra = [max(0.0, b - i) for b, i in pairs]
    return {"samples": len(pairs), "share_with_other_requests": sum(1 for e in extra if e > 0) / len(pairs), "mean_other_requests": statistics.mean(extra)}


def score_run(run: Path, data_dir: Path) -> Dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    preds = load_predictions(run)
    result: Dict[str, Any] = {"run": run.name, "label": manifest["config"].get("label", manifest["config"]["name"]),
                              "config": manifest["config"], "datasets": {}}
    all_rows: List[Dict[str, Any]] = []
    for name, dinfo in manifest["datasets"].items():
        items = {it["id"]: it for it in load(data_dir, name)}
        rows = [r for r in preds.values() if r["dataset"] == name]
        all_rows += rows
        ok = [r for r in rows if not r.get("error")]
        lat = [r["latency_s"] for r in ok]
        d = _acc(rows)
        d.update({
            "expected_items": dinfo["n_items"],
            "missing_items": dinfo["n_items"] - len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
            "no_answer": sum(1 for r in ok if r.get("pred") is None),
            "truncated": sum(1 for r in ok if r.get("finish_reason") == "length"),
            "latency_s": {"mean": statistics.mean(lat) if lat else None, "median": _pct(lat, 0.5), "p90": _pct(lat, 0.9)},
            "ttft_s_median": _pct([r.get("ttft_s") for r in ok], 0.5),
            "completion_tokens": {"mean": statistics.mean([r["completion_tokens"] for r in ok]) if ok else None,
                                  "median": _pct([r["completion_tokens"] for r in ok], 0.5)},
            "prompt_tokens_mean": statistics.mean([r["prompt_tokens"] for r in ok if r.get("prompt_tokens") is not None] or [0]),
            "decode_tps_median": _pct([r.get("decode_tps") for r in ok], 0.5),
            "subsets": _subsets(name, rows, items),
        })
        result["datasets"][name] = d
    ok = [r for r in all_rows if not r.get("error")]
    sessions = manifest.get("sessions", [])
    busy = sum(max(0.0, s.get("end", s["start"]) - s["start"]) for s in sessions)
    tokens = sum(r["completion_tokens"] for r in ok)
    result["overall"] = {**_acc(all_rows), "wall_s": busy, "completion_tokens": tokens,
                         "throughput_tok_s": tokens / busy if busy else None,
                         "questions_per_hour": 3600 * len(ok) / busy if busy else None,
                         "decode_tps_median": _pct([r.get("decode_tps") for r in ok], 0.5),
                         "latency_s_median": _pct([r["latency_s"] for r in ok], 0.5),
                         "concurrency": manifest["config"].get("concurrency", 1)}
    res = _resources(run, sessions)
    if res:
        res["gpu_energy_wh_per_question"] = res["gpu_energy_wh"] / len(ok) if ok else None
        result["resources"] = res
    result["manifest"] = {k: manifest.get(k) for k in ("medbench_version", "git_rev", "prompt_version", "host", "server", "datasets", "sessions")}
    (run / "summary.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    return result


def compare(a: Path, b: Path, seed: int = 0, n_boot: int = 10000) -> Dict[str, Any]:
    """Paired comparison on the items both runs answered: McNemar exact test and a bootstrap 95% CI of the accuracy difference."""
    pa, pb = load_predictions(a), load_predictions(b)
    out: Dict[str, Any] = {}
    datasets = sorted({r["dataset"] for r in pa.values()} & {r["dataset"] for r in pb.values()})
    rng = random.Random(seed)
    for name in datasets + ["ALL"]:
        ids = sorted(i for i in pa if i in pb and (name == "ALL" or pa[i]["dataset"] == name))
        if not ids:
            continue
        xa = [1 if pa[i].get("correct") else 0 for i in ids]
        xb = [1 if pb[i].get("correct") else 0 for i in ids]
        both = sum(1 for x, y in zip(xa, xb) if x and y)
        only_a = sum(1 for x, y in zip(xa, xb) if x and not y)
        only_b = sum(1 for x, y in zip(xa, xb) if y and not x)
        neither = len(ids) - both - only_a - only_b
        diffs = [x - y for x, y in zip(xa, xb)]
        boots = []
        for _ in range(n_boot):
            s = 0
            for _ in range(len(diffs)):
                s += diffs[rng.randrange(len(diffs))]
            boots.append(s / len(diffs))
        boots.sort()
        out[name] = {"n": len(ids), "acc_a": sum(xa) / len(ids), "acc_b": sum(xb) / len(ids), "diff_a_minus_b": sum(diffs) / len(ids),
                     "diff_ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
                     "both": both, "only_a": only_a, "only_b": only_b, "neither": neither, "mcnemar_p": mcnemar_exact(only_a, only_b)}
    return out

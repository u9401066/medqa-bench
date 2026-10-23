"""medbench CLI: prepare data, run a model, score runs, write the report."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

from .datasets import DEFAULT_SUITE, SOURCES, prepare


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="medbench", description=__doc__)
    ap.add_argument("--data", default="data", help="dataset folder [data]")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download, verify (sha256) and normalise the datasets")
    p.add_argument("--datasets", default=",".join(DEFAULT_SUITE))

    p = sub.add_parser("run", help="run one model config over datasets (resumable)")
    p.add_argument("--config", required=True)
    p.add_argument("--datasets", default=",".join(DEFAULT_SUITE))
    p.add_argument("--out", help="run folder [results/<config name>]")
    p.add_argument("--sample", type=int, help="evenly spaced subset per dataset (pilot)")
    p.add_argument("--baseline-s", type=float, default=15.0, help="idle seconds recorded before the first request (power baseline)")

    p = sub.add_parser("score", help="score runs (writes <run>/summary.json)")
    p.add_argument("runs", nargs="+")

    p = sub.add_parser("report", help="write the Markdown report for two or more runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="reports/REPORT.md")
    p.add_argument("--title", default="醫學問答評測：單張 RTX 4090 上的本地模型")
    p.add_argument("--notes", help="Markdown file appended as discussion/limitations")

    p = sub.add_parser("docx", help="write the Word report for two or more runs (needs the [report] extra)")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="reports/REPORT.docx")
    p.add_argument("--title", default="醫學問答評測報告")
    p.add_argument("--notes", help="Markdown file with the discussion/limitations sections")
    p.add_argument("--env", default="reports/environment.json", help="test environment record (medbench env)")
    p.add_argument("--repo-url", default="https://github.com/u9401066/medqa-bench")

    p = sub.add_parser("env", help="record this host (OS, CPU, RAM, GPU, driver, PCIe, disks) into an environment JSON")
    p.add_argument("--out", default="reports/environment.json")

    a = ap.parse_args(argv)
    data = Path(a.data)
    if a.cmd == "prepare":
        info = prepare(data, [d for d in a.datasets.split(",") if d])
        (data / "datasets.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
        for k, v in info.items():
            print("%-12s %5d items  sha256 %s  (%s@%s)" % (k, v["n"], v["sha256"][:12], v["repo"], v["revision"][:10]))
        return 0
    if a.cmd == "run":
        from .runner import run

        cfg = json.loads(Path(a.config).read_text())
        names = [d for d in a.datasets.split(",") if d]
        for n in names:
            if n not in SOURCES:
                print("unknown dataset %s" % n, file=sys.stderr)
                return 2
        if not all((data / ("%s.jsonl" % n)).exists() for n in names):
            prepare(data, names)
        out = Path(a.out or ("results/%s" % cfg["name"]))
        res = asyncio.run(run(cfg, data, names, out, sample=a.sample, baseline_s=a.baseline_s))
        print(json.dumps(res))
        return 0
    if a.cmd == "score":
        from .score import score_run

        for r in a.runs:
            s = score_run(Path(r), data)
            o = s["overall"]
            print("%-34s acc %.1f%% (%d/%d)  %.0f q/h  %s tok/s decode" % (s["label"][:34], 100 * o["acc"], o["correct"], o["n"],
                  o["questions_per_hour"] or 0, o["decode_tps_median"]))
            for name, d in s["datasets"].items():
                print("   %-12s %.1f%% [%.1f, %.1f]  n=%d  no-answer %d  truncated %d  errors %d" % (
                    name, 100 * d["acc"], 100 * d["ci95"][0], 100 * d["ci95"][1], d["n"], d["no_answer"], d["truncated"], d["errors"]))
        return 0
    if a.cmd == "report":
        from .report import write_report

        path = write_report([Path(r) for r in a.runs], data, Path(a.out), a.title, Path(a.notes) if a.notes else None)
        print(path)
        return 0
    if a.cmd == "docx":
        try:
            from .docx_report import write_docx
        except ImportError as e:
            print("medbench docx needs python-docx and matplotlib: pip install -e '.[report]' (%s)" % e, file=sys.stderr)
            return 2
        path = write_docx([Path(r) for r in a.runs], data, Path(a.out), a.title, Path(a.notes) if a.notes else None,
                          Path(a.env) if a.env else None, a.repo_url)
        print(path)
        return 0
    if a.cmd == "env":
        import datetime

        from .envinfo import collect

        out = Path(a.out)
        rec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        rec["collected_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        rec["host"] = collect()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(rec["host"], ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

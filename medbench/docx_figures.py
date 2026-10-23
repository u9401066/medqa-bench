"""Figures for the Word report (PNG bytes): accuracy with 95% CI, per-question output length and latency, resource use
over the run. Labels are in English so they render without CJK fonts."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#7F7F7F"]
SHORT = {"medqa_usmle": "MedQA-USMLE", "medqa_twmle": "MedQA-TWMLE", "pubmedqa": "PubMedQA", "medmcqa": "MedMCQA"}


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def accuracy(S: List[Dict[str, Any]], names: List[str], majority: Dict[str, float]) -> bytes:
    groups = names + ["ALL"]
    k = len(S)
    w = 0.8 / k
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for si, s in enumerate(S):
        accs, lo, hi = [], [], []
        for g in groups:
            d = s["overall"] if g == "ALL" else s["datasets"][g]
            accs.append(100 * d["acc"])
            lo.append(100 * (d["acc"] - d["ci95"][0]))
            hi.append(100 * (d["ci95"][1] - d["acc"]))
        xs = [i + (si - (k - 1) / 2) * w for i in range(len(groups))]
        ax.bar(xs, accs, w * 0.95, color=COLORS[si % len(COLORS)], label=s["label"], yerr=[lo, hi], capsize=3,
               error_kw={"elinewidth": 1})
        for x, a, h in zip(xs, accs, hi):
            ax.text(x, a + h + 1.2, "%.1f" % a, ha="center", va="bottom", fontsize=8)
    for i, g in enumerate(groups):
        if g in majority:
            ax.hlines(100 * majority[g], i - 0.45, i + 0.45, colors="black", linestyles="dashed", linewidth=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["%s\n(n=%d)" % (SHORT.get(g, g), S[0]["datasets"][g]["n"]) if g != "ALL" else "All\n(n=%d)" % S[0]["overall"]["n"]
                        for g in groups])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="black", linestyle="dashed", linewidth=1))
    labels.append("Majority-class baseline")
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(labels), fontsize=8, frameon=False)
    return _png(fig)


def speed(preds: List[Dict[str, Dict[str, Any]]], labels: List[str], names: List[str]) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    k = len(preds)
    for ax, key, title in ((axes[0], "completion_tokens", "Output tokens per question (reasoning + answer)"),
                           (axes[1], "latency_s", "Wall-clock latency per question (s)")):
        data, pos, cols = [], [], []
        for gi, n in enumerate(names):
            for si, p in enumerate(preds):
                vals = [r[key] for r in p.values() if r["dataset"] == n and not r.get("error") and r.get(key) is not None]
                data.append(vals)
                pos.append(gi * (k + 1) + si)
                cols.append(COLORS[si % len(COLORS)])
        bp = ax.boxplot(data, positions=pos, widths=0.75, whis=(5, 95), showfliers=False, patch_artist=True,
                        medianprops={"color": "black"})
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        ax.set_xticks([gi * (k + 1) + (k - 1) / 2 for gi in range(len(names))])
        ax.set_xticklabels([SHORT.get(n, n) for n in names])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend([Patch(color=COLORS[i % len(COLORS)]) for i in range(k)], labels, fontsize=8, loc="upper right")
    fig.text(0.5, -0.02, "Boxes: 25th-75th percentile, line: median, whiskers: 5th-95th percentile", ha="center", fontsize=8)
    return _png(fig)


def resources(runs: List[Path], S: List[Dict[str, Any]], vram_total_gib: float, ram_total_gib: float) -> bytes:
    keys = [("gpu_power_w", "GPU power (W)", 1.0), ("gpu_mem_used_mib", "GPU memory used,\nwhole card (GiB)", 1 / 1024),
            ("ram_used_mib", "System RAM used (GiB)", 1 / 1024)]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 6.6), sharex=True)
    for si, (run, s) in enumerate(zip(runs, S)):
        with open(run / "resources.csv", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("t") and r.get("gpu_power_w") not in (None, "")]
        sessions = [x for x in s["manifest"]["sessions"] if "end" in x]
        xh: List[float] = []
        ys: Dict[str, List[float]] = {k: [] for k, _, _ in keys}
        offset = 0.0
        for n, ss in enumerate(sessions):
            bins: Dict[int, List[Dict[str, str]]] = {}
            for r in rows:
                t = float(r["t"])
                if ss["start"] <= t <= ss["end"]:
                    bins.setdefault(int((t - ss["start"]) // 60), []).append(r)
            for b in sorted(bins):
                xh.append((offset + b * 60 + 30) / 3600)
                for k, _, scale in keys:
                    v = [float(r[k]) for r in bins[b] if r.get(k) not in (None, "")]
                    ys[k].append(scale * sum(v) / len(v) if v else math.nan)
            offset += ss["end"] - ss["start"]
            if n < len(sessions) - 1:
                for ax in axes:
                    ax.axvline(offset / 3600, color=COLORS[si % len(COLORS)], linestyle=":", linewidth=1)
        for ax, (k, _, _) in zip(axes, keys):
            ax.plot(xh, ys[k], color=COLORS[si % len(COLORS)], linewidth=1, label=s["label"])
    for ax, (k, title, _) in zip(axes, keys):
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[1].axhline(vram_total_gib, color="grey", linestyle="--", linewidth=1)
    axes[2].axhline(ram_total_gib, color="grey", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, vram_total_gib * 1.08)
    axes[2].set_ylim(0, ram_total_gib * 1.08)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[-1].set_xlabel("Benchmark running time (h); 1-minute means; dotted line = run resumed in a new session; "
                        "dashed grey = capacity", fontsize=8)
    return _png(fig)

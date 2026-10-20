"""Formal Markdown report (Traditional Chinese) from scored runs: methods, accuracy, paired tests, speed, resources, reproduction."""
from __future__ import annotations

import datetime as _dt
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

from .datasets import SOURCES, load
from .prompts import PROMPT_VERSION, TEMPLATES
from .score import compare, score_run

DATASET_TITLES = {
    "medqa_usmle": "MedQA-USMLE（英文，4 選 1）",
    "medqa_twmle": "MedQA-TWMLE（繁體中文，台灣醫師國考，4 選 1）",
    "pubmedqa": "PubMedQA-L（英文，摘要閱讀，yes/no/maybe）",
    "medmcqa": "MedMCQA validation（英文，印度醫學入學，單選 4 選 1）",
}


def _p(x: Optional[float], d: int = 1) -> str:
    return "—" if x is None else ("%." + str(d) + "f%%") % (100 * x)


def _f(x: Optional[float], d: int = 1) -> str:
    return "—" if x is None else ("%." + str(d) + "f") % x


def _gib(mib: Optional[float]) -> str:
    return "—" if mib is None else "%.1f GiB" % (mib / 1024)


def _pv(p: float) -> str:
    return "<0.001" if p < 0.001 else "%.3f" % p


def _ci(d: Dict[str, Any]) -> str:
    return "%s [%s–%s]" % (_p(d["acc"]), _p(d["ci95"][0]), _p(d["ci95"][1]))


def write_report(runs: List[Path], data_dir: Path, out: Path, title: str, notes: Optional[Path] = None) -> Path:
    S = [score_run(r, data_dir) for r in runs]
    labels = [s["label"] for s in S]
    names = [n for n in S[0]["datasets"] if all(n in s["datasets"] for s in S)]
    L: List[str] = []
    w = L.append

    w("# %s" % title)
    w("")
    w("產生時間：%s（medbench %s，prompt %s）  " % (_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), S[0]["manifest"]["medbench_version"], PROMPT_VERSION))
    w("受測系統：%s" % "、".join("**%s**" % l for l in labels))
    w("")

    # ---- abstract
    w("## 摘要")
    w("")
    tot = sum(S[0]["datasets"][n]["n"] for n in names)
    w("以 %d 題公開醫學題庫（%s）在單張 RTX 4090 上做零樣本（zero-shot）評測，同一份題目、提示詞、取樣參數與逐題種子。" % (
        tot, "、".join("%s %d 題" % (n, S[0]["datasets"][n]["n"]) for n in names)))
    w("")
    for s in S:
        o = s["overall"]
        res = s.get("resources", {})
        w("- **%s**：整體正確率 %s（%d/%d）；解碼中位數 %s tok/s；每題中位延遲 %s s；吞吐 %s 題/小時；GPU 能耗 %s Wh/題。" % (
            s["label"], _ci(o), o["correct"], o["n"], _f(o["decode_tps_median"]),
            _f(o["latency_s_median"]), _f(o["questions_per_hour"], 0), _f(res.get("gpu_energy_wh_per_question"), 2)))
    if len(S) >= 2:
        c = compare(runs[0], runs[1])
        if "ALL" in c:
            a = c["ALL"]
            w("- 配對比較（%s − %s）：差 %+.1f 個百分點，bootstrap 95%% CI [%+.1f, %+.1f]，McNemar 精確檢定 p = %s（不一致題 %d vs %d）。" % (
                labels[0], labels[1], 100 * a["diff_a_minus_b"], 100 * a["diff_ci95"][0], 100 * a["diff_ci95"][1], _pv(a["mcnemar_p"]),
                a["only_a"], a["only_b"]))
    w("")

    # ---- methods
    w("## 1. 方法")
    w("")
    w("### 1.1 資料集")
    w("")
    w("全部從 Hugging Face 固定 revision 下載，下載後比對 SHA-256，題數不符即中止。")
    w("")
    w("| 資料集 | 來源（repo@revision / 檔案） | 題數 | SHA-256（原始 parquet） |")
    w("|---|---|---:|---|")
    for n in names:
        src = SOURCES[n]
        w("| %s | `%s@%s` / `%s` | %d | `%s…` |" % (DATASET_TITLES.get(n, n), src.repo, src.revision[:10], src.path, S[0]["datasets"][n]["n"], src.sha256[:16]))
    w("")
    notes_ds = {
        "medqa_usmle": "MedQA-USMLE：test split 全部 1,273 題（4 選項版本），依 `meta_info` 分 Step 1 與 Step 2&3。",
        "medqa_twmle": "MedQA-TWMLE：test split 全部 1,413 題。題幹以「承上題」等開頭、需要前一題情境的題目另列子集（本評測一律單題作答，不提供前題）。",
        "pubmedqa": "PubMedQA-L：專家標註 1,000 題，提供題目＋摘要內文（不含 long answer 結論段），答 yes/no/maybe；另報告 macro-F1。",
        "medmcqa": "MedMCQA：validation split 的單選題。",
    }
    for n in names:
        if n in notes_ds:
            w("- " + notes_ds[n])
    w("")
    w("### 1.2 作答協定")
    w("")
    w("- 單一 user 訊息、零樣本、不給範例；模型思考（reasoning）開啟，effort 一律 **medium**。")
    w("- 只評分最終回答（`content`），不看思考內容。解析規則：取最後一個 `Answer: X`（MCQ 為 A–D，PubMedQA 為 yes/no/maybe）；找不到視為答錯並計入「無答案」。")
    w("- 每題用由題目 ID 雜湊得到的固定 seed；各系統的取樣參數見 1.4。")
    w("- 被截斷（`finish_reason=length`）照樣解析，另外計數。HTTP/串流錯誤會重試 2 次，仍失敗記為錯誤且算答錯。")
    w("")
    w("提示詞範本（`medbench/prompts.py`，版本 %s）：" % PROMPT_VERSION)
    w("")
    for k, v in TEMPLATES.items():
        w("<details><summary>%s</summary>" % k)
        w("")
        w("```text")
        w(v.strip())
        w("```")
        w("")
        w("</details>")
        w("")
    w("### 1.3 統計方法")
    w("")
    w("- 正確率 95% 信賴區間：Wilson score interval。")
    w("- 兩系統配對比較：只用兩邊都有作答紀錄的同一批題目；McNemar 精確檢定（雙尾二項檢定，看不一致題）；正確率差的 95% CI 用配對 bootstrap（10,000 次，seed 0）。")
    w("- 速度：每題時間從送出請求到串流結束（含排隊與 prefill）；TTFT 為第一個（思考或回答）token 的時間；解碼速度 = (completion tokens − 1) / (最後 token 時間 − 第一 token 時間)。")
    w("- 資源：每秒取樣一次 `nvidia-smi`（VRAM、使用率、功耗、溫度）、`/proc/meminfo`、`/proc/stat` 與伺服器行程 RSS。每段執行前先記錄 15 秒閒置作為功耗基準；能耗以梯形積分計算。")
    w("")
    w("### 1.4 受測系統與硬體")
    w("")
    host = S[0]["manifest"]["host"] or {}
    w("硬體：%s；CPU %s；RAM %s GiB；%s。" % (host.get("gpu", "—"), host.get("cpu", "—"), host.get("ram_gib", "—"), host.get("platform", "—")))
    w("")
    w("| 項目 | %s |" % " | ".join(labels))
    w("|---|%s|" % "|".join("---" for _ in S))
    sysrows = [("模型", "model"), ("權重 / 量化", "weights"), ("推論引擎", "engine"), ("放置方式", "placement"), ("取樣參數", "sampling")]
    for title_, key in sysrows:
        w("| %s | %s |" % (title_, " | ".join(str((s["config"].get("system") or {}).get(key, "—")) for s in S)))
    w("| 併發請求數 | %s |" % " | ".join(str(s["config"].get("concurrency", 1)) for s in S))
    w("| 請求額外參數 | %s |" % " | ".join("`%s`" % json.dumps(s["config"].get("request") or {}, ensure_ascii=False) for s in S))
    w("| 執行期間 | %s |" % " | ".join(_sessions(s) for s in S))
    w("")

    # ---- results
    w("## 2. 正確率")
    w("")
    w("| 資料集 | n | 多數類別基準 | %s |" % " | ".join(labels))
    w("|---|---:|---:|%s|" % "|".join("---:" for _ in S))
    for n in names:
        golds = [it["answer"] for it in load(data_dir, n)]
        top = max(set(golds), key=golds.count)
        w("| %s | %d | %s（%s） | %s |" % (n, S[0]["datasets"][n]["n"], _p(golds.count(top) / len(golds)), top,
                                         " | ".join(_ci(s["datasets"][n]) for s in S)))
    w("| **全部** | %d | — | %s |" % (S[0]["overall"]["n"], " | ".join("**%s**" % _ci(s["overall"]) for s in S)))
    w("")
    w("多數類別基準 = 永遠回答該資料集最常見的正確答案時的正確率（隨機猜測的參考線）。中括號為 95% Wilson 信賴區間。")
    w("")
    w("### 2.1 子集")
    w("")
    w("| 資料集 | 子集 | n | %s |" % " | ".join(labels))
    w("|---|---|---:|%s|" % "|".join("---:" for _ in S))
    for n in names:
        for sub, d in S[0]["datasets"][n]["subsets"].items():
            if isinstance(d, dict):
                w("| %s | %s | %d | %s |" % (n, sub, d["n"], " | ".join(_ci(s["datasets"][n]["subsets"][sub]) for s in S)))
            else:
                w("| %s | %s | — | %s |" % (n, sub, " | ".join("%.3f" % s["datasets"][n]["subsets"][sub] for s in S)))
    w("")
    if len(S) >= 2:
        w("### 2.2 配對比較")
        w("")
        for a_i, b_i in combinations(range(len(S)), 2):
            c = compare(runs[a_i], runs[b_i])
            w("**%s（A） vs %s（B）**" % (labels[a_i], labels[b_i]))
            w("")
            w("| 資料集 | n | A | B | A − B（95% CI） | 都對 | 只有 A 對 | 只有 B 對 | 都錯 | McNemar p |")
            w("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
            for n, d in c.items():
                w("| %s | %d | %s | %s | %+.1f [%+.1f, %+.1f] | %d | %d | %d | %d | %s |" % (
                    "**全部**" if n == "ALL" else n, d["n"], _p(d["acc_a"]), _p(d["acc_b"]), 100 * d["diff_a_minus_b"],
                    100 * d["diff_ci95"][0], 100 * d["diff_ci95"][1], d["both"], d["only_a"], d["only_b"], d["neither"], _pv(d["mcnemar_p"])))
            w("")
    w("### 2.3 失敗模式")
    w("")
    w("| 資料集 | 指標 | %s |" % " | ".join(labels))
    w("|---|---|%s|" % "|".join("---:" for _ in S))
    for n in names:
        for key, t in (("no_answer", "無可解析答案"), ("truncated", "達 max_tokens 截斷"), ("errors", "請求錯誤"), ("missing_items", "未作答")):
            w("| %s | %s | %s |" % (n, t, " | ".join(str(s["datasets"][n][key]) for s in S)))
    w("")

    w("### 2.4 答案分佈與混淆")
    w("")
    w("選擇題預測答案分佈（檢查位置偏誤；none = 無可解析答案）：")
    w("")
    w("| 資料集 | 系統 | %s |" % " | ".join(["A", "B", "C", "D", "none"]))
    w("|---|---|%s|" % "|".join("---:" for _ in range(5)))
    for n in names:
        if S[0]["datasets"][n]["gold_distribution"].keys() <= {"A", "B", "C", "D", "E"}:
            w("| %s | 正確答案 | %s |" % (n, " | ".join(str(S[0]["datasets"][n]["gold_distribution"].get(k, 0)) for k in ["A", "B", "C", "D", "none"])))
            for s in S:
                w("| %s | %s | %s |" % (n, s["label"], " | ".join(str(s["datasets"][n]["pred_distribution"].get(k, 0)) for k in ["A", "B", "C", "D", "none"])))
    w("")
    if "pubmedqa" in names:
        w("PubMedQA 混淆矩陣（列 = 正確答案，欄 = 模型回答）：")
        w("")
        for s in S:
            cm = s["datasets"]["pubmedqa"]["confusion"]
            w("**%s**" % s["label"])
            w("")
            w("| 正確 \\ 回答 | yes | no | maybe | none |")
            w("|---|---:|---:|---:|---:|")
            for gl in ("yes", "no", "maybe"):
                w("| %s | %s |" % (gl, " | ".join(str(cm.get(gl, {}).get(k, 0)) for k in ("yes", "no", "maybe", "none"))))
            w("")

    # ---- speed
    w("## 3. 速度")
    w("")
    w("| 資料集 | 指標 | %s |" % " | ".join(labels))
    w("|---|---|%s|" % "|".join("---:" for _ in S))
    for n in names:
        rows = (("輸入 tokens（平均）", lambda d: _f(d["prompt_tokens_mean"], 0)),
                ("輸出 tokens（中位/平均）", lambda d: "%s / %s" % (_f(d["completion_tokens"]["median"], 0), _f(d["completion_tokens"]["mean"], 0))),
                ("TTFT 中位（s）", lambda d: _f(d["ttft_s_median"], 2)),
                ("每題延遲 中位 / P90（s）", lambda d: "%s / %s" % (_f(d["latency_s"]["median"]), _f(d["latency_s"]["p90"]))),
                ("單請求解碼中位（tok/s）", lambda d: _f(d["decode_tps_median"])))
        for t, fn in rows:
            w("| %s | %s | %s |" % (n, t, " | ".join(fn(s["datasets"][n]) for s in S)))
    w("| 全部 | 實際執行時間 | %s |" % " | ".join("%.2f h" % (s["overall"]["wall_s"] / 3600) for s in S))
    w("| 全部 | 總輸出 tokens | %s |" % " | ".join("{:,}".format(s["overall"]["completion_tokens"]) for s in S))
    w("| 全部 | 總吞吐（tok/s，含併發） | %s |" % " | ".join(_f(s["overall"]["throughput_tok_s"]) for s in S))
    w("| 全部 | 題/小時 | %s |" % " | ".join(_f(s["overall"]["questions_per_hour"], 0) for s in S))
    w("")

    # ---- resources
    w("## 4. 資源耗用")
    w("")
    w("| 指標 | %s |" % " | ".join(labels))
    w("|---|%s|" % "|".join("---:" for _ in S))
    R = [s.get("resources") or {} for s in S]

    def g(r: Dict[str, Any], *ks: str) -> Optional[float]:
        for k in ks:
            if not isinstance(r, dict) or r.get(k) is None:
                return None
            r = r[k]
        return r  # type: ignore[return-value]

    rrows = (("GPU VRAM 平均 / 峰值（整張卡）", lambda r: "%s / %s" % (_gib(g(r, "gpu_mem_used_mib", "mean")), _gib(g(r, "gpu_mem_used_mib", "peak")))),
             ("GPU 使用率平均", lambda r: _f(g(r, "gpu_util_pct_mean"), 0) + "%"),
             ("GPU 功耗 閒置 / 平均 / 峰值（W）", lambda r: "%s / %s / %s" % (_f(g(r, "gpu_power_w", "baseline_idle"), 0), _f(g(r, "gpu_power_w", "mean"), 0), _f(g(r, "gpu_power_w", "peak"), 0))),
             ("GPU 能耗總計（Wh）", lambda r: _f(g(r, "gpu_energy_wh"), 0)),
             ("GPU 能耗（扣閒置，Wh）", lambda r: _f(g(r, "gpu_energy_wh_above_idle"), 0)),
             ("GPU 能耗 / 題（Wh）", lambda r: _f(g(r, "gpu_energy_wh_per_question"), 2)),
             ("GPU 能耗 / 答對一題（Wh）", lambda r: _f(g(r, "gpu_energy_wh_per_correct"), 2)),
             ("伺服器行程 RSS 平均 / 峰值", lambda r: "%s / %s" % (_gib(g(r, "server_rss_mib", "mean")), _gib(g(r, "server_rss_mib", "peak")))),
             ("系統 RAM 使用 平均 / 峰值", lambda r: "%s / %s" % (_gib(g(r, "ram_used_mib", "mean")), _gib(g(r, "ram_used_mib", "peak")))),
             ("CPU 使用率平均（全機）", lambda r: _f(g(r, "cpu_pct_mean"), 0) + "%"),
             ("GPU 溫度峰值（°C）", lambda r: _f(g(r, "gpu_temp_c_peak"), 0)),
             ("同時有其他使用者請求的時間比例", lambda r: _p(g(r, "external_load", "share_with_other_requests"))))
    for t, fn in rrows:
        w("| %s | %s |" % (t, " | ".join(fn(r) for r in R)))
    w("")
    w("註：VRAM、功耗與系統 RAM 為整機量測，包含同時常駐的其他服務（見第 6 節）。")
    w("")

    if notes and notes.exists():
        w(notes.read_text(encoding="utf-8").strip())
        w("")

    # ---- reproduce
    w("## 重現步驟")
    w("")
    w("```bash")
    w("git clone https://github.com/u9401066/medqa-bench && cd medqa-bench")
    w("uv venv && uv pip install -e '.[test]'")
    w("medbench prepare                       # 下載、驗證 SHA-256、正規化成 data/*.jsonl")
    for s in S:
        w("medbench run --config configs/%s.json   # 可中斷續跑" % s["config"]["name"])
    w("medbench score %s" % " ".join("results/%s" % s["config"]["name"] for s in S))
    w("medbench report %s --notes reports/notes.md" % " ".join("results/%s" % s["config"]["name"] for s in S))
    w("```")
    w("")
    w("每個 `results/<run>/` 內含 `manifest.json`（設定、git 版本、伺服器 /props、資料雜湊、主機資訊、執行時段）、`predictions.jsonl`（逐題答案與時間）、`raw.jsonl.gz`（完整思考與回答）、`resources.csv`（每秒資源）、`summary.json`。")
    w("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    (out.parent / (out.stem + ".json")).write_text(json.dumps({"runs": S}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return out


def _sessions(s: Dict[str, Any]) -> str:
    ss = [x for x in (s["manifest"].get("sessions") or []) if "end" in x]
    if not ss:
        return "—"
    f = lambda t: _dt.datetime.fromtimestamp(t).strftime("%m-%d %H:%M")  # noqa: E731
    return "；".join("%s→%s" % (f(x["start"]), f(x["end"])) for x in ss)

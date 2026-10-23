"""Formal Word (.docx) report for technical readers: cover with repository link, summary, benchmarks and reference
baselines, protocol, test environment (hardware, software, model files with hashes), results with figures, discussion,
limitations, reproduction and appendices. Optional extra: pip install -e '.[report]'."""
from __future__ import annotations

import datetime as _dt
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from . import docx_figures as figs
from .datasets import SOURCES, load
from .docx_base import DocBuilder
from .prompts import PROMPT_VERSION, TEMPLATES
from .report import _ci, _f, _p, _pv
from .score import compare, load_predictions, score_run

REPO_URL = "https://github.com/u9401066/medqa-bench"
ROOT = Path(__file__).resolve().parents[1]

REFS = {
    "jin2021": ("Jin D, et al. What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset "
                "from Medical Exams. Applied Sciences 2021;11(14):6421. arXiv:2009.13081", "https://arxiv.org/abs/2009.13081"),
    "jin2019": ("Jin Q, et al. PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP-IJCNLP 2019. "
                "arXiv:1909.06146", "https://arxiv.org/abs/1909.06146"),
    "singhal2023": ("Singhal K, et al. Towards Expert-Level Medical Question Answering with Large Language Models. "
                    "arXiv:2305.09617, 2023", "https://arxiv.org/abs/2305.09617"),
    "wilson1927": ("Wilson EB. Probable inference, the law of succession, and statistical inference. "
                   "J Am Stat Assoc 1927;22(158):209–212", None),
    "mcnemar1947": ("McNemar Q. Note on the sampling error of the difference between correlated proportions or "
                    "percentages. Psychometrika 1947;12(2):153–157", None),
    "efron1993": ("Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall/CRC, 1993", None),
}
REF_ORDER = list(REFS)

# Published reference points: accuracy and source. The settings differ from this benchmark (prompting, few-shot,
# ensembles, retrieval, model versions), so they only give the order of magnitude.
BASELINES = {
    "medqa_usmle": [("隨機猜測（4 選 1）", 0.25, None),
                    ("MedQA 原論文最佳方法（檢索＋閱讀模型，英文題）", 0.367, "jin2021"),
                    ("Med-PaLM（首個超過 USMLE 式題目「及格」分數的模型）", 0.672, "singhal2023"),
                    ("Med-PaLM 2（論文最高分，含集成與提示策略）", 0.865, "singhal2023")],
    "medqa_twmle": [("隨機猜測（4 選 1）", 0.25, None),
                    ("MedQA 原論文最佳方法（繁體中文題）", 0.420, "jin2021")],
    "pubmedqa": [("隨機猜測（3 選 1）", 1 / 3, None),
                 ("PubMedQA 原論文最佳模型（BioBERT 多階段微調）", 0.681, "jin2019"),
                 ("單一人類標註者", 0.780, "jin2019")],
}

DS_INFO = {
    "medqa_usmle": ("MedQA-USMLE", "英文，4 選 1", "美國醫師執照考試（USMLE）式題目，test split 全部；依 meta_info 分 Step 1 / Step 2&3", "jin2021"),
    "medqa_twmle": ("MedQA-TWMLE", "繁體中文，4 選 1", "台灣醫師國考題，test split 全部；24 題「承上題」需要前一題情境", "jin2021"),
    "pubmedqa": ("PubMedQA-L", "英文，yes / no / maybe", "PubMed 摘要（去掉結論段）＋研究問題，專家標註的 1,000 題", "jin2019"),
    "medmcqa": ("MedMCQA", "英文，4 選 1", "印度醫學入學考試題，validation split 單選題", None),
}

GLOSSARY = [
    ("零樣本（zero-shot）", "提示詞不附範例題，模型直接作答。"),
    ("思考強度（reasoning effort）", "控制模型在回答前「思考」多長的設定；本評測兩邊都用 medium。"),
    ("MoE（mixture of experts）", "每層有很多「專家」子網路、每個 token 只用其中少數幾個的模型架構；總參數大、每步計算量小。"),
    ("專家快取 / 低 RAM 模式", "Strata 把常用的專家放 VRAM、其餘放系統 RAM，需要時經 PCIe 搬到 GPU。"),
    ("MTP 推測解碼", "模型附帶的草稿頭一次預測多個 token，主模型一次驗證；「草稿接受率」越高越快。"),
    ("PQ2_0（三元量化）", "PrismML 的權重格式，權重只取 −1 / 0 / +1 再乘縮放係數；Bonsai 27B 的權重檔約 7.1 GiB。"),
    ("IQ3_S / GSQ-RCO", "IQ3_S 是 llama.cpp 的 i-quant 量化等級；本模型為 ISTA-DASLab 的 GSQ-RCO 重量化，各層專家混用 IQ2_S～IQ3_S 與 IQ4_NL。"),
    ("TTFT", "從送出請求到收到第一個 token（思考或回答）的時間。"),
    ("prefill / 解碼", "prefill：讀入提示詞；解碼：逐 token 產生輸出。速度單位 tok/s。"),
    ("Wilson 信賴區間", "比例（正確率）的 95% 信賴區間，在樣本數有限或接近 0%/100% 時比常態近似準確。"),
    ("McNemar 精確檢定", "比較兩系統在同一批題目的對錯：只看「一邊對一邊錯」的題目，檢定兩個方向的數量是否相同。"),
    ("配對 bootstrap", "對題目重複抽樣 10,000 次，得到兩系統正確率差的 95% 信賴區間。"),
    ("macro-F1", "對 yes / no / maybe 三類各算 F1 再平均；類別不平衡時比正確率更能反映少數類別的表現。"),
]


def cite(key: str) -> str:
    return "[%d]" % (REF_ORDER.index(key) + 1)


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _utc(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def _n(x: Optional[float], d: int = 0) -> str:
    return "—" if x is None else ("{:,.%df}" % d).format(x)


def write_docx(runs: List[Path], data_dir: Path, out: Path, title: str, notes: Optional[Path] = None,
               env: Optional[Path] = None, repo_url: str = REPO_URL) -> Path:
    S = [score_run(r, data_dir) for r in runs]
    preds = [load_predictions(r) for r in runs]
    names = [n for n in S[0]["datasets"] if all(n in s["datasets"] for s in S)]
    majority, majority_label = {}, {}
    for n in names:
        golds = [it["answer"] for it in load(data_dir, n)]
        majority_label[n] = max(sorted(set(golds)), key=golds.count)
        majority[n] = golds.count(majority_label[n]) / len(golds)
    ctx = {"runs": runs, "S": S, "preds": preds, "names": names, "majority": majority, "majority_label": majority_label, "data_dir": data_dir,
           "labels": [s["label"] for s in S], "repo_url": repo_url, "title": title,
           "env": json.loads(env.read_text(encoding="utf-8")) if env and env.exists() else {},
           "notes": notes.read_text(encoding="utf-8") if notes and notes.exists() else "",
           "results_commit": _git("log", "-1", "--format=%H", "--", "results") or _git("rev-parse", "HEAD") or "unknown",
           "pairs": {(i, j): compare(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))}}
    b = DocBuilder("%s — %s" % (title, " vs ".join(ctx["labels"])))
    _cover(b, ctx)
    _summary(b, ctx)
    _methods(b, ctx)
    _accuracy(b, ctx)
    _speed(b, ctx)
    _resources(b, ctx)
    if ctx["notes"]:
        b.markdown(ctx["notes"])
    _reproduce(b, ctx)
    _appendices(b, ctx)
    props = b.d.core_properties
    props.title, props.subject, props.author = title, " vs ".join(ctx["labels"]), "medqa-bench"
    props.keywords = "MedQA, PubMedQA, LLM benchmark, llama.cpp, Strata"
    props.comments = repo_url
    out.parent.mkdir(parents=True, exist_ok=True)
    b.save(out)
    return out


def _period(S: List[Dict[str, Any]]) -> str:
    ss = [x for s in S for x in s["manifest"]["sessions"] if "end" in x]
    return "%s – %s（UTC）" % (_utc(min(x["start"] for x in ss)), _utc(max(x["end"] for x in ss))) if ss else "—"


def _host_line(h: Dict[str, Any]) -> str:
    virt = h.get("virtualization")
    kind = "%s 虛擬機" % virt.upper() if virt and virt != "none" else "實體機"
    return "%s：%s（PCIe %s）、%s 個 CPU（%s）、RAM %s GiB" % (kind, h.get("gpu", "—"), h.get("gpu_pcie", "—"),
                                                          h.get("logical_cpus", "—"), h.get("cpu", "—"), h.get("ram_gib", "—"))


def _cover(b: DocBuilder, c: Dict[str, Any]) -> None:
    S, names = c["S"], c["names"]
    for _ in range(3):
        b.para("")
    b.para(c["title"], size=26, bold=True, color=b.d.styles["Title"].font.color.rgb, align="c")
    b.para("、".join(c["labels"]).replace("、", " 與 ", 1) if len(S) == 2 else "、".join(c["labels"]), size=13, align="c")
    b.para("正確率、速度與資源耗用比較 · 零樣本 · %s 題" % _n(S[0]["overall"]["n"]), size=11, align="c", space_after=24)
    host = (c["env"].get("host") or S[0]["manifest"]["host"] or {})
    rows = [
        ("文件日期", _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")),
        ("評測期間", _period(S)),
        ("題庫", "、".join("%s %s 題" % (DS_INFO.get(n, (n,))[0], _n(S[0]["datasets"][n]["n"])) for n in names)),
        ("受測系統", "\n".join(c["labels"])),
        ("測試主機", _host_line(host)),
        ("程式、原始資料與結果", "[%s](%s)" % (c["repo_url"], c["repo_url"])),
        ("結果資料版本", "[commit %s](%s/tree/%s)（results/ 的最後一次修改）" % (c["results_commit"][:7], c["repo_url"], c["results_commit"])),
        ("產生方式", "`medbench docx`（medbench %s，提示詞 %s）從 results/ 自動產生；數字與 reports/REPORT.md 相同" % (__version__, PROMPT_VERSION)),
    ]
    b.table(["項目", "內容"], rows, align="ll", widths=[3.6, 13.4], size=9.5)
    b.page_break()
    b.heading("目錄", 1)
    b.headings.pop()
    b.toc_here()
    b.page_break()


def _summary(b: DocBuilder, c: Dict[str, Any]) -> None:
    S, names, labels = c["S"], c["names"], c["labels"]
    b.heading("摘要", 1)
    b.para("本報告在同一台 RTX 4090 主機上，用 %s 題公開醫學題庫比較 %s。兩邊使用同一份題目、提示詞、取樣參數、思考強度與逐題 seed，"
           "評分只看最終回答。所有程式、設定、逐題輸出與每秒資源紀錄都在 [%s](%s)，可依第 8 節完整重現。"
           % (_n(S[0]["overall"]["n"]), "與".join("**%s**" % lab for lab in labels), c["repo_url"], c["repo_url"]))
    rows = [[DS_INFO.get(n, (n,))[0]] + [_ci(s["datasets"][n]) for s in S] for n in names]
    rows.append(["**整體**"] + ["**%s**" % _ci(s["overall"]) for s in S])
    pair_row = None
    if (0, 1) in c["pairs"] and len(S) == 2:
        a = c["pairs"][(0, 1)]["ALL"]
        d, lo, hi = -100 * a["diff_a_minus_b"], -100 * a["diff_ci95"][1], -100 * a["diff_ci95"][0]
        pair_row = (len(rows) + 1, "%s 比 %s %s %.1f 個百分點（95%% CI %.1f–%.1f），McNemar 精確檢定 p %s" % (
            labels[1], labels[0], "高" if d >= 0 else "低", abs(d), min(abs(lo), abs(hi)) if lo * hi > 0 else lo,
            max(abs(lo), abs(hi)) if lo * hi > 0 else hi, "< 0.001" if a["mcnemar_p"] < 0.001 else "= %.3f" % a["mcnemar_p"]))
        rows.append(["配對差（同一批題目）", "", ""])
    rows.append(["無可解析答案 / 截斷"] + ["%d / %d" % (sum(d["no_answer"] for d in s["datasets"].values()),
                                               sum(d["truncated"] for d in s["datasets"].values())) for s in S])
    rows.append(["單請求解碼中位"] + ["%s tok/s（併發 %s）" % (_f(s["overall"]["decode_tps_median"]), s["overall"]["concurrency"]) for s in S])
    rows.append(["每題平均輸出 tokens"] + [_n(s["overall"]["completion_tokens"] / max(1, s["overall"]["n"])) for s in S])
    rows.append(["每題中位延遲"] + ["%s s" % _f(s["overall"]["latency_s_median"]) for s in S])
    rows.append(["吞吐（全部題目耗時）"] + ["%s 題/小時（%.1f h）" % (_n(s["overall"]["questions_per_hour"]), s["overall"]["wall_s"] / 3600) for s in S])
    R = [s.get("resources") or {} for s in S]
    rows.append(["VRAM 平均（整張卡）"] + ["%.1f GiB" % (r["gpu_mem_used_mib"]["mean"] / 1024) if r else "—" for r in R])
    rows.append(["伺服器 RSS / CPU 平均"] + ["%.1f GiB / %.0f%%" % (r["server_rss_mib"]["mean"] / 1024, r["cpu_pct_mean"]) if r else "—" for r in R])
    rows.append(["GPU 能耗 / 題（/ 答對一題）"] + ["%.2f Wh（%.2f Wh）" % (r["gpu_energy_wh_per_question"], r["gpu_energy_wh_per_correct"]) if r else "—" for r in R])
    t = b.table(["指標"] + labels, rows, align="l" + "r" * len(S), widths=[4.4] + [12.6 / len(S)] * len(S), size=9,
                caption="主要結果（中括號為 95% 信賴區間）")
    if pair_row:
        b.merge_row(t, pair_row[0], 1, len(S), pair_row[1], size=9)
    concl = _notes_section(c["notes"], "### 6.4")
    if concl:
        b.heading("結論與建議", 2)
        b.markdown(concl)
    b.page_break()


def _notes_section(notes: str, heading_prefix: str) -> str:
    out, on = [], False
    for line in notes.splitlines():
        if line.startswith(heading_prefix):
            on = True
            continue
        if on and line.startswith("#"):
            break
        if on:
            out.append(line)
    return "\n".join(out).strip()


def _methods(b: DocBuilder, c: Dict[str, Any]) -> None:
    S, names, labels, data_dir = c["S"], c["names"], c["labels"], c["data_dir"]
    b.heading("1. 測試方法與環境", 1)
    b.heading("1.1 基準測試題庫", 2)
    b.para("三個公開題庫全部題目都納入（不抽樣）。資料從 Hugging Face 的固定 revision 下載，下載後比對 SHA-256，題數不符即中止，"
           "所以任何人重跑都會拿到逐位元相同的題目。")
    rows = []
    for n in names:
        src, info = SOURCES[n], DS_INFO.get(n, (n, "", "", None))
        rows.append([info[0], info[1], _n(S[0]["datasets"][n]["n"]), info[2], src.license, cite(info[3]) if info[3] else ""])
    b.table(["題庫", "語言 / 題型", "題數", "內容", "授權", "文獻"], rows, align="llrlll", widths=[2.6, 2.8, 1.3, 7.1, 2.0, 1.2],
            caption="題庫概要")
    rows = []
    for n in names:
        src = SOURCES[n]
        url = "https://huggingface.co/datasets/%s/blob/%s/%s" % (src.repo, src.revision, src.path)
        rows.append([DS_INFO.get(n, (n,))[0], "[%s@%s](%s)\n`%s`" % (src.repo, src.revision[:10], url, src.path), "`%s`" % src.sha256])
    b.table(["題庫", "來源（Hugging Face repo@revision、檔案）", "原始 parquet 的 SHA-256"], rows, align="lll",
            widths=[2.6, 7.4, 7.0], size=7.5, caption="資料來源與雜湊（下載時驗證）")
    rows = []
    for n in names:
        golds = [it["answer"] for it in load(data_dir, n)]
        keys = sorted(set(golds), key=lambda k: (len(k), k)) if n != "pubmedqa" else ["yes", "no", "maybe"]
        rows.append([DS_INFO.get(n, (n,))[0], "、".join("%s %d" % (k, golds.count(k)) for k in keys)])
    b.table(["題庫", "正確答案分佈"], rows, align="ll", widths=[3.0, 14.0], caption="正確答案分佈")

    b.heading("1.2 參考基準", 2)
    b.para("下表列出隨機猜測、多數類別（永遠回答最常見答案）以及原始論文與知名模型的公開數字，作為量級參考。"
           "文獻數字的作答條件（提示詞、few-shot、集成、檢索、選項數、模型版本）與本評測不同，**不能直接比較**；本評測的正式比較只有第 2 節的兩系統配對比較。")
    rows = []
    for n in names:
        title = DS_INFO.get(n, (n,))[0]
        refs = list(BASELINES.get(n, []))
        refs.insert(1, ("多數類別（永遠回答 %s）" % c["majority_label"][n], c["majority"][n], None))
        for name, acc, ref in refs:
            rows.append([title, name, _p(acc), cite(ref) if ref else "本資料計算" if "多數" in name else "—"])
        for s in S:
            rows.append([title, "**本評測：%s**" % s["label"], "**%s**" % _p(s["datasets"][n]["acc"]), "第 2 節"])
    b.table(["題庫", "參考點", "正確率", "出處"], rows, align="llrl", widths=[2.6, 9.6, 2.0, 2.8], caption="參考基準與本評測結果")

    b.heading("1.3 作答協定", 2)
    for t in ("單一 user 訊息、零樣本、不給範例題；模型思考（reasoning）開啟，兩邊的思考強度都是 **medium**。",
              "提示詞要求最後一行以 `Answer: X` 作答（三種範本見附錄 B）。只解析最終回答（`content`）的最後一個 `Answer: X`／`答案：X`；"
              "思考內容不評分。解析不到視為答錯，並計入「無可解析答案」。",
              "每題的 seed = SHA-256(題目 ID) 的前 31 bits，兩邊同一題用同一個 seed；兩個引擎的亂數實作不同，所以不會產生相同的取樣路徑。",
              "`max_tokens` = 16,384（思考＋回答）。達上限而截斷（`finish_reason=length`）的輸出照樣解析並另外計數。",
              "HTTP / 串流錯誤重試 2 次；中斷後重跑同一指令會跳過已完成的題目、重做失敗與進行中的題目，每題只保留一筆結果。"):
        b.bullet(t)
    if all(re.match(r"https?://(127\.0\.0\.1|localhost)[:/]", s["config"]["base_url"]) for s in S):
        b.bullet("用戶端（medbench）與受測伺服器在同一台主機，經 localhost 直接連到各自的 API（%s），不經過其他代理。"
                 % "、".join("%s：`%s`" % (s["label"], s["config"]["base_url"]) for s in S))
    rows = [[lab, str(s["config"].get("concurrency", 1)), "`%s`" % json.dumps(s["config"].get("request") or {}, ensure_ascii=False)]
            for lab, s in zip(labels, S)]
    b.table(["系統", "併發", "每個請求的參數"], rows, align="lcl", widths=[4.2, 1.2, 11.6], size=8, caption="請求參數")

    b.heading("1.4 受測系統與測試環境", 2)
    env = c["env"]
    h = env.get("host") or S[0]["manifest"]["host"] or {}
    rows = [("主機類型", "%s 虛擬機" % h["virtualization"].upper() if h.get("virtualization") not in (None, "none") else "實體機"),
            ("CPU", "%s 個邏輯 CPU（%s）" % (h.get("logical_cpus", "—"), h.get("cpu", "—"))),
            ("記憶體", "RAM %s GiB；swap %s GiB" % (h.get("ram_gib", "—"), h.get("swap_gib", "—"))),
            ("GPU", "%s；PCIe %s（直通）" % (h.get("gpu", "—"), h.get("gpu_pcie", "—"))),
            ("磁碟", "；".join(h.get("disks") or ["—"])),
            ("作業系統", "%s，kernel %s" % (h.get("os", "—"), h.get("kernel", "—"))),
            ("CUDA（driver API）", h.get("cuda_driver_api", "—"))]
    b.table(["項目", "規格"], rows, align="ll", widths=[3.6, 13.4], caption="硬體與作業系統")
    if env.get("software"):
        b.table(["軟體", "版本", "說明"], [[x["name"], x["version"], ("[%s](%s)\n" % (x["url"], x["url"]) if x.get("url") else "") + x.get("note", "")]
                                       for x in env["software"]], align="lll", widths=[4.4, 5.6, 7.0], size=8, caption="軟體版本")
    rows = []
    for key, name in (("model", "模型"), ("weights", "權重 / 量化"), ("engine", "推論引擎"), ("placement", "放置方式"), ("sampling", "取樣參數")):
        rows.append([name] + [str((s["config"].get("system") or {}).get(key, "—")) for s in S])
    rows.append(["伺服器 build_info"] + [str(((s["manifest"].get("server") or {}).get("/props") or {}).get("build_info", "—")) for s in S])
    b.table(["項目"] + labels, rows, align="l" * (1 + len(S)), widths=[2.8] + [14.2 / len(S)] * len(S), size=8, caption="受測系統")
    for m in env.get("models", []):
        url = "https://huggingface.co/%s/tree/%s" % (m["repo"], m["revision"])
        rows = [("來源", "[%s](%s)（revision `%s`）" % (m["repo"], url, m["revision"][:12])), ("基礎模型", m.get("base", "—"))]
        for f in m["files"]:
            rows.append(("檔案", "`%s`\n%s bytes，SHA-256 `%s`" % (f["name"], _n(f["bytes"]), f["sha256"])))
        rows.append(("驗證", m.get("verified", "—")))
        b.table(["項目", "內容"], rows, align="ll", widths=[2.6, 14.4], size=8, caption="模型檔案：%s" % m["system"])
    if env.get("co_resident"):
        b.table(["同機常駐服務", "說明", "評測期間"], [[x["name"], x["detail"], x["during_runs"]] for x in env["co_resident"]],
                align="lll", widths=[4.0, 6.5, 6.5], size=8, caption="評測期間同一台主機上的其他服務",
                note="VRAM 與功耗是整張卡的量測值，包含 Clef-Flash 佔用的約 6.5 GB；伺服器啟動參數見附錄 A。")

    b.heading("1.5 統計與量測方法", 2)
    for t in ("正確率的 95%% 信賴區間：Wilson score interval %s。" % cite("wilson1927"),
              "兩系統比較：只用兩邊都有作答的同一批題目；McNemar 精確檢定（雙尾，看「一邊對一邊錯」的題目）%s；"
              "正確率差的 95%% 信賴區間用配對 bootstrap（10,000 次，seed 0）%s。" % (cite("mcnemar1947"), cite("efron1993")),
              "PubMedQA 另報 macro-F1（yes / no / maybe 各自 F1 的平均）。",
              "延遲：從送出請求到串流結束（含排隊與 prefill）。TTFT：到第一個思考或回答 token。"
              "單請求解碼速度 = (completion tokens − 1) / (最後 token 時間 − 第一 token 時間)。",
              "伺服器端 timings：伺服器在串流最後回報的 llama.cpp 格式計時（prompt / 解碼的 token 數與毫秒、推測解碼的草稿數與接受數），"
              "不含網路與排隊；伺服器沒有回報時以「—」表示（本評測兩個伺服器都有回報）。",
              "資源：每秒取樣 `nvidia-smi`（VRAM、使用率、功耗、溫度）、`/proc/meminfo`、`/proc/stat`、伺服器行程 RSS，以及伺服器 `/slots`"
              "（與我方進行中的請求比較，可看出是否有其他使用者同時請求）。每段執行前記錄 15 秒閒置作為功耗基準；能耗以梯形積分計算，只含 GPU。"):
        b.bullet(t)

    b.heading("1.6 執行時段", 2)
    rows = []
    for lab, s in zip(labels, S):
        for i, ss in enumerate(s["manifest"]["sessions"]):
            if "end" not in ss:
                continue
            rows.append([lab, str(i + 1), _utc(ss["start"]), _utc(ss["end"]) + ("（推得）" if ss.get("end_inferred") else ""),
                         "%.2f h" % ((ss["end"] - ss["start"]) / 3600), _n(ss.get("todo")), (ss.get("git_rev") or "—")[:7]])
    b.table(["系統", "段", "開始（UTC）", "結束（UTC）", "時長", "待答題數", "medbench commit"], rows, align="lcllrrl",
            widths=[4.4, 0.8, 2.8, 3.4, 1.6, 1.8, 2.2], size=8, caption="執行時段", note=_sessions_note(S))


def _sessions_note(S: List[Dict[str, Any]]) -> Optional[str]:
    ss = [x for s in S for x in s["manifest"]["sessions"] if "end" in x]
    out = []
    if any(x.get("end_inferred") for x in ss):
        out.append("「推得」：該段程序被中止、沒有寫入結束時間，結束時間取該段最後完成的題目；中止時進行中的題目不記錄，續跑時重做。")
    if any(not x.get("git_rev") for x in ss):
        out.append("「—」：該段開始時 medbench 還沒有記錄每段的 commit；各段使用的提示詞版本記錄在 manifest.json。")
    return "".join(out) or None


def _accuracy(b: DocBuilder, c: Dict[str, Any]) -> None:
    S, names, labels = c["S"], c["names"], c["labels"]
    b.page_break()
    b.heading("2. 正確率", 1)
    rows = [[DS_INFO.get(n, (n,))[0], _n(S[0]["datasets"][n]["n"]), _p(c["majority"][n])] + [_ci(s["datasets"][n]) for s in S] for n in names]
    rows.append(["**全部**", _n(S[0]["overall"]["n"]), "—"] + ["**%s**" % _ci(s["overall"]) for s in S])
    b.table(["題庫", "n", "多數類別基準"] + labels, rows, align="lrr" + "r" * len(S), widths=[3.0, 1.4, 2.6] + [10.0 / len(S)] * len(S),
            size=9, caption="正確率（中括號為 95% Wilson 信賴區間）")
    b.figure(figs.accuracy(S, names, c["majority"]), "各題庫正確率與 95% 信賴區間；虛線為多數類別基準")

    b.heading("2.1 子集", 2)
    rows = []
    for n in names:
        for sub, d in S[0]["datasets"][n]["subsets"].items():
            if isinstance(d, dict):
                rows.append([DS_INFO.get(n, (n,))[0], sub, _n(d["n"])] + [_ci(s["datasets"][n]["subsets"][sub]) for s in S])
            else:
                rows.append([DS_INFO.get(n, (n,))[0], sub, "—"] + ["%.3f" % s["datasets"][n]["subsets"][sub] for s in S])
    b.table(["題庫", "子集", "n"] + labels, rows, align="llr" + "r" * len(S), widths=[2.8, 4.2, 1.2] + [8.8 / len(S)] * len(S),
            size=8.5, caption="子集正確率")

    b.heading("2.2 配對比較", 2)
    for (i, j), comp in c["pairs"].items():
        b.para("A = **%s**，B = **%s**。只用兩邊都有作答的題目。" % (labels[i], labels[j]))
        rows = [["**全部**" if n == "ALL" else DS_INFO.get(n, (n,))[0], _n(d["n"]), _p(d["acc_a"]), _p(d["acc_b"]),
                 "%+.1f [%+.1f, %+.1f]" % (100 * d["diff_a_minus_b"], 100 * d["diff_ci95"][0], 100 * d["diff_ci95"][1]),
                 _n(d["both"]), _n(d["only_a"]), _n(d["only_b"]), _n(d["neither"]), _pv(d["mcnemar_p"])]
                for n, d in sorted(comp.items(), key=lambda kv: (names + ["ALL"]).index(kv[0]) if kv[0] in names + ["ALL"] else 99)]
        b.table(["題庫", "n", "A", "B", "A − B（95% CI）", "都對", "只 A 對", "只 B 對", "都錯", "McNemar p"], rows,
                align="lrrrrrrrrr", widths=[2.6, 1.2, 1.4, 1.4, 3.2, 1.3, 1.5, 1.5, 1.3, 1.6], size=8, caption="配對比較")

    b.heading("2.3 失敗模式", 2)
    rows = []
    for n in names:
        for key, t in (("no_answer", "無可解析答案"), ("truncated", "達 max_tokens 截斷"), ("errors", "請求錯誤"), ("missing_items", "未作答")):
            rows.append([DS_INFO.get(n, (n,))[0], t] + [str(s["datasets"][n][key]) for s in S])
    b.table(["題庫", "指標"] + labels, rows, align="ll" + "r" * len(S), widths=[3.0, 4.0] + [10.0 / len(S)] * len(S), size=8.5,
            caption="失敗模式（題數）")

    b.heading("2.4 答案分佈與混淆", 2)
    keys = ["A", "B", "C", "D", "none"]
    rows = []
    for n in names:
        if set(S[0]["datasets"][n]["gold_distribution"]) <= {"A", "B", "C", "D", "E"}:
            rows.append([DS_INFO.get(n, (n,))[0], "正確答案"] + [str(S[0]["datasets"][n]["gold_distribution"].get(k, 0)) for k in keys])
            for s in S:
                rows.append([DS_INFO.get(n, (n,))[0], s["label"]] + [str(s["datasets"][n]["pred_distribution"].get(k, 0)) for k in keys])
    if rows:
        b.table(["題庫", "來源"] + keys, rows, align="ll" + "r" * 5, widths=[2.8, 6.2] + [1.6] * 5, size=8.5,
                caption="選擇題的答案分佈（檢查位置偏誤；none = 無可解析答案）")
    if "pubmedqa" in names:
        for s in S:
            cm = s["datasets"]["pubmedqa"]["confusion"]
            rows = [[g] + [str(cm.get(g, {}).get(k, 0)) for k in ("yes", "no", "maybe", "none")] for g in ("yes", "no", "maybe")]
            b.table(["正確答案 \\ 模型回答", "yes", "no", "maybe", "none"], rows, align="lrrrr", widths=[4.4, 2.0, 2.0, 2.0, 2.0],
                    size=8.5, caption="PubMedQA 混淆矩陣：%s" % s["label"])


def _speed(b: DocBuilder, c: Dict[str, Any]) -> None:
    S, names, labels = c["S"], c["names"], c["labels"]
    b.heading("3. 速度", 1)
    rows = []
    for n in names:
        title = DS_INFO.get(n, (n,))[0]
        for t, fn in (("輸入 tokens（平均）", lambda d: _f(d["prompt_tokens_mean"], 0)),
                      ("輸出 tokens（中位 / 平均）", lambda d: "%s / %s" % (_f(d["completion_tokens"]["median"], 0), _f(d["completion_tokens"]["mean"], 0))),
                      ("TTFT 中位（s）", lambda d: _f(d["ttft_s_median"], 2)),
                      ("每題延遲 中位 / P90（s）", lambda d: "%s / %s" % (_f(d["latency_s"]["median"]), _f(d["latency_s"]["p90"]))),
                      ("單請求解碼中位（tok/s）", lambda d: _f(d["decode_tps_median"])),
                      ("伺服器端 prefill / 解碼（tok/s）", lambda d: "%s / %s" % (_f((d.get("server_timings") or {}).get("prefill_tps_median"), 0),
                                                                          _f((d.get("server_timings") or {}).get("decode_tps_median")))),
                      ("推測解碼草稿接受率", lambda d: _p((d.get("server_timings") or {}).get("draft_acceptance")))):
            rows.append([title, t] + [fn(s["datasets"][n]) for s in S])
    b.table(["題庫", "指標"] + labels, rows, align="ll" + "r" * len(S), widths=[2.8, 5.0] + [9.2 / len(S)] * len(S), size=8.5,
            caption="各題庫的速度")
    rows = [["併發請求數"] + [str(s["overall"]["concurrency"]) for s in S],
            ["實際執行時間"] + ["%.2f h" % (s["overall"]["wall_s"] / 3600) for s in S],
            ["總輸出 tokens"] + [_n(s["overall"]["completion_tokens"]) for s in S],
            ["總吞吐（tok/s，含併發）"] + [_f(s["overall"]["throughput_tok_s"]) for s in S],
            ["題 / 小時"] + [_n(s["overall"]["questions_per_hour"]) for s in S]]
    b.table(["全部題目"] + labels, rows, align="l" + "r" * len(S), widths=[5.0] + [12.0 / len(S)] * len(S), size=9, caption="整體吞吐")
    b.figure(figs.speed(c["preds"], labels, names), "每題輸出 tokens（左）與每題延遲（右）的分佈")
    b.note("單請求解碼以用戶端收到的串流計時（含網路與排隊）；伺服器端數字為 llama.cpp 格式的 timings。兩邊的併發設定不同（見第 5 節），"
           "比較速度時請同時看單請求解碼速度與題 / 小時。")


def _resources(b: DocBuilder, c: Dict[str, Any]) -> None:
    S, labels = c["S"], c["labels"]
    b.heading("4. 資源耗用", 1)
    R = [s.get("resources") or {} for s in S]

    def g(r: Dict[str, Any], *ks: str) -> Optional[float]:
        for k in ks:
            if not isinstance(r, dict) or r.get(k) is None:
                return None
            r = r[k]
        return r  # type: ignore[return-value]

    def gib(x: Optional[float]) -> str:
        return "—" if x is None else "%.1f GiB" % (x / 1024)

    rows = (("GPU VRAM 平均 / 峰值（整張卡）", lambda r: "%s / %s" % (gib(g(r, "gpu_mem_used_mib", "mean")), gib(g(r, "gpu_mem_used_mib", "peak")))),
            ("GPU 使用率平均", lambda r: _f(g(r, "gpu_util_pct_mean"), 0) + "%"),
            ("GPU 功耗 閒置 / 平均 / 峰值（W）", lambda r: "%s / %s / %s" % (_f(g(r, "gpu_power_w", "baseline_idle"), 0), _f(g(r, "gpu_power_w", "mean"), 0), _f(g(r, "gpu_power_w", "peak"), 0))),
            ("GPU 能耗總計 / 扣除閒置（Wh）", lambda r: "%s / %s" % (_n(g(r, "gpu_energy_wh")), _n(g(r, "gpu_energy_wh_above_idle")))),
            ("GPU 能耗 / 題（Wh）", lambda r: _f(g(r, "gpu_energy_wh_per_question"), 2)),
            ("GPU 能耗 / 答對一題（Wh）", lambda r: _f(g(r, "gpu_energy_wh_per_correct"), 2)),
            ("伺服器行程 RSS 平均 / 峰值", lambda r: "%s / %s" % (gib(g(r, "server_rss_mib", "mean")), gib(g(r, "server_rss_mib", "peak")))),
            ("系統 RAM 使用 平均 / 峰值", lambda r: "%s / %s" % (gib(g(r, "ram_used_mib", "mean")), gib(g(r, "ram_used_mib", "peak")))),
            ("CPU 使用率平均（全機）", lambda r: _f(g(r, "cpu_pct_mean"), 0) + "%"),
            ("GPU 溫度峰值（°C）", lambda r: _f(g(r, "gpu_temp_c_peak"), 0)),
            ("同時有其他使用者請求的時間比例", lambda r: _p(g(r, "external_load", "share_with_other_requests"))))
    b.table(["指標"] + labels, [[t] + [fn(r) for r in R] for t, fn in rows], align="l" + "r" * len(S),
            widths=[6.0] + [11.0 / len(S)] * len(S), size=9, caption="資源耗用（評測期間每秒取樣）",
            note="VRAM、功耗與系統 RAM 為整機量測，包含同機常駐的其他服務（第 1.4 節）；能耗只計 GPU，不含 CPU 與其他元件。")
    h = c["env"].get("host") or S[0]["manifest"]["host"] or {}
    vram = 24.0
    try:
        vram = float(str(h.get("gpu", "")).split(",")[1].split()[0]) / 1024
    except (IndexError, ValueError):
        pass
    b.figure(figs.resources(c["runs"], S, vram, float(h.get("ram_gib") or 64)), "評測期間的 GPU 功耗、整張卡 VRAM 與系統 RAM（每分鐘平均）")


def _reproduce(b: DocBuilder, c: Dict[str, Any]) -> None:
    S, repo = c["S"], c["repo_url"]
    b.heading("8. 重現步驟與資料檔案", 1)
    b.para("全部程式、設定、逐題結果與資源紀錄都在 [%s](%s)（公開）。需要 Python ≥ 3.10、能連 Hugging Face，以及一個 OpenAI 相容的"
           "串流 API 伺服器（llama.cpp、Strata、vLLM 等）；要重跑本報告的兩個系統，需要 24 GB VRAM 的 NVIDIA GPU，Strata 另需約 55 GB RAM。" % (repo, repo))
    cmds = ["git clone %s && cd medqa-bench" % repo,
            "uv venv && uv pip install -e '.[test,report]'   # 或 python -m venv .venv && pip install -e '.[test,report]'",
            "medbench prepare            # 下載、驗證 SHA-256、正規化成 data/*.jsonl"]
    for s in S:
        launch = s["config"].get("launch")
        if launch:
            cmds.append("# %s 需要整張 GPU：暫停正式服務、啟動受測伺服器、跑完自動恢復" % s["label"])
            cmds.append("PARALLEL=%s STRATA_CONFIG=%s scripts/strata_window.sh configs/%s.json" % (
                launch.get("parallel", 1), launch.get("engine_config", "<engine config>"), s["config"]["name"]))
        else:
            cmds.append("medbench run --config configs/%s.json   # 可中斷續跑" % s["config"]["name"])
    paths = " ".join("results/%s" % s["config"]["name"] for s in S)
    cmds += ["medbench score %s" % paths,
             "medbench report %s --notes reports/notes.md" % paths,
             "medbench docx %s --notes reports/notes.md --env reports/environment.json" % paths]
    b.code("\n".join(cmds))
    b.para("驗證方式：在全新 clone 執行 `medbench prepare`，資料雜湊會與表 2 相同；不必重跑模型，直接執行 `medbench report` / `medbench docx`，"
           "就會從 repo 內的逐題結果重新算出本文件的所有數字（本文件發布前已用全新 clone 如此驗證）。")
    desc = {"manifest.json": "設定、medbench 版本與 commit、提示詞版本、資料雜湊、主機資訊、伺服器 /props、每段執行時間",
            "predictions.jsonl": "每題一行：答案、對錯、TTFT、延遲、tokens、解碼速度、finish reason、伺服器 timings",
            "raw.jsonl.gz": "每題完整的思考內容與最終回答（gzip）",
            "resources.csv": "每秒一列：VRAM、GPU 使用率 / 功耗 / 溫度、系統 RAM、伺服器 RSS、CPU、進行中請求、伺服器忙碌槽位",
            "summary.json": "`medbench score` 的統計結果"}
    rows = []
    for run in c["runs"]:
        for fn, d in desc.items():
            p = run / fn
            size = p.stat().st_size if p.exists() else None
            rows.append(["`results/%s/%s`" % (run.name, fn), "—" if size is None else "%.1f MB" % (size / 1e6) if size >= 1e6
                         else "%.0f KB" % (size / 1e3), d])
    rows.append(["`reports/REPORT.md` / `REPORT.json`", "", "Markdown 版報告與全部統計數字"])
    rows.append(["`reports/environment.json`", "", "測試環境、模型檔案來源與雜湊、軟體版本、同機服務、啟動參數"])
    rows.append(["`results/logs/*.log.gz`", "", "Strata 伺服器 log（試跑與正式執行）"])
    b.table(["檔案", "大小", "內容"], rows, align="lrl", widths=[6.6, 1.6, 8.8], size=8, caption="repo 內的結果檔案")


def _appendices(b: DocBuilder, c: Dict[str, Any]) -> None:
    S = c["S"]
    b.page_break()
    b.heading("附錄 A：設定檔與啟動參數", 1)
    for s in S:
        b.heading("%s：medbench 設定（configs/%s.json）" % (s["label"], s["config"]["name"]), 2)
        b.code(json.dumps(s["config"], indent=1, ensure_ascii=False))
        eng = (s["config"].get("launch") or {}).get("engine_config")
        if eng and (ROOT / eng).exists():
            b.heading("%s：Strata engine 設定（%s）" % (s["label"], eng), 3)
            b.code((ROOT / eng).read_text(encoding="utf-8"))
    launch = c["env"].get("launch") or {}
    if launch:
        b.heading("伺服器啟動參數", 2)
        for name, cmd in launch.items():
            b.para("**%s**" % name)
            b.code(re.sub(r" (?=--?[A-Za-z])", "\n  ", cmd) if len(cmd) > 160 else cmd)

    b.heading("附錄 B：提示詞範本（%s）" % PROMPT_VERSION, 1)
    b.para("`{question}`、`{options}`（每行「A. …」）、`{letters}`、`{context}` 由題目代入；三種範本都要求最後一行以 `Answer: X` 作答。")
    for k, v in TEMPLATES.items():
        b.heading(k, 3)
        b.code(v.strip(), size=8)

    b.heading("附錄 C：名詞解釋", 1)
    b.table(["名詞", "說明"], GLOSSARY, align="ll", widths=[4.4, 12.6], size=8.5)

    b.heading("附錄 D：參考文獻與連結", 1)
    for i, key in enumerate(REF_ORDER):
        text, url = REFS[key]
        b.numbered("[%d]" % (i + 1), text + (" [%s](%s)" % (url, url) if url else ""))
    b.para("")
    links = [("本評測程式與結果", c["repo_url"])]
    for n in c["names"]:
        src = SOURCES[n]
        links.append(("題庫 %s" % DS_INFO.get(n, (n,))[0], "https://huggingface.co/datasets/%s" % src.repo))
    for m in c["env"].get("models", []):
        links.append(("模型 %s" % m["system"], "https://huggingface.co/%s" % m["repo"]))
    for x in c["env"].get("software", []):
        if x.get("url") and x["url"] != c["repo_url"]:
            links.append((x["name"], x["url"]))
    links.append(("llama.cpp（上游）", "https://github.com/ggml-org/llama.cpp"))
    seen = set()
    for name, url in links:
        if url in seen:
            continue
        seen.add(url)
        b.bullet("%s：[%s](%s)" % (name, url, url))

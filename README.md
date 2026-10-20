# medqa-bench

可重現的醫學問答評測流程，適用於任何 OpenAI 相容（`/v1/chat/completions` 串流）的本地 LLM 伺服器（llama.cpp、Strata、vLLM…）。

- **固定資料**：3 個公開題庫共 3,686 題，從 Hugging Face 固定 revision 下載並比對 SHA-256。
- **同一協定**：零樣本、單一 user 訊息、思考開啟、相同取樣參數、逐題固定 seed、只評分最終回答。
- **可續跑**：每題完成就寫入 `predictions.jsonl`；中斷後重跑同一指令會跳過已完成題、重試失敗題。
- **完整紀錄**：設定、程式 git 版本、伺服器 `/props`、資料雜湊、主機資訊、執行時段寫進 `manifest.json`；完整思考與回答在 `raw.jsonl`。
- **統計**：Wilson 95% CI、McNemar 精確檢定、配對 bootstrap 差異 CI、子集分析、PubMedQA macro-F1。
- **速度與資源**：TTFT、延遲、單請求解碼速度、總吞吐；每秒取樣 VRAM／GPU 使用率／功耗／溫度、伺服器 RSS、系統 RAM、CPU，計算能耗（Wh/題，含扣除閒置功耗），並記錄共用伺服器上其他使用者的請求。
- **報告**：`medbench report` 產生正式 Markdown 報告（方法、結果、配對比較、速度、資源、重現步驟）。

最新報告：[reports/REPORT.md](reports/REPORT.md)

## 資料集

| 名稱 | 內容 | 題數 | 來源（固定 revision） | 授權 |
|---|---|---:|---|---|
| `medqa_usmle` | MedQA 美國醫師執照考試，英文，4 選 1（Step 1：679；Step 2&3：594） | 1,273 | `bigbio/med_qa@484a6c0` `med_qa_en_4options_source/test` | MIT |
| `medqa_twmle` | MedQA 台灣醫師國考，繁體中文，4 選 1（24 題為「承上題」） | 1,413 | `bigbio/med_qa@484a6c0` `med_qa_tw_source/test` | MIT |
| `pubmedqa` | PubMedQA-L 專家標註，英文摘要閱讀，yes/no/maybe（552/338/110） | 1,000 | `qiaojin/PubMedQA@9001f28` `pqa_labeled` | MIT |
| `medmcqa`（選用） | MedMCQA validation 單選題，英文 | 2,816 | `openlifescienceai/medmcqa` | Apache-2.0 |

引用：Jin et al., *What Disease does this Patient Have?*（MedQA），Applied Sciences 2021；Jin et al., *PubMedQA*，EMNLP 2019。

## 安裝與使用

```bash
git clone https://github.com/u9401066/medqa-bench && cd medqa-bench
uv venv && uv pip install -e '.[test]'     # 或 python -m venv .venv && pip install -e '.[test]'
.venv/bin/pytest -q                         # 31 個測試（解析器、統計、假伺服器端到端）

medbench prepare                                                    # 下載、驗證、正規化到 data/
medbench run --config configs/bonsai27b-pq2-medium.json --sample 10 --out results/pilot-bonsai   # 先試跑
medbench run --config configs/bonsai27b-pq2-medium.json             # 全部 3,686 題，可中斷續跑
medbench score results/bonsai27b-pq2-medium results/strata-flashnext-iq3s-medium
medbench report results/bonsai27b-pq2-medium results/strata-flashnext-iq3s-medium --notes reports/notes.md
```

### 設定檔（`configs/*.json`）

| 欄位 | 說明 |
|---|---|
| `name` / `label` | 結果資料夾名稱 / 報告中的名稱 |
| `base_url`、`model`、`headers` | OpenAI 相容伺服器 |
| `concurrency` | 同時送出的請求數 |
| `request` | 併入每個請求的參數（`max_tokens`、取樣參數、`reasoning_effort` 或 `chat_template_kwargs`） |
| `server_info` | 執行前讀取並存進 manifest 的端點（例如 `/props`） |
| `processes` | 用來計算 RSS 的伺服器行程命令列片段 |
| `slots_path` | 伺服器忙碌槽位端點（預設 `/slots`；設為 `null` 停用） |
| `system` | 報告用的模型／量化／引擎描述 |

### 輸出（`results/<name>/`）

| 檔案 | 內容 |
|---|---|
| `manifest.json` | 設定、medbench 版本與 git commit、prompt 版本、資料雜湊、主機、伺服器資訊、每段執行的時間 |
| `predictions.jsonl` | 每題：答案、對錯、TTFT、延遲、tokens、解碼速度、finish reason、伺服器 timings |
| `raw.jsonl(.gz)` | 每題完整思考與回答 |
| `resources.csv` | 每秒：VRAM、GPU 使用率／功耗／溫度、系統 RAM、伺服器 RSS、CPU、我方進行中請求、伺服器忙碌槽位 |
| `summary.json` | `medbench score` 的結果 |

## 作答與評分協定

- 提示詞見 [medbench/prompts.py](medbench/prompts.py)（`PROMPT_VERSION = "v1"`），三種範本（英文 MCQ、繁中 MCQ、PubMedQA），都要求最後一行 `Answer: X`。
- 只解析最終回答（`content`）的最後一個 `Answer: X`／`答案：X`；思考內容不評分。解析不到 = 答錯，另計「無答案」。
- 逐題 seed = SHA-256(題目 ID) 前 31 bits；同一題在每個系統用同一 seed。
- `finish_reason=length`（截斷）仍解析並另外計數；請求錯誤重試 2 次，續跑時再重試。

## 一次佔滿 GPU 的模型：維護時段腳本

[scripts/strata_window.sh](scripts/strata_window.sh) 會暫停正式服務、啟動受測伺服器、跑 medbench，**無論成功、失敗或被中斷都會停掉受測伺服器並恢復正式服務**：

```bash
PARALLEL=3 STRATA_CONFIG=configs/engine/strata-iq3_s.json scripts/strata_window.sh configs/strata-flashnext-iq3s-medium.json
```

## 限制

- 題庫公開多年，可能出現在模型訓練資料中（資料汙染），分數宜視為相對比較而非絕對能力。
- 單次取樣（非多次平均）；溫度 1.0 下單題結果有隨機性，配對檢定已反映此不確定性。
- 共用伺服器上的速度會受其他使用者影響；`resources.csv` 的 `server_busy` 與 `inflight` 可量化此影響。
- 零樣本、無外部工具（無檢索、無 MCP），測的是模型本身的醫學知識與推理。

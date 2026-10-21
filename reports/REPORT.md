# 醫學問答評測：單張 RTX 4090 上的本地模型

產生時間：2026-10-21 11:15（medbench 0.1.0，prompt v1）  
受測系統：**Bonsai 27B PQ2_0**、**Strata Qwen3.8-Flash-Next 125B IQ3_S**

## 摘要

以 3686 題公開醫學題庫（medqa_usmle 1273 題、medqa_twmle 1413 題、pubmedqa 1000 題）在單張 RTX 4090 上做零樣本（zero-shot）評測，同一份題目、提示詞、取樣參數與逐題種子。

- **Bonsai 27B PQ2_0**：整體正確率 78.3% [77.0%–79.6%]（2887/3686）；解碼中位數 81.8 tok/s；每題中位延遲 13.7 s；吞吐 606 題/小時；GPU 能耗 0.56 Wh/題。
- **Strata Qwen3.8-Flash-Next 125B IQ3_S**：整體正確率 88.0% [86.9%–89.0%]（3244/3686）；解碼中位數 87.6 tok/s；每題中位延遲 10.6 s；吞吐 308 題/小時；GPU 能耗 0.67 Wh/題。
- 配對比較（Bonsai 27B PQ2_0 − Strata Qwen3.8-Flash-Next 125B IQ3_S）：差 -9.7 個百分點，bootstrap 95% CI [-10.9, -8.5]，McNemar 精確檢定 p = <0.001（不一致題 111 vs 468）。

## 1. 方法

### 1.1 資料集

全部從 Hugging Face 固定 revision 下載，下載後比對 SHA-256，題數不符即中止。

| 資料集 | 來源（repo@revision / 檔案） | 題數 | SHA-256（原始 parquet） |
|---|---|---:|---|
| MedQA-USMLE（英文，4 選 1） | `bigbio/med_qa@484a6c066f` / `med_qa_en_4options_source/test-00000-of-00001.parquet` | 1273 | `b50b49fad2642c10…` |
| MedQA-TWMLE（繁體中文，台灣醫師國考，4 選 1） | `bigbio/med_qa@484a6c066f` / `med_qa_tw_source/test-00000-of-00001.parquet` | 1413 | `9756921fcff39ee4…` |
| PubMedQA-L（英文，摘要閱讀，yes/no/maybe） | `qiaojin/PubMedQA@9001f2853f` / `pqa_labeled/train-00000-of-00001.parquet` | 1000 | `3d56bd1abc118845…` |

- MedQA-USMLE：test split 全部 1,273 題（4 選項版本），依 `meta_info` 分 Step 1 與 Step 2&3。
- MedQA-TWMLE：test split 全部 1,413 題。題幹以「承上題」等開頭、需要前一題情境的題目另列子集（本評測一律單題作答，不提供前題）。
- PubMedQA-L：專家標註 1,000 題，提供題目＋摘要內文（不含 long answer 結論段），答 yes/no/maybe；另報告 macro-F1。

### 1.2 作答協定

- 單一 user 訊息、零樣本、不給範例；模型思考（reasoning）開啟，effort 一律 **medium**。
- 只評分最終回答（`content`），不看思考內容。解析規則：取最後一個 `Answer: X`（MCQ 為 A–D，PubMedQA 為 yes/no/maybe）；找不到視為答錯並計入「無答案」。
- 每題用由題目 ID 雜湊得到的固定 seed；各系統的取樣參數見 1.4。
- 被截斷（`finish_reason=length`）照樣解析，另外計數。HTTP/串流錯誤會重試 2 次，仍失敗記為錯誤且算答錯。

提示詞範本（`medbench/prompts.py`，版本 v1）：

<details><summary>MCQ_EN (MedQA-USMLE, MedMCQA)</summary>

```text
Answer the following multiple-choice question about medicine. Exactly one option is correct.

{question}

{options}

Think it through, then give your final answer on the last line in the form "Answer: X", where X is one of {letters}.
```

</details>

<details><summary>MCQ_ZH_TW (MedQA-TWMLE)</summary>

```text
請回答以下醫學單選題，只有一個正確選項。

{question}

{options}

請先思考，最後一行以「Answer: X」的格式作答，X 為 {letters} 其中之一。
```

</details>

<details><summary>YESNO_EN (PubMedQA)</summary>

```text
Read the abstract (without its conclusion) and answer the research question with yes, no or maybe.

Abstract:
{context}

Question: {question}

Think it through, then give your final answer on the last line in the form "Answer: yes", "Answer: no" or "Answer: maybe".
```

</details>

### 1.3 統計方法

- 正確率 95% 信賴區間：Wilson score interval。
- 兩系統配對比較：只用兩邊都有作答紀錄的同一批題目；McNemar 精確檢定（雙尾二項檢定，看不一致題）；正確率差的 95% CI 用配對 bootstrap（10,000 次，seed 0）。
- 速度：每題時間從送出請求到串流結束（含排隊與 prefill）；TTFT 為第一個（思考或回答）token 的時間；解碼速度 = (completion tokens − 1) / (最後 token 時間 − 第一 token 時間)。
- 資源：每秒取樣一次 `nvidia-smi`（VRAM、使用率、功耗、溫度）、`/proc/meminfo`、`/proc/stat` 與伺服器行程 RSS。每段執行前先記錄 15 秒閒置作為功耗基準；能耗以梯形積分計算。

### 1.4 受測系統與硬體

硬體：NVIDIA GeForce RTX 4090, 24564 MiB, 570.211.01；CPU 13th Gen Intel(R) Core(TM) i7-13700；RAM 54.9 GiB；Linux-5.15.0-190-generic-x86_64-with-glibc2.35。

| 項目 | Bonsai 27B PQ2_0 | Strata Qwen3.8-Flash-Next 125B IQ3_S |
|---|---|---|
| 模型 | Ternary-Bonsai-2-27B-Abliterated-v2（Qwen3.8-27B 架構，dense 27B） | Qwen3.8-Flash-Next-GSQ-RCO（MoE 約 125B 總參數） |
| 權重 / 量化 | PQ2_0 三元（ternary）權重 + MTP 頭，GGUF 7.1 GiB（7,303 MiB） | IQ3_S（GSQ-RCO 重量化），GGUF 兩片共約 83.6 GB |
| 推論引擎 | PrismML llama.cpp fork prism-b10787（llama-server）+ MTP 推測解碼 | Strata 0.1.45.3（engine 自行編譯 CUDA 12.9）+ MTP 推測解碼（--spec 4） |
| 放置方式 | 全部層在 GPU（-ngl 99），KV 131,072 tokens / 4 slots，flash-attn | 低 RAM 模式 resident 變體：注意力/共用權重與 GPU 專家快取（約 10.2 GiB）在 VRAM，其餘專家常駐 RAM；context 32,768、KV int8、一次一個請求（實測 parallel 3 總吞吐較低） |
| 取樣參數 | temperature 1.0、top_p 0.95、top_k 20、min_p 0.05（模型 GGUF 內建預設），逐題 seed | temperature 1.0、top_p 0.95、top_k 20、min_p 0.05（與 Bonsai 相同），逐題 seed；關閉 Strata 的 answer_room 思考上限 |
| 併發請求數 | 3 | 1 |
| 請求額外參數 | `{"max_tokens": 16384, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "chat_template_kwargs": {"reasoning_effort": "medium"}}` | `{"max_tokens": 16384, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "reasoning_effort": "medium"}` |
| 執行期間 | 10-20 16:51→10-20 17:59；10-21 06:14→10-21 11:11 | 10-20 18:16→10-21 06:14 |

## 2. 正確率

| 資料集 | n | 多數類別基準 | Bonsai 27B PQ2_0 | Strata Qwen3.8-Flash-Next 125B IQ3_S |
|---|---:|---:|---:|---:|
| medqa_usmle | 1273 | 27.7%（A） | 84.8% [82.7%–86.6%] | 94.3% [92.9%–95.4%] |
| medqa_twmle | 1413 | 26.4%（B） | 77.5% [75.2%–79.6%] | 89.8% [88.1%–91.3%] |
| pubmedqa | 1000 | 55.2%（yes） | 71.3% [68.4%–74.0%] | 77.5% [74.8%–80.0%] |
| **全部** | 3686 | — | **78.3% [77.0%–79.6%]** | **88.0% [86.9%–89.0%]** |

多數類別基準 = 永遠回答該資料集最常見的正確答案時的正確率（隨機猜測的參考線）。中括號為 95% Wilson 信賴區間。

### 2.1 子集

| 資料集 | 子集 | n | Bonsai 27B PQ2_0 | Strata Qwen3.8-Flash-Next 125B IQ3_S |
|---|---|---:|---:|---:|
| medqa_usmle | step1 | 679 | 85.7% [82.9%–88.1%] | 95.6% [93.8%–96.9%] |
| medqa_usmle | step2&3 | 594 | 83.7% [80.5%–86.4%] | 92.8% [90.4%–94.6%] |
| medqa_twmle | standalone | 1389 | 77.9% [75.6%–80.0%] | 90.2% [88.5%–91.7%] |
| medqa_twmle | depends_on_previous (承上題) | 24 | 54.2% [35.1%–72.1%] | 66.7% [46.7%–82.0%] |
| pubmedqa | gold=yes | 552 | 80.8% [77.3%–83.9%] | 82.1% [78.6%–85.0%] |
| pubmedqa | gold=no | 338 | 67.8% [62.6%–72.5%] | 84.6% [80.4%–88.1%] |
| pubmedqa | gold=maybe | 110 | 34.5% [26.3%–43.8%] | 32.7% [24.7%–41.9%] |
| pubmedqa | macro_f1 | — | 0.608 | 0.660 |

### 2.2 配對比較

**Bonsai 27B PQ2_0（A） vs Strata Qwen3.8-Flash-Next 125B IQ3_S（B）**

| 資料集 | n | A | B | A − B（95% CI） | 都對 | 只有 A 對 | 只有 B 對 | 都錯 | McNemar p |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| medqa_twmle | 1413 | 77.5% | 89.8% | -12.3 [-14.4, -10.3] | 1068 | 27 | 201 | 117 | <0.001 |
| medqa_usmle | 1273 | 84.8% | 94.3% | -9.5 [-11.5, -7.5] | 1046 | 33 | 154 | 40 | <0.001 |
| pubmedqa | 1000 | 71.3% | 77.5% | -6.2 [-8.6, -3.8] | 662 | 51 | 113 | 174 | <0.001 |
| **全部** | 3686 | 78.3% | 88.0% | -9.7 [-10.9, -8.5] | 2776 | 111 | 468 | 331 | <0.001 |

### 2.3 失敗模式

| 資料集 | 指標 | Bonsai 27B PQ2_0 | Strata Qwen3.8-Flash-Next 125B IQ3_S |
|---|---|---:|---:|
| medqa_usmle | 無可解析答案 | 2 | 0 |
| medqa_usmle | 達 max_tokens 截斷 | 1 | 0 |
| medqa_usmle | 請求錯誤 | 0 | 0 |
| medqa_usmle | 未作答 | 0 | 0 |
| medqa_twmle | 無可解析答案 | 3 | 0 |
| medqa_twmle | 達 max_tokens 截斷 | 3 | 0 |
| medqa_twmle | 請求錯誤 | 0 | 0 |
| medqa_twmle | 未作答 | 0 | 0 |
| pubmedqa | 無可解析答案 | 0 | 0 |
| pubmedqa | 達 max_tokens 截斷 | 0 | 0 |
| pubmedqa | 請求錯誤 | 0 | 0 |
| pubmedqa | 未作答 | 0 | 0 |

### 2.4 答案分佈與混淆

選擇題預測答案分佈（檢查位置偏誤；none = 無可解析答案）：

| 資料集 | 系統 | A | B | C | D | none |
|---|---|---:|---:|---:|---:|---:|
| medqa_usmle | 正確答案 | 353 | 309 | 346 | 265 | 0 |
| medqa_usmle | Bonsai 27B PQ2_0 | 340 | 317 | 364 | 250 | 2 |
| medqa_usmle | Strata Qwen3.8-Flash-Next 125B IQ3_S | 341 | 315 | 348 | 269 | 0 |
| medqa_twmle | 正確答案 | 335 | 373 | 365 | 340 | 0 |
| medqa_twmle | Bonsai 27B PQ2_0 | 339 | 367 | 390 | 314 | 3 |
| medqa_twmle | Strata Qwen3.8-Flash-Next 125B IQ3_S | 340 | 381 | 359 | 333 | 0 |

PubMedQA 混淆矩陣（列 = 正確答案，欄 = 模型回答）：

**Bonsai 27B PQ2_0**

| 正確 \ 回答 | yes | no | maybe | none |
|---|---:|---:|---:|---:|
| yes | 446 | 21 | 85 | 0 |
| no | 32 | 229 | 77 | 0 |
| maybe | 56 | 16 | 38 | 0 |

**Strata Qwen3.8-Flash-Next 125B IQ3_S**

| 正確 \ 回答 | yes | no | maybe | none |
|---|---:|---:|---:|---:|
| yes | 453 | 24 | 75 | 0 |
| no | 20 | 286 | 32 | 0 |
| maybe | 50 | 24 | 36 | 0 |

## 3. 速度

| 資料集 | 指標 | Bonsai 27B PQ2_0 | Strata Qwen3.8-Flash-Next 125B IQ3_S |
|---|---|---:|---:|
| medqa_usmle | 輸入 tokens（平均） | 275 | 275 |
| medqa_usmle | 輸出 tokens（中位/平均） | 1428 / 1800 | 1054 / 1166 |
| medqa_usmle | TTFT 中位（s） | 0.49 | 0.91 |
| medqa_usmle | 每題延遲 中位 / P90（s） | 18.1 / 40.1 | 13.4 / 20.1 |
| medqa_usmle | 單請求解碼中位（tok/s） | 81.7 | 85.5 |
| medqa_usmle | 伺服器端 prefill / 解碼中位（tok/s） | 1243 / 81.8 | 299 / 85.4 |
| medqa_usmle | 推測解碼草稿接受率 | 59.9% | 64.2% |
| medqa_twmle | 輸入 tokens（平均） | 158 | 158 |
| medqa_twmle | 輸出 tokens（中位/平均） | 1131 / 1525 | 765 / 917 |
| medqa_twmle | TTFT 中位（s） | 0.41 | 0.69 |
| medqa_twmle | 每題延遲 中位 / P90（s） | 14.5 / 34.8 | 9.2 / 16.7 |
| medqa_twmle | 單請求解碼中位（tok/s） | 81.5 | 89.6 |
| medqa_twmle | 伺服器端 prefill / 解碼中位（tok/s） | 833 / 81.6 | 225 / 89.5 |
| medqa_twmle | 推測解碼草稿接受率 | 58.8% | 66.7% |
| pubmedqa | 輸入 tokens（平均） | 416 | 416 |
| pubmedqa | 輸出 tokens（中位/平均） | 692 / 745 | 628 / 675 |
| pubmedqa | TTFT 中位（s） | 0.46 | 0.97 |
| pubmedqa | 每題延遲 中位 / P90（s） | 8.7 / 14.7 | 8.3 / 12.5 |
| pubmedqa | 單請求解碼中位（tok/s） | 82.7 | 88.7 |
| pubmedqa | 伺服器端 prefill / 解碼中位（tok/s） | 1583 / 82.8 | 430 / 88.6 |
| pubmedqa | 推測解碼草稿接受率 | 64.6% | 66.8% |

「單請求解碼」以用戶端收到的串流計時（含網路與排隊）；「伺服器端」為伺服器在串流最後回報的 llama.cpp 格式 timings（prompt_n/prompt_ms、predicted_n/predicted_ms、draft_n/draft_n_accepted；沒有回報時顯示 —）。

| 資料集 | 指標 | Bonsai 27B PQ2_0 | Strata Qwen3.8-Flash-Next 125B IQ3_S |
|---|---|---:|---:|
| 全部 | 實際執行時間 | 6.08 h | 11.96 h |
| 全部 | 總輸出 tokens | 5,190,929 | 3,455,056 |
| 全部 | 總吞吐（tok/s，含併發） | 237.2 | 80.2 |
| 全部 | 題/小時 | 606 | 308 |

## 4. 資源耗用

| 指標 | Bonsai 27B PQ2_0 | Strata Qwen3.8-Flash-Next 125B IQ3_S |
|---|---:|---:|
| GPU VRAM 平均 / 峰值（整張卡） | 20.7 GiB / 20.7 GiB | 23.4 GiB / 23.4 GiB |
| GPU 使用率平均 | 88% | 92% |
| GPU 功耗 閒置 / 平均 / 峰值（W） | 48 / 341 / 366 | 72 / 206 / 267 |
| GPU 能耗總計（Wh） | 2075 | 2467 |
| GPU 能耗（扣閒置，Wh） | 1782 | 1605 |
| GPU 能耗 / 題（Wh） | 0.56 | 0.67 |
| GPU 能耗 / 答對一題（Wh） | 0.72 | 0.76 |
| 伺服器行程 RSS 平均 / 峰值 | 25.6 GiB / 26.5 GiB | 43.0 GiB / 43.0 GiB |
| 系統 RAM 使用 平均 / 峰值 | 30.6 GiB / 32.1 GiB | 48.6 GiB / 49.0 GiB |
| CPU 使用率平均（全機） | 12% | 96% |
| GPU 溫度峰值（°C） | 71 | 59 |
| 同時有其他使用者請求的時間比例 | 0.0% | 0.0% |

註：VRAM、功耗與系統 RAM 為整機量測，包含同時常駐的其他服務（見第 5 節）；能耗只計 GPU，不含 CPU 與其他元件。

## 5. 測試環境與設定細節

- **主機**：KVM 虛擬機，12 個 vCPU（宿主 CPU 為 Intel i7-13700）、RAM 55 GiB（另有 16 GiB swap）、單張 RTX 4090 24 GB 以 PCIe 直通（Gen4 x16，driver 570.211.01）、QEMU 虛擬磁碟、Ubuntu 22.04。CPU 使用率以這 12 個 vCPU 計算。
- **共用 GPU**：兩次執行期間，路由用的決策模型 Clef-Flash（llama.cpp，約 6.5 GB VRAM）都常駐在同一張卡上。評測期間 Bonsai 沒有其他使用者請求（第 4 節最後一列為 0%），路由器因此也沒有送請求給 Clef，它只占 VRAM、不占運算。第 4 節的 VRAM 與功耗為整張卡的量測值。
- **Bonsai 27B**：在正式服務上執行（llama-server :8081，4 個 slot），評測佔用 3 個併發、保留 1 個給使用者；`resources.csv` 的 `server_busy` 欄位記錄伺服器忙碌槽位，與我方 `inflight` 比較即可看出是否有其他使用者同時請求（第 4 節最後一列）。伺服器設定 `--cache-ram 24576`（最多 24 GiB 的提示快取放在 RAM），所以 llama-server 行程的 RSS 隨評測逐步長到約 20–26 GiB；這是伺服器快取設定，不是模型本身的需求（模型權重 7.1 GiB 全在 VRAM）。
- **Strata 125B**：需要整張 GPU 與大部分 RAM，因此在維護時段執行（暫停 Bonsai，由 `scripts/strata_window.sh` 啟動 Strata，跑完自動恢復正式服務）。採 Strata setup 為這台機器建議的 IQ3_S（低 RAM 模式的 resident 變體：GPU 放約 10.2 GiB 的熱門專家，其餘約 37 GiB 專家常駐 RAM），只改了三處（見 `configs/strata-flashnext-iq3s-medium.json` 的 `launch`）：
  1. engine 改用本機自行編譯的 CUDA 12.9 版（官方現成版需要 driver ≥ 575.51，本機為 570）。
  2. `--max-context` 131,072 → 32,768：本評測最長的提示加上 16,384 個輸出 token 不超過 2 萬，縮小 KV 快取可把 VRAM 留給專家快取。
  3. `answer_room: false`：Strata 預設在 `max_tokens` ≥ 1,024 時把思考上限設為 `max_tokens` 的 3/4（此處 12,288），以保證有空間作答；llama.cpp 沒有這個機制，為了讓兩邊只受同一個 `max_tokens = 16,384` 限制而關閉。
- **併發設定的選擇（Strata）**：試跑比較了三種設定：parallel 3 + 131k context（GPU 專家快取只剩 3.1 GiB，每請求約 9 tok/s）、parallel 3 + 32k（專家快取 8.6 GiB，每請求約 17 tok/s，總計約 51 tok/s）、parallel 1 + 32k（專家快取 10.2 GiB，約 70 tok/s）。專家放在 RAM 的 MoE 模型同時解多個請求時，每一步要搬的專家是各請求的聯集，PCIe 頻寬成為瓶頸，所以一次一個請求的總吞吐反而最高，正式評測採 parallel 1。Bonsai 是全部在 GPU 的密集模型，3 個併發可以提高總吞吐，因此兩邊各用自己最有效率的設定；比較速度時請看「單請求解碼速度」與「題/小時」兩種指標。
- **思考強度**：兩邊都是 `reasoning_effort = medium`。Bonsai 透過 chat template（`chat_template_kwargs`）；Strata 用自己的 `effort_end` 方式（同樣的 medium 指令放在回答前的短 system turn）。兩者都是各自系統官方的 medium 實作。
- **執行時段（伺服器時間）**：Bonsai 分兩段，10-20 16:51 起跑了 509 題（為了做 Strata 試跑而中斷，該段結束時間由最後完成的題目推得），10-21 06:14–11:11 跑完其餘 3,177 題；Strata 10-20 18:16 – 10-21 06:14 一次跑完（正式服務停機約 12 小時，期間 Bonsai 暫停）。中斷時進行中的題目不記錄，續跑時重做，所以每題只有一筆結果。兩段程式的提示詞與解析器相同（`medbench/prompts.py` 自第一個 commit 起未修改）；續跑後全部輸出重新解析，結果與記錄一致（0 筆不符）。
- **模型版本**：Bonsai 為 `Ternary-Bonsai-2-27B-Abliterated-v2`（PQ2_0 + MTP）。「Abliterated」表示移除了拒答方向的權重修改版，這是目前正式服務部署的版本，不是原始 Qwen3.8-27B。

## 6. 討論

### 6.1 正確率：Strata 125B 在三個題庫都顯著較高

- 整體 **Strata 88.0% vs Bonsai 78.3%**，配對差 9.7 個百分點（95% CI 8.5–10.9），McNemar p < 0.001。兩者答案不一致的 579 題中，只有 Strata 答對的有 468 題，只有 Bonsai 答對的有 111 題（約 4.2 : 1）。
- 三個題庫方向一致：MedQA-USMLE 94.3% vs 84.8%（差 9.5）、MedQA-TWMLE 89.8% vs 77.5%（差 12.3）、PubMedQA 77.5% vs 71.3%（差 6.2），各自 p < 0.001。兩者在 USMLE 都遠高於 MedQA 文獻常用的約 60% 及格參考線。
- **繁體中文差距最大**：從 USMLE 換到台灣國考，Bonsai 下降 7.3 個百分點，Strata 只下降 4.5。台灣題目除了語言，也包含本地法規、公衛與用藥習慣，這部分 27B 的三元量化模型明顯吃虧。「承上題」24 題兩者都偏低（54% vs 67%），因為本評測不提供前題情境。
- **USMLE Step 1 與 Step 2&3**：兩者在臨床情境較多的 Step 2&3 都比 Step 1 低 2–3 個百分點（Bonsai 85.7→83.7，Strata 95.6→92.8）。
- **PubMedQA**：兩者都很少答對「maybe」（34.5% vs 32.7%），這是此資料集已知的難點（原論文報告的單一人類標註者正確率約 78%）。主要差別在「no」：Bonsai 只答對 67.8%，有 77 題把「no」答成「maybe」；Strata 答對 84.6%（答成 maybe 的只有 32 題）。Bonsai 整體答「maybe」200 次（正確答案只有 110 題是 maybe），傾向保守模糊；macro-F1 0.608 vs 0.660。
- **兩者都答錯** 331 題（9.0%），可能包含真正困難的題目與題庫本身的標註問題，本報告未逐題審查。
- **失敗模式**：Strata 3,686 題全部有可解析答案、沒有截斷。Bonsai 有 5 題無答案（0.14%）：其中 4 題是退化輸出，在思考中無限重複同一個字或詞（例如「perivascular」、「患」、「n」）直到 16,384 tokens 上限，這是極低位元量化模型的典型現象；另 1 題思考結束後沒有輸出回答。即使把這 5 題算成答對，結論也不變。
- **位置偏誤**：兩者的選項分佈都接近正確答案分佈，沒有明顯偏向特定字母（第 2.4 節）。

### 6.2 速度：Strata 單請求較快、想得較短；Bonsai 總吞吐是 2 倍

- **單請求解碼**：Strata 一次只服務一個請求，中位 87.6 tok/s；Bonsai 同時 3 個請求、每個中位 81.8 tok/s（Bonsai 單獨一個請求時，前一日同機測得約 128 tok/s）。兩者都用推測解碼（MTP），草稿接受率 Bonsai 59–65%（每次草擬 2 個）、Strata 64–67%（每次 4 個）。
- **輸出長度**：在相同的 medium 思考強度下，Strata 平均每題輸出 937 tokens，Bonsai 1,408 tokens（多 50%）。因此即使 Bonsai 同時處理 3 題，Strata 每題的中位延遲仍較短（10.6 s vs 13.7 s；USMLE P90 20 s vs 40 s）。
- **總吞吐**：Bonsai 3 併發合計 237 tok/s、每小時 606 題，全部 3,686 題 6.1 小時；Strata 80 tok/s、每小時 308 題，11.96 小時。Strata 增加併發反而更慢（第 5 節試跑），這是專家放在 RAM 的 MoE 模型在單張 24 GB 卡上的結構性限制。
- **首 token 時間**：題目都很短（平均 160–420 tokens），Bonsai 伺服器端 prefill 833–1,583 tok/s、TTFT 中位 0.4–0.5 s；Strata 225–430 tok/s、TTFT 0.7–1.0 s。短提示下 Strata 每個請求有固定的專家搬運開銷；長文件的 prefill 不在本評測範圍。

### 6.3 資源與能耗：Strata 需要整台機器

- **Bonsai**：整張卡 VRAM 20.7 GiB（Bonsai 約 14 GiB + Clef 6.5 GiB），仍可同時服務其他使用者；llama-server RSS 25.6 GiB 幾乎都是 `--cache-ram 24576` 的提示快取，可依需求調小；CPU 平均 12%；GPU 平均 341 W（3 個併發幾乎吃滿 GPU 運算）。
- **Strata**：VRAM 23.4 GiB（整張卡）、RSS 43 GiB、系統 RAM 48.6/54.9 GiB、12 個 vCPU 平均 96%；GPU 使用率 92% 但平均功耗只有 206 W（Bonsai 341 W），GPU 多在等待專家資料，瓶頸在 PCIe 搬運與 CPU，而不是 GPU 運算。執行時必須停掉 Bonsai。
- **能耗（只計 GPU）**：每題 0.56 Wh vs 0.67 Wh；以「每答對一題」計 0.72 vs 0.76 Wh，幾乎相同。Strata 的 CPU 負載很高，計入 CPU 後差距會拉大（未量測）。

### 6.4 結論

1. **品質**：在這台 RTX 4090 + 55 GiB RAM 的機器上，Strata 跑的 Qwen3.8-Flash-Next 125B（IQ3_S）醫學問答正確率明顯高於目前正式服務的 Bonsai 27B（PQ2_0），整體多 9.7 個百分點，繁體中文多 12.3 個百分點，而且沒有退化輸出。
2. **代價**：Strata 需要整張 GPU、約 43 GiB RAM 和幾乎全部 CPU，不能與 Bonsai 同時運作；總吞吐只有 Bonsai 的一半，增加併發也無法改善。Bonsai 留有 VRAM 與 CPU 餘裕，可同時服務多人並搭配路由與工具。
3. **建議**：正式多人服務維持 Bonsai；需要較高醫學正確性、可以接受一次一人的場景（例如批次審閱、離峰時段、單人深度問答），可用 `scripts/strata_window.sh` 的方式切換到 Strata。若要長期同時提供兩者，需要更多 RAM（讓 Strata 不必用低 RAM 模式）或第二張 GPU。

## 7. 限制

- **資料汙染**：MedQA（2020）與 PubMedQA（2019）公開多年，題目可能出現在兩個模型的訓練資料中；分數宜作為兩系統的相對比較，不代表臨床能力。
- **單次取樣**：溫度 1.0、每題一次。逐題固定 seed 讓同一系統可重跑，但兩個引擎的隨機數實作不同，同一 seed 不會產生相同取樣路徑。配對檢定與 bootstrap 已把單題隨機性納入不確定性。
- **TWMLE 承上題**：24 題需要前一題的情境，本評測單題作答，這些題目對兩個系統都較不利（第 2.1 節另列）。
- **零樣本、無工具**：不使用檢索、MCP 工具或 few-shot 範例，測的是模型本身的醫學知識與推理；實際服務中 router 可接 PubMed 等工具，表現可能不同。
- **量化**：兩個模型都是極低位元量化（Bonsai 三元 PQ2_0；Strata IQ3_S 等級的混合量化，各層專家為 IQ2_S～IQ3_S、部分 IQ4_NL），結果代表「這台機器上可跑的版本」，不代表原始 BF16 模型。
- **速度條件不同**：Bonsai 3 併發、Strata 1 併發（見第 5 節），且 Strata 的速度取決於專家快取命中率與 PCIe，換到 VRAM/RAM 較多的機器會不同。
- **RAM 餘裕**：Strata IQ3_S 執行時系統 RAM 使用平均 48.6 GiB（總量 54.9 GiB），伺服器行程 RSS 43 GiB，另有約 4 GiB 的其他行程被換出到 swap；12 小時內沒有發生 OOM，但餘裕很小。
- **CPU 能耗未計**：第 4 節只量 GPU 功耗。Strata 在整段評測中讓全機 CPU 平均使用率達 96%（Bonsai 12%），若計入 CPU，Strata 每題的實際能耗會比表中高。

## 重現步驟

```bash
git clone https://github.com/u9401066/medqa-bench && cd medqa-bench
uv venv && uv pip install -e '.[test]'
medbench prepare                       # 下載、驗證 SHA-256、正規化成 data/*.jsonl
medbench run --config configs/bonsai27b-pq2-medium.json   # 可中斷續跑
# Strata Qwen3.8-Flash-Next 125B IQ3_S 需要整張 GPU：暫停正式服務、啟動受測伺服器、跑完自動恢復
PARALLEL=1 STRATA_CONFIG=configs/engine/strata-iq3_s-32k.json scripts/strata_window.sh configs/strata-flashnext-iq3s-medium.json
medbench score results/bonsai27b-pq2-medium results/strata-flashnext-iq3s-medium
medbench report results/bonsai27b-pq2-medium results/strata-flashnext-iq3s-medium --notes reports/notes.md
```

每個 `results/<run>/` 內含 `manifest.json`（設定、git 版本、伺服器 /props、資料雜湊、主機資訊、執行時段）、`predictions.jsonl`（逐題答案與時間）、`raw.jsonl.gz`（完整思考與回答）、`resources.csv`（每秒資源）、`summary.json`。

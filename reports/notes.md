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
- 三個題庫方向一致：MedQA-USMLE 94.3% vs 84.8%（差 9.5）、MedQA-TWMLE 89.8% vs 77.5%（差 12.3）、PubMedQA 77.5% vs 71.3%（差 6.2），各自 p < 0.001。作為量級參考：Med-PaLM 2 論文指出 Med-PaLM 在 MedQA 的 67.2% 是第一個超過 USMLE 式題目「及格」分數的模型，Med-PaLM 2 最高 86.5%（Singhal et al., 2023；作答條件與本評測不同，不宜直接比較）。
- **繁體中文差距最大**：從 USMLE 換到台灣國考，Bonsai 下降 7.3 個百分點，Strata 只下降 4.5。可能的原因是台灣題目除了語言不同，也包含本地法規、公衛與用藥習慣，較依賴模型的知識量；本評測沒有依題目類型細分，這點未經驗證。「承上題」24 題兩者都偏低（54% vs 67%），因為本評測不提供前題情境。
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

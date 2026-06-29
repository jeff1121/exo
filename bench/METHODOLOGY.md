# Exo-Bench — 方法論

exo bench 在受控條件下測量 exo 叢集的推論吞吐量與資源消耗。它會向 `/bench/chat/completions` 端點送出提示，收集伺服器回報的時間統計，並在每次執行期間記錄系統層級指標（功耗、GPU 使用率、溫度）。

目標是取得準確、透明且可重現的數據，以比較不同模型與不同配置下的速度與擴展性，並能隨著 EXO 新增最佳化與功能時追蹤結果變化。

以下是 Exo-Bench 的技術摘要。雖然方法論與基準測試可能會隨時間變更，但本文件會在變更時保持更新。如果你發現方法論問題，或希望新增功能，請開 GitHub issue！

---

## 提示建構

基準測試需要精確 token 長度的提示。不幸的是，我們無法直接存取模型，只能透過 chat completion 端點。為了解決這點，我們會建立一個在 token 化後可達到特定提示長度的請求。

做法如下：

1. 透過模型的 `apply_chat_template()` 對範例訊息進行 token 化，以量測額外開銷（system tokens、special tokens、chat formatting）。
2. 對重複的原子字串（預設為 `"a "`）進行二分搜尋，找出在樣板展開後可精確產生目標 token 數的內容長度。
3. 回傳內容字串與驗證過的 token 計數。

實際 token 數會在每一列結果中記錄為 `pp_tokens`，讓後續分析可確認提示是否命中目標。

由於 chat template 格式化，某些非常小的 pp 基準可能無法達成，例如 pp=32 可能不可行。這個取捨是因為如此小的提示在真實場景中通常不太有趣或實用。

---

## Bench 端點

當請求透過 `/bench/chat/completions` 端點到達伺服器時，與一般 chat completion 相比有三點不同：

- **KV prefix cache 預設停用**。每個請求都從冷快取開始，確保 prefill 計時不受先前請求影響。`--use-prefix-cache` 選項請見 [Prefix Cache Mode](#prefix-cache-mode)。
- **禁止 EOS token**。logits processor 會抑制所有 end-of-sequence token，強制模型精確生成 `max_tokens` 個 token。這可保證生成長度一致，公平比較 TPS——模型無法透過提早停止縮短執行時間。
- **不解析模型輸出**。bench 收集路徑會串接原始 token 文字，不做任何模型特定後處理（thinking 標籤擷取、結構化輸出處理等）。這是為了避免工具解析或結構錯誤破壞基準測試——我們測的是速度；效能品質指標請見 Exo-Eval。

---

## 計時

### Prefill TPS

在伺服器端以每個 task 測量。

```
prefill_tps = num_prompt_tokens / prefill_wall_seconds
```

### Generation TPS

在伺服器端以每個 task 測量。每個 task 會在 token 到達時記錄 wall-clock 時間戳：

- 第一個生成 token：記錄時間戳
- 後續每個 token：更新時間戳

生成完成時：

```
gen_span = last_token_time - first_token_time
generation_tps = (completion_tokens - 1) / gen_span
```

分子會排除第一個 token，因為速率衡量的是 token 間吞吐量——以第一到最後一個 token 的時間差，除以區間數。

這也代表 tg=1 不可用。

---

## 並行度

### 單一請求

客戶端會在 HTTP 往返外圍記錄 wall-clock `elapsed_s`（網路延遲 + 伺服器 prefill + generation + 回應序列化）。這是端到端延遲的便利指標。權威 TPS 數值來自 `generation_stats` 回應中的伺服器端每 task 計時。

### 並行請求

當設定 `--concurrency N` 且 N > 1 時，所有 N 個請求必須同時打到伺服器。機制如下：

1. 提示只建構一次，並由所有執行緒共用。
2. 每個執行緒各自使用獨立 HTTP 連線。
3. 執行緒 barrier 會阻擋所有執行緒，直到全部準備完成。
4. 第一個通過 barrier 的執行緒記錄批次開始時間並通知其他執行緒。
5. 所有執行緒以相同開始時間為基準，然後送出 HTTP 請求。
6. 每個執行緒的 `elapsed_s` 皆由共用開始時間量測到自身回應完成。

**批次牆鐘時間（Batch wall time）** 是 N 個請求中最大的 `elapsed_s`——也就是最後一個請求完成所需時間。

### 聚合 TPS

```
per_req_tps = max(generation_tps across N concurrent requests)
agg_gen_tps = per_req_tps * concurrency
```

使用 `max` 而非 `mean`，因為所有請求都並行對同一模型執行。最快請求的生成速率代表系統每串流吞吐能力；再乘上並行度即可得到聚合吞吐量。

---

## Prefix Cache 模式

當傳入 `--use-prefix-cache` 時，KV prefix cache 在基準測試期間保持啟用。這能透過跳過重複 prefill 工作加速重複執行；當提示處理不是測試重點時（例如量測生成吞吐量或多組配置下的功耗）特別有用。

每個回應都包含 `prefix_cache_hit` 欄位（`"none"`、`"partial"` 或 `"exact"`）：

- **none**：冷 prefill——沒有可用的快取 KV 狀態。回報的 `prompt_tps` 是真實 prefill 吞吐量。
- **partial**：提示前綴命中快取。只會 prefill 剩餘 token。回報的 `prompt_tps` 反映未快取部分的真實吞吐量。這常見於遞增 `--pp` 共用前綴時（例如 `--pp 1000,5000`——5000 token 提示會重用 1000 token 的快取項，並 prefill 剩餘 4000 token）。
- **exact**：整個提示命中快取（例如在 `--repeat` 下重複相同 `--pp`）。不會執行 prefill。回報的 `prompt_tps` 來自該快取項最初建立時的 TPS，而非新的量測。

**此模式下 Prompt TPS 為近似值。** exact-hit 執行會回報原始冷/部分 prefill 儲存的 TPS，而不是新量測值。若要取得準確冷 prefill 數據，請不要使用 `--use-prefix-cache`。

遞增 `--pp` 順序（例如 `--pp 1000,5000,10000`）最有參考價值：第一個為冷啟動，其餘大小可得到有意義的部分命中。遞減順序則會產生 exact hit，並回報來自較長提示原始執行的近似 TPS。

---

## 預熱

在正式量測前，會先用第一組 pp/tg 組合送出 `--warmup N`（預設：0）個捨棄請求。預熱結果不會納入輸出。

---

## 系統指標

背景執行緒會以 1 Hz 輪詢每個節點，收集：

- GPU 使用率 (%)
- 溫度 (C)
- 系統功耗 (W)
- CPU 叢集使用率（效能核心與效率核心）

**能量** 透過對每個推論視窗（每筆基準請求或並行批次的 wall-clock 區間）內的功率樣本做梯形積分計算。平均功率為 `total_joules / total_inference_seconds`。此外，伺服器會在每個非串流 `/bench/chat/completions` 回應中回傳 `power_usage` 區塊，將能量拆分為 prefill 與 generation 階段，分界點錨定在 runner 的第一個非 `PrefillProgressChunk`。

---

## 輸出格式

結果會以 JSON 寫出，含三個頂層鍵：

- **`runs`**：每請求結果物件陣列，每筆包含：
  - `elapsed_s`、`output_text_preview`（前 200 字元）
  - `stats`：`{ prompt_tps, generation_tps, prompt_tokens, generation_tokens, peak_memory_usage }`
  - `power_usage`：伺服器端總量 + prefill/generation 拆分、各節點拆分（僅非串流請求）
  - 佈署中繼資料：`model_id`、`placement_sharding`、`placement_instance_meta`、`placement_nodes`
  - 執行中繼資料：`pp_tokens`、`tg`、`repeat_index`、`concurrency`、`concurrent_index`
  - `download_duration_s`（若模型為首次下載）
- **`cluster`**：基準測試當下的叢集狀態快照。
- **`system_metrics`**：各節點時間序列樣本（GPU、功耗、溫度）。

---

## 重現結果

```bash
cd bench && uv run python exo_bench.py \
  --model "mlx-community/Qwen3.5-27B-4bit" \
  --instance-meta jaccl \
  --sharding tensor \
  --min-nodes 2 --max-nodes 2 \
  --pp 512 4096 --tg 128 \
  --repeat 3 \
  --warmup 1
```

執行 --help 以查看所有可用旗標。

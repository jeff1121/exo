# EXO API – 技術參考

本文件說明 **EXO** 服務所提供的 REST API，實作位於：

`src/exo/master/api.py`

此 API 用於管理叢集中的模型實例、檢視叢集狀態，並透過多種 API 相容介面執行推論。

Base URL 範例：

```
http://localhost:52415
```

## 1. 一般 / 中繼端點

### 取得主節點 ID

**GET** `/node_id`

回傳目前主節點的識別碼。

**回應（範例）：**

```json
{
  "node_id": "node-1234"
}
```

### 取得叢集狀態

**GET** `/state`

回傳目前叢集狀態，包含節點與啟用中的實例。

**回應：**
描述拓撲、節點與實例的 JSON 物件。

### 取得事件

**GET** `/events`

回傳主節點記錄的內部事件列表（主要用於除錯與可觀測性）。

**回應：**
事件物件陣列。

## 2. 模型實例管理

### 建立實例

**POST** `/instance`

在叢集中建立新的模型實例。

**請求主體（範例）：**

```json
{
  "instance": {
    "model_id": "llama-3.2-1b",
    "placement": { }
  }
}
```

**回應：**
指令確認。實例建立為非同步；在對該模型送出推論請求前，客戶端應先透過 `/instance/await` 等待模型出現。

### 刪除實例

**DELETE** `/instance/{instance_id}`

依 ID 刪除既有實例。

**路徑參數：**

* `instance_id`：string，要刪除的實例 ID

**回應：**
狀態 / 確認 JSON。

### 取得實例

**GET** `/instance/{instance_id}`

回傳特定實例的詳細資訊。

**路徑參數：**

* `instance_id`：string

**回應：**
實例描述 JSON。

### 等待實例

**GET** `/instance/await?model_id=...&timeout_seconds=0`

等待 API 狀態中出現請求模型的實例。回應為 SSE 串流，因此客戶端在等待期間會收到 keep-alive 註解。

**查詢參數：**

* `model_id`：string，必填
* `timeout_seconds`：float，選填，預設 `0`。`0` 表示無限等待；正值表示超過該秒數即逾時。正值最大為 `300`。

**串流訊息：**

```text
data: {"type": "ready", "instance": {...}}

data: {"type": "timeout", "message": "No instance found for model ..."}
```

兩種訊息的 HTTP 狀態皆為 `200`，因為串流在最終結果確定前就已開始。終端訊息可由 `type` 欄位區分。

### 預覽放置結果

**GET** `/instance/previews?model_id=...`

回傳指定模型的可能放置預覽。

**查詢參數：**

* `model_id`：string，必填

**回應：**
放置預覽物件陣列。

### 計算放置方案

**GET** `/instance/placement`

針對潛在實例計算放置方案，但不建立實例。

**查詢參數（常見）：**

* `model_id`：string
* `sharding`：string 或設定
* `instance_meta`：JSON 編碼中繼資料
* `min_nodes`：integer

**回應：**
描述建議放置 / 實例設定的 JSON 物件。

### 放置實例

**POST** `/place_instance`

使用伺服器放置邏輯為模型放置實例。

**請求主體：**
描述要放置實例的 JSON。

**回應：**
指令確認。實例可能不會立即可用；在送出推論請求前，請先透過 `/instance/await` 等待實例出現。

## 3. 模型

### 列出模型

**GET** `/models`
**GET** `/v1/models`（別名）

回傳可用模型及其中繼資料列表。

**查詢參數：**

* `status`：string（選填）- 可用 `downloaded` 篩選僅顯示已下載模型

**回應：**
模型描述子陣列，包含自訂 HuggingFace 模型使用的 `is_custom` 欄位。

### 新增自訂模型

**POST** `/models/add`

從 HuggingFace hub 新增自訂模型。

**請求主體（範例）：**

```json
{
  "model_id": "mlx-community/my-custom-model"
}
```

**回應：**
已新增模型的模型描述子。

**安全性說明：**
若模型設定啟用 `trust_remote_code`，基於安全性必須明確 opt-in（預設為 false）。

### 刪除自訂模型

**DELETE** `/models/custom/{model_id}`

刪除使用者新增的自訂模型卡片。

**路徑參數：**

* `model_id`：string，要刪除的自訂模型 ID

**回應：**
包含已刪除模型 ID 的確認 JSON。

### 搜尋模型

**GET** `/models/search`

在 HuggingFace Hub 搜尋 mlx-community 模型。

**查詢參數：**

* `query`：string（選填）- 搜尋關鍵字
* `limit`：integer（預設：20）- 回傳結果上限

**回應：**
HuggingFace 模型搜尋結果陣列。

## 4. 推論 / Chat Completions

### OpenAI 相容 Chat Completions

**POST** `/v1/chat/completions`

使用 OpenAI 相容 schema 執行 chat completion 請求。支援串流與非串流模式。

**請求主體（範例）：**

```json
{
  "model": "llama-3.2-1b",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello" }
  ],
  "stream": false
}
```

**請求參數：**

* `model`：string，必填 - 要使用的模型 ID
* `messages`：array，必填 - 對話訊息
* `stream`：boolean（預設：false）- 啟用串流回應
* `max_tokens`：integer（選填）- 最大生成 token 數
* `temperature`：float（選填）- 取樣溫度
* `top_p`：float（選填）- nucleus sampling 參數
* `top_k`：integer（選填）- top-k sampling 參數
* `stop`：string 或 array（選填）- 停止序列
* `seed`：integer（選填）- 供可重現性的隨機種子
* `enable_thinking`：boolean（選填）- 為支援模型啟用 thinking mode（DeepSeek V3.1、Qwen3、GLM-4.7）
* `tools`：array（選填）- function calling 的工具定義
* `logprobs`：boolean（選填）- 回傳對數機率
* `top_logprobs`：integer（選填）- 要回傳的最高對數機率數量

**回應：**
OpenAI 相容的 chat completion 回應。

**串流回應格式：**
當 `stream=true` 時，回傳格式如下的 Server-Sent Events (SSE)：

```
data: {"id":"...","object":"chat.completion","created":...,"model":"...","choices":[...]}

data: [DONE]
```

**非串流回應包含使用量統計：**

```json
{
  "id": "...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama-3.2-1b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 8,
    "total_tokens": 23
  }
}
```

**取消：**
你可以透過關閉 HTTP 連線取消進行中的生成。伺服器會偵測中斷並停止處理。

### Claude Messages API

**POST** `/v1/messages`

使用 Claude Messages API 格式執行 chat completion 請求。支援串流與非串流模式。

**請求主體（範例）：**

```json
{
  "model": "llama-3.2-1b",
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "max_tokens": 1024,
  "stream": false
}
```

**串流回應格式：**
當 `stream=true` 時，回傳包含 Claude 專用事件類型的 Server-Sent Events：

* `message_start` - 訊息生成開始
* `content_block_start` - 內容區塊開始
* `content_block_delta` - 增量內容片段
* `content_block_stop` - 內容區塊完成
* `message_delta` - 訊息中繼資料更新
* `message_stop` - 訊息生成完成

**回應：**
Claude 相容的 messages 回應。

### OpenAI Responses API

**POST** `/v1/responses`

使用 OpenAI Responses API 格式執行 chat completion 請求。支援串流與非串流模式。

**請求主體（範例）：**

```json
{
  "model": "llama-3.2-1b",
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "stream": false
}
```

**串流回應格式：**
當 `stream=true` 時，回傳包含 response 專用事件類型的 Server-Sent Events：

* `response.created` - 回應生成開始
* `response.in_progress` - 回應生成中
* `response.output_item.added` - 新增輸出項目
* `response.output_item.done` - 輸出項目完成
* `response.done` - 回應生成完成

**回應：**
OpenAI Responses API 相容回應。

### 基準化 Chat Completions

**POST** `/bench/chat/completions`

與 `/v1/chat/completions` 相同，但會額外回傳效能與生成統計。

**請求主體：**
與 `/v1/chat/completions` 相同 schema。

**回應：**
chat completion 加上基準指標，包含：

* `prompt_tps` - 提示處理期間的每秒 token 數
* `generation_tps` - 生成期間的每秒 token 數
* `prompt_tokens` - 提示 token 數
* `generation_tokens` - 已生成 token 數
* `peak_memory_usage` - 生成期間峰值記憶體使用量

### 取消命令

**POST** `/v1/cancel/{command_id}`

取消進行中的生成命令（文字或圖片）。會通知 workers 並關閉串流。

**路徑參數：**

* `command_id`：string，要取消的命令 ID

**回應（範例）：**

```json
{
  "message": "Command cancelled.",
  "command_id": "cmd-abc-123"
}
```

若命令不存在或已完成，回傳 404。

## 5. Ollama API 相容性

EXO 提供 Ollama API 相容性，可搭配 OpenWebUI 等工具。

### Ollama Chat

**POST** `/ollama/api/chat`
**POST** `/ollama/api/api/chat`（別名）
**POST** `/ollama/api/v1/chat`（別名）

使用 Ollama API 格式執行聊天請求。

**請求主體（範例）：**

```json
{
  "model": "llama-3.2-1b",
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "stream": false
}
```

**回應：**
Ollama 相容的聊天回應。

### Ollama Generate

**POST** `/ollama/api/generate`

使用 Ollama API 格式執行文字生成請求。

**請求主體（範例）：**

```json
{
  "model": "llama-3.2-1b",
  "prompt": "Hello",
  "stream": false
}
```

**回應：**
Ollama 相容的生成回應。

### Ollama Tags

**GET** `/ollama/api/tags`
**GET** `/ollama/api/api/tags`（別名）
**GET** `/ollama/api/v1/tags`（別名）

以 Ollama tags 格式回傳已下載模型列表。

**回應：**
含中繼資料的模型 tags 陣列。

### Ollama Show

**POST** `/ollama/api/show`

以 Ollama show 格式回傳模型資訊。

**請求主體：**

```json
{
  "name": "llama-3.2-1b"
}
```

**回應：**
包含 modelfile 與 family 的模型詳細資訊。

### Ollama PS

**GET** `/ollama/api/ps`

回傳執行中模型（啟用實例）列表。

**回應：**
啟用中的模型實例陣列。

### Ollama Version

**GET** `/ollama/api/version`
**HEAD** `/ollama/`（別名）
**HEAD** `/ollama/api/version`（別名）

回傳 Ollama API 相容性的版本資訊。

**回應：**

```json
{
  "version": "exo v1.0"
}
```

## 6. 圖片生成與編輯

### 圖片生成

**POST** `/v1/images/generations`

使用 OpenAI 相容 schema（含額外 advanced_params）執行圖片生成請求。支援串流與非串流模式。

**請求主體（範例）：**

```json
{
  "prompt": "a robot playing chess",
  "model": "exolabs/FLUX.1-dev",
  "n": 1,
  "size": "1024x1024",
  "stream": false,
  "response_format": "b64_json"
}
```

**請求參數：**

* `prompt`：string，必填 - 圖片文字描述
* `model`：string，必填 - 圖片模型 ID
* `n`：integer（預設：1）- 生成圖片數量
* `size`：string（預設："auto"）- 圖片尺寸。支援尺寸：
  - `512x512`
  - `768x768`
  - `1024x768`
  - `768x1024`
  - `1024x1024`
  - `1024x1536`
  - `1536x1024`
  - `1024x1365`
  - `1365x1024`
* `stream`：boolean（預設：false）- 啟用部分圖片串流
* `partial_images`：integer（預設：0）- 生成期間要串流的部分圖片數
* `response_format`：string（預設："b64_json"）- `url` 或 `b64_json`
* `quality`：string（預設："medium"）- `high`、`medium` 或 `low`
* `output_format`：string（預設："png"）- `png`、`jpeg` 或 `webp`
* `advanced_params`：object（選填）- 進階生成參數

**進階參數（`advanced_params`）：**

| Parameter | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `seed` | int | >= 0 | 用於可重現生成的隨機種子 |
| `num_inference_steps` | int | 1-100 | 去噪步數 |
| `guidance` | float | 1.0-20.0 | classifier-free guidance scale |
| `negative_prompt` | string | - | 描述圖片中要避免出現內容的文字 |

**非串流回應：**

```json
{
  "created": 1234567890,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAA...",
      "url": null
    }
  ]
}
```

**串流回應格式：**
當 `stream=true` 且 `partial_images > 0` 時，回傳 Server-Sent Events：

```
data: {"type":"partial","image_index":0,"partial_index":1,"total_partials":5,"format":"png","data":{"b64_json":"..."}}

data: {"type":"final","image_index":0,"format":"png","data":{"b64_json":"..."}}

data: [DONE]
```

### 圖片編輯

**POST** `/v1/images/edits`

使用 FLUX.1-Kontext-dev 或相似模型執行圖片編輯請求（img2img）。

**請求（multipart/form-data）：**

* `image`：file，必填 - 要編輯的輸入圖片
* `prompt`：string，必填 - 期望變更的文字描述
* `model`：string，必填 - 圖片編輯模型 ID（例如 `exolabs/FLUX.1-Kontext-dev`）
* `n`：integer（預設：1）- 要生成的編輯後圖片數量
* `size`：string（選填）- 輸出圖片尺寸
* `response_format`：string（預設："b64_json"）- `url` 或 `b64_json`
* `input_fidelity`：string（預設："low"）- `low` 或 `high` - 控制輸出與輸入圖片的貼近程度
* `stream`：string（預設："false"）- 啟用串流
* `partial_images`：string（預設："0"）- 要串流的部分圖片數
* `quality`：string（預設："medium"）- `high`、`medium` 或 `low`
* `output_format`：string（預設："png"）- `png`、`jpeg` 或 `webp`
* `advanced_params`：string（選填）- JSON 編碼進階參數

**回應：**
與 `/v1/images/generations` 相同格式。

### 基準化圖片生成

**POST** `/bench/images/generations`

與 `/v1/images/generations` 相同，但會額外回傳生成統計。

**請求主體：**
與 `/v1/images/generations` 相同 schema。

**回應：**
圖片生成加上基準指標，包含：

* `seconds_per_step` - 每個去噪步驟平均耗時
* `total_generation_time` - 總生成時間
* `num_inference_steps` - 使用的推論步數
* `num_images` - 生成圖片數量
* `image_width` - 輸出圖片寬度
* `image_height` - 輸出圖片高度
* `peak_memory_usage` - 生成期間峰值記憶體使用量

### 基準化圖片編輯

**POST** `/bench/images/edits`

與 `/v1/images/edits` 相同，但會額外回傳生成統計。

**請求：**
與 `/v1/images/edits` 相同 schema。

**回應：**
與 `/bench/images/generations` 相同格式，包含 `generation_stats`。

### 列出圖片

**GET** `/images`

列出所有已儲存圖片。

**回應：**
圖片中繼資料陣列，包含 URL 與到期時間。

### 取得圖片

**GET** `/images/{image_id}`

依 ID 取得已儲存圖片。

**路徑參數：**

* `image_id`：string，圖片 ID

**回應：**
具有正確 content type 的圖片檔案。

## 7. 完整端點摘要

```
# General
GET     /node_id
GET     /state
GET     /events

# Instance Management
POST    /instance
GET     /instance/await
GET     /instance/previews
GET     /instance/placement
GET     /instance/{instance_id}
DELETE  /instance/{instance_id}
POST    /place_instance

# Models
GET     /models
GET     /v1/models
POST    /models/add
DELETE  /models/custom/{model_id}
GET     /models/search

# Text Generation (OpenAI Chat Completions)
POST    /v1/chat/completions
POST    /bench/chat/completions

# Text Generation (Claude Messages API)
POST    /v1/messages

# Text Generation (OpenAI Responses API)
POST    /v1/responses

# Text Generation (Ollama API)
POST    /ollama/api/chat
POST    /ollama/api/api/chat
POST    /ollama/api/v1/chat
POST    /ollama/api/generate
GET     /ollama/api/tags
GET     /ollama/api/api/tags
GET     /ollama/api/v1/tags
POST    /ollama/api/show
GET     /ollama/api/ps
GET     /ollama/api/version
HEAD    /ollama/
HEAD    /ollama/api/version

# Command Control
POST    /v1/cancel/{command_id}

# Image Generation
POST    /v1/images/generations
POST    /bench/images/generations
POST    /v1/images/edits
POST    /bench/images/edits
GET     /images
GET     /images/{image_id}
```

## 8. 備註

### API 相容性

EXO 提供多種 API 相容介面：

* **OpenAI Chat Completions API** - 相容 OpenAI 客戶端與工具
* **Claude Messages API** - 相容 Anthropic Claude API 格式
* **OpenAI Responses API** - 相容 OpenAI Responses API 格式
* **Ollama API** - 相容 Ollama 與 OpenWebUI 等工具

既有 OpenAI、Claude 或 Ollama 客戶端只需調整 base URL 即可指向 EXO。

### 自訂模型

你可以透過 `/models/add` 端點從 HuggingFace 新增自訂模型。模型列表回應中的 `is_custom` 欄位可識別自訂模型。

**安全性：** 若模型需要 `trust_remote_code`，基於安全性必須明確啟用（預設為 false）。僅在你信任該模型遠端程式碼時啟用。

### 使用量統計

chat completion 回應包含以下使用量統計：

* `prompt_tokens` - 提示中的 token 數
* `completion_tokens` - 已生成 token 數
* `total_tokens` - 提示與生成 token 的總和

### 請求取消

你可以透過以下方式取消進行中的請求：

1. 關閉 HTTP 連線（串流請求）
2. 呼叫 `/v1/cancel/{command_id}`（任意請求）

伺服器會偵測取消並立即停止處理。

### 實例放置

實例放置端點可讓你在建立實例前先規劃與預覽叢集配置，有助於最佳化跨節點資源使用。

### 可觀測性

`/events` 與 `/state` 端點主要用於操作可視性與除錯。

# 為 EXO 做出貢獻

感謝你對貢獻 EXO 的興趣！

## 開始之前

從原始碼執行 EXO：

**先決條件：**
- [uv](https://github.com/astral-sh/uv)（用於 Python 相依套件管理）
  ```bash
  brew install uv
  ```
- [rust](https://github.com/rust-lang/rustup)（用於建置 Rust 綁定，目前需 nightly）
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup toolchain install nightly
  ```
- [macmon](https://github.com/vladkens/macmon)（用於 Apple Silicon 的硬體監控）
  請使用此儲存庫固定的 fork 版本，而不是 Homebrew 的 `macmon`。
  ```bash
  cargo install --git https://github.com/vladkens/macmon \
    --rev a1cd06b6cc0d5e61db24fd8832e74cd992097a7d \
    macmon \
    --force
  ```

```bash
git clone https://github.com/exo-explore/exo.git
cd exo/dashboard
npm install && npm run build && cd ..
uv run exo
```

## 開發

EXO 由 Rust、Python 與 TypeScript（儀表板使用 Svelte）混合建構，且程式碼庫仍在快速演進。在開始工作前：

- 拉取最新原始碼，確保你基於最新程式碼工作
- 讓你的變更聚焦 —— 每個 pull request 只實作一個功能或修一個問題
- 避免把不相關的變更混在一起，即使它們看起來很小

這能讓審查更快，也有助於我們在專案演進時維持程式碼品質。

## 程式碼風格

盡可能撰寫純函式。新增程式碼時，除非有充分理由，否則優先使用 Rust。充分利用可用的型別系統 —— Rust 的型別系統、Python 型別提示，以及 TypeScript 型別。註解應說明你「為什麼」這麼做，而不是程式碼「做了什麼」—— 尤其是在不直觀的決策上。

提交前請執行 `nix fmt` 自動格式化程式碼。

## 模型卡

EXO 使用以 TOML 為基礎的模型卡來定義模型中繼資料與能力。模型卡儲存在：
- `resources/inference_model_cards/`：文字生成模型
- `resources/image_model_cards/`：圖像生成模型
- `~/.exo/custom_model_cards/`：使用者新增的自訂模型

### 新增模型卡

要新增模型，請建立一個具備以下結構的 TOML 檔案：

```toml
model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
n_layers = 16
hidden_size = 2048
supports_tensor = true
tasks = ["TextGeneration"]
family = "llama"
quantization = "4bit"
base_model = "Llama 3.2 1B"
capabilities = ["text"]

[storage_size]
in_bytes = 729808896
```

### 必填欄位

- `model_id`：Hugging Face 模型識別碼
- `n_layers`：transformer 層數
- `hidden_size`：隱藏維度大小
- `supports_tensor`：模型是否支援 tensor parallelism
- `tasks`：支援任務清單（`TextGeneration`、`TextToImage`、`ImageToImage`）
- `family`：模型家族（例如 "llama"、"deepseek"、"qwen"）
- `quantization`：量化等級（例如 "4bit"、"8bit"、"bf16"）
- `base_model`：人類可讀的基礎模型名稱
- `capabilities`：能力清單（例如 `["text"]`、`["text", "thinking"]`）

### 選填欄位

- `components`：多元件模型使用（例如有獨立文字編碼器與 transformer 的影像模型）
- `uses_cfg`：模型是否使用 classifier-free guidance（影像模型）
- `trust_remote_code`：是否允許遠端程式碼執行（出於安全性預設為 `false`）

### Capabilities

`capabilities` 欄位定義模型能做什麼：
- `text`：標準文字生成
- `thinking`：模型支援 chain-of-thought 推理
- `thinking_toggle`：可透過 `enable_thinking` 參數啟用／停用 thinking
- `image_edit`：模型支援 image-to-image 編輯（FLUX.1-Kontext）

### 安全性說明

出於安全考量，`trust_remote_code` 預設為 `false`。僅在模型明確要求從 Hugging Face hub 執行遠端程式碼時才啟用。

## API 轉接器

EXO 透過 adapter pattern 支援多種 API 格式。轉接器會把各 API 的請求格式轉換為內部 `TextGenerationTaskParams` 格式，並把內部 token chunk 轉回各 API 的回應格式。

### 轉接器架構

所有轉接器都位於 `src/exo/master/adapters/`，並遵循相同模式：

1. 將 API 特定請求轉為 `TextGenerationTaskParams`
2. 同時處理串流與非串流回應產生
3. 將內部 `TokenChunk` 物件轉換為 API 特定格式
4. 管理錯誤處理與邊界情況

### 現有轉接器

- `chat_completions.py`：OpenAI Chat Completions API
- `claude.py`：Anthropic Claude Messages API
- `responses.py`：OpenAI Responses API
- `ollama.py`：Ollama API（相容 OpenWebUI）

### 新增 API 轉接器

要新增對新 API 格式的支援：

1. 在 `src/exo/master/adapters/` 建立新的轉接器檔案
2. 實作請求轉換函式：
   ```python
   def your_api_request_to_text_generation(
       request: YourAPIRequest,
   ) -> TextGenerationTaskParams:
       # Convert API request to internal format
       pass
   ```
3. 實作串流回應產生：
   ```python
   async def generate_your_api_stream(
       command_id: CommandId,
       chunk_stream: AsyncGenerator[TokenChunk | ErrorChunk | ToolCallChunk, None],
   ) -> AsyncGenerator[str, None]:
       # Convert internal chunks to API-specific streaming format
       pass
   ```
4. 實作非串流回應收集：
   ```python
   async def collect_your_api_response(
       command_id: CommandId,
       chunk_stream: AsyncGenerator[TokenChunk | ErrorChunk | ToolCallChunk, None],
   ) -> AsyncGenerator[str]:
       # Collect all chunks and return single response
       pass
   ```
5. 在 `src/exo/master/api.py` 註冊轉接器端點

轉接器模式可讓 API 特定邏輯與核心推論系統隔離。內部系統（worker、runner、event sourcing）只會看到 `TextGenerationTaskParams` 與 `TokenChunk` 物件 —— 不會有 API 特定型別跨越轉接器邊界。

詳細 API 文件請參考 [docs/api.md](docs/api.md)。

## 測試

目前專案階段中，EXO 仍高度依賴手動測試，但這正在改進。提交變更前，請在變更前後都進行測試，以展示你的改動如何改善行為。請在你可用的硬體條件下盡力測試 —— 若你需要測試協助，請提出，我們會盡力幫忙。也請盡可能新增自動化測試 —— 我們正在積極大幅提升自動化測試能力。

## 提交變更

1. Fork 此儲存庫
2. 建立功能分支（`git checkout -b feature/your-feature`）
3. 提交你的變更（`git commit -am 'Add some feature'`）
4. 推送分支（`git push origin feature/your-feature`）
5. 開啟 Pull Request 並遵循 PR 範本

## 回報問題

若你發現 bug 或有功能需求，請在 GitHub 開 issue，並附上：
- 對問題或功能的清楚描述
- 重現步驟（針對 bug）
- 預期行為與實際行為
- 你的環境（macOS 版本、硬體等）

## 有問題嗎？

加入我們的社群：
- [X](https://x.com/exolabs)

<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/imgs/exo-logo-black-bg.jpg">
  <img alt="exo logo" src="/docs/imgs/exo-logo-transparent.png" width="50%" height="50%">
</picture>

exo：在本機執行前沿 AI。由 [exo labs](https://x.com/exolabs) 維護。

<p align="center">
  <a href="https://discord.gg/TJ4P57arEm" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://x.com/exolabs" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/twitter/follow/exolabs?style=social" alt="X"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0.html" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-Apache2.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

</div>

---

exo 會把你的所有裝置連成一個 AI 叢集。exo 不只讓你能執行單一裝置放不下的大模型，還提供 [首日支援 Thunderbolt 上的 RDMA](https://x.com/exolabs/status/2001817749744476256?s=20)，讓模型在新增裝置時跑得更快。

## Features

- **Automatic Device Discovery**：執行 exo 的裝置會自動互相發現，無需手動設定。
- **RDMA over Thunderbolt**：exo 提供 [首日支援 Thunderbolt 5 上的 RDMA](https://x.com/exolabs/status/2001817749744476256?s=20)，可將裝置間延遲降低 99%。
- **Topology-Aware Auto Parallel**：exo 會根據即時裝置拓撲，自動找出把模型切分到所有可用裝置的最佳方式，並考量每條連線的資源、延遲與頻寬。
- **Tensor Parallelism**：exo 支援模型分片，在 2 台裝置上最高可加速 1.8 倍，在 4 台裝置上最高可加速 3.2 倍。
- **MLX Support**：exo 使用 [MLX](https://github.com/ml-explore/mlx) 作為推論後端，並使用 [MLX distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) 做分散式通訊。
- **Multiple API Compatibility**：相容 OpenAI Chat Completions API、Claude Messages API、OpenAI Responses API 與 Ollama API，可沿用既有工具與客戶端。
- **Custom Model Support**：可從 HuggingFace hub 載入自訂模型，擴充可用模型範圍。

## Dashboard

exo 內建儀表板，可用來管理叢集並與模型對話。

<p align="center">
  <img src="docs/imgs/dashboard-cluster-view.png" alt="exo dashboard - cluster view showing 4 x M3 Ultra Mac Studio with DeepSeek v3.1 and Kimi-K2-Thinking loaded" width="80%" />
</p>
<p align="center"><em>4 × 512GB M3 Ultra Mac Studio running DeepSeek v3.1 (8-bit) and Kimi-K2-Thinking (4-bit)</em></p>

## Benchmarks

<details>
  <summary>Qwen3-235B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-1-qwen3-235b.jpeg" alt="Benchmark - Qwen3-235B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

<details>
  <summary>DeepSeek v3.1 671B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-2-deepseek-3.1-671b.jpeg" alt="Benchmark - DeepSeek v3.1 671B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

<details>
  <summary>Kimi K2 Thinking (native 4-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-3-kimi-k2-thinking.jpeg" alt="Benchmark - Kimi K2 Thinking (native 4-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

---

## Quick Start

執行 exo 的裝置會自動互相發現，不需要任何手動設定。每台裝置都會提供一個 API 與儀表板讓你操作叢集（執行於 `http://localhost:52415`）。

執行 exo 有兩種方式：

### Run from Source (macOS)

若你已安裝 [Nix](https://nixos.org/)，可跳過下方大部分步驟，直接執行 exo：

```bash
nix run .#exo
```

**Note:** To accept the Cachix binary cache (and avoid the Xcode Metal ToolChain), add to `/etc/nix/nix.conf`:
```
trusted-users = root    (or your username)
experimental-features = nix-command flakes
```
Then restart the Nix daemon: `sudo launchctl kickstart -k system/org.nixos.nix-daemon`

**Prerequisites:**
- [Xcode](https://developer.apple.com/xcode/)（提供 MLX 編譯所需的 Metal ToolChain）
- [brew](https://github.com/Homebrew/brew)（macOS 上方便的套件管理）

  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- [uv](https://github.com/astral-sh/uv)（Python 相依套件管理）
- [node](https://github.com/nodejs/node)（建置 dashboard）

  ```bash
  brew install uv node
  ```
- [rust](https://github.com/rust-lang/rustup)（建置 Rust 綁定，目前需 nightly）

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup toolchain install nightly
  ```
- [macmon](https://github.com/vladkens/macmon)（Apple Silicon 的硬體監控）

  請安裝此儲存庫固定的 fork 版本，不要用 Homebrew 的 `macmon`。
  Homebrew `macmon 0.6.1` 在 Apple M5 上仍會崩潰。

  ```bash
  cargo install --git https://github.com/vladkens/macmon \
    --rev a1cd06b6cc0d5e61db24fd8832e74cd992097a7d \
    macmon \
    --force
  ```

Clone the repo, build the dashboard, and run exo:

```bash
# Clone exo
git clone https://github.com/exo-explore/exo

# Build dashboard
cd exo/dashboard && npm install && npm run build && cd ..

# Run exo
uv run exo
```

這會在 http://localhost:52415/ 啟動 exo dashboard 與 API。


*請查看 RDMA 章節以在 MacOS >=26.2 啟用此功能！*


### Run from Source (Linux)

**Prerequisites:**

- [uv](https://github.com/astral-sh/uv)（Python 相依套件管理）
- [node](https://github.com/nodejs/node)（建置 dashboard）- 版本 18 或以上
- [rust](https://github.com/rust-lang/rustup)（建置 Rust 綁定，目前需 nightly）

**Installation methods:**

**Option 1: Using system package manager (Ubuntu/Debian example):**
```bash
# Install Node.js and npm
sudo apt update
sudo apt install nodejs npm

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust (using rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
```

**Option 2: Using Homebrew on Linux (if preferred):**
```bash
# Install Homebrew on Linux
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install uv node

# Install Rust (using rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
```

**Note:** `macmon` 套件僅支援 macOS，Linux 不需要。

Clone the repo, build the dashboard, and run exo:

```bash
# Clone exo
git clone https://github.com/exo-explore/exo

# Build dashboard
cd exo/dashboard && npm install && npm run build && cd ..

# Run exo
uv run exo
```

這會在 http://localhost:52415/ 啟動 exo dashboard 與 API。

**給 Linux 使用者的重要說明：** 目前 exo 在 Linux 上是以 CPU 執行。Linux 平台 GPU 支援正在開發中。如果你希望支援特定 Linux 硬體，請先[搜尋現有功能請求](https://github.com/exo-explore/exo/issues)，或建立新的請求。

**Configuration Options:**

- `--no-worker`：在不啟動 worker 元件的情況下執行 exo。適用於只負責網路與協調、不執行推論任務的協調節點。對於 GPU 資源不足但網路連線良好的機器特別有用。

  ```bash
  uv run exo --no-worker
  ```

- `--legacy-daemon`：以傳統 SysV 風格背景 daemon 模式執行 exo（double-fork daemonization）。此選項用於舊式 init script；systemd 與 launchd 應在前景執行 exo，不需此旗標。

  ```bash
  uv run exo --legacy-daemon
  ```

**File Locations (Linux):**

exo 在 Linux 上遵循 [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)：

- **Configuration files**: `~/.config/exo/`（或 `$XDG_CONFIG_HOME/exo/`）
- **Data files**: `~/.local/share/exo/`（或 `$XDG_DATA_HOME/exo/`）
- **Cache files**: `~/.cache/exo/`（或 `$XDG_CACHE_HOME/exo/`）
- **Log files**: `~/.cache/exo/exo_log/`（含自動 log rotation）
- **Custom model cards**: `~/.local/share/exo/custom_model_cards/`

你可以透過設定對應的 XDG 環境變數來覆寫這些位置。

### macOS App

exo 提供一個可在 Mac 背景執行的 macOS app。

<img src="docs/imgs/macos-app-one-macbook.png" alt="exo macOS App - running on a MacBook" width="35%" />

macOS app 需要 macOS Tahoe 26.2 或更新版本。

在此下載最新版本：[EXO-latest.dmg](https://assets.exolabs.net/EXO-latest.dmg)。

你也可以使用 Homebrew 安裝最新版本：

```bash
brew install --cask exo
```

App 會要求修改系統設定並安裝新的網路設定檔權限。這部分仍在持續改進中。

**Custom Namespace for Cluster Isolation:**

macOS app 內建自訂 namespace 功能，可將你的 exo 叢集與同網路上的其他叢集隔離。此功能透過 `EXO_LIBP2P_NAMESPACE` 設定：

- **Use cases**:
  - 在同一網路上執行多個彼此獨立的 exo 叢集
  - 將開發/測試叢集與正式叢集隔離
  - 防止意外加入其他叢集

- **Configuration**：可在 app 的 Advanced 設定中調整（或在從原始碼執行時設定 `EXO_LIBP2P_NAMESPACE` 環境變數）

為了除錯，啟動時會記錄 namespace。

#### Uninstalling the macOS App

建議透過 app 本身解除安裝：點選選單列圖示 → Advanced → Uninstall。這會乾淨移除所有系統元件。

如果你已刪除 app，可執行獨立解除安裝腳本：

```bash
sudo ./app/EXO/uninstall-exo.sh
```

這會移除：
- Network setup LaunchDaemon
- Network configuration script
- Log files
- "exo" network location

**Note:** 你仍需到 System Settings → General → Login Items 手動移除 EXO。

---

### Enabling RDMA on macOS

RDMA 是 macOS 26.2 新增能力。它可在任何具 Thunderbolt 5 的 Mac 上運作（M4 Pro Mac Mini、M4 Max Mac Studio、M4 Max MacBook Pro、M3 Ultra Mac Studio）。

請參考 caveats 以進行即時故障排除。

在 macOS 啟用 RDMA，請依下列步驟：

1. 關機。
2. 按住電源鍵 10 秒直到開機選單出現。
3. 選擇 "Options" 進入 Recovery mode。
4. 進入 Recovery UI 後，從 Utilities 選單開啟 Terminal。
5. 在 Terminal 輸入：
   ```
   rdma_ctl enable
   ```
   然後按 Enter。
6. 重新開機。

完成後，macOS 將啟用 RDMA，剩下的會由 exo 處理。

**Important Caveats**

1. 要加入 RDMA 叢集的裝置，必須與叢集中所有其他裝置互相連接。
2. 連接線必須支援 TB5。
3. Mac Studio 不能使用 Ethernet 埠旁邊的 Thunderbolt 5 埠。
4. 若從原始碼執行，請使用 `tmp/set_rdma_network_config.sh` 腳本，它會停用 Thunderbolt Bridge 並在每個 RDMA 埠設定 dhcp。
5. 不同 MacOS 版本的 RDMA 埠可能無法彼此發現。請確保所有裝置的 OS 版本完全一致（包含 beta 版本號）。

---

## Environment Variables

exo 支援多個環境變數做設定：

| Variable | Description | Default |
|----------|-------------|---------|
| `EXO_DEFAULT_MODELS_DIR` | 模型下載與快取的預設目錄。永遠是可寫入目錄清單中的第一個。 | `~/.local/share/exo/models` (Linux) 或 `~/.exo/models` (macOS) |
| `EXO_MODELS_DIRS` | 以冒號分隔的額外可寫入模型下載目錄。會在預設目錄後依序檢查；使用第一個有足夠可用空間的目錄。 | None |
| `EXO_MODELS_READ_ONLY_DIRS` | 以冒號分隔的唯讀目錄，用於搜尋預先下載模型（例如 NFS 掛載、共享儲存）。這裡的模型不可刪除。 | None |
| `EXO_OFFLINE` | 離線執行（僅使用本機模型） | `false` |
| `EXO_ENABLE_IMAGE_MODELS` | 啟用圖像模型支援 | `false` |
| `EXO_LIBP2P_NAMESPACE` | 用於叢集隔離的自訂 namespace | None |
| `EXO_FAST_SYNCH` | 控制 MLX_METAL_FAST_SYNCH 行為（JACCL 後端） | Auto |
| `EXO_TRACING_ENABLED` | 啟用分散式 tracing 做效能分析 | `false` |

**Example usage:**

```bash
# Use pre-downloaded models from NFS mount (read-only)
EXO_MODELS_READ_ONLY_DIRS=/mnt/nfs/models:/opt/ai-models uv run exo

# Download models to an external SSD (falls back to default dir if full)
EXO_MODELS_DIRS=/Volumes/ExternalSSD/exo-models uv run exo

# Run in offline mode
EXO_OFFLINE=true uv run exo

# Enable image models
EXO_ENABLE_IMAGE_MODELS=true uv run exo

# Use custom namespace for cluster isolation
EXO_LIBP2P_NAMESPACE=my-dev-cluster uv run exo
```

---

### Using the API

exo 提供多種 API 相容介面，以便最大化與既有工具相容：

- **OpenAI Chat Completions API** - 相容 OpenAI 客戶端
- **Claude Messages API** - 相容 Anthropic 的 Claude 格式
- **OpenAI Responses API** - 相容 OpenAI 的 Responses 格式
- **Ollama API** - 相容 Ollama 與 OpenWebUI 等工具

如果你偏好透過 API 與 exo 互動，以下範例示範建立小模型（`mlx-community/Llama-3.2-1B-Instruct-4bit`）實例、送出 chat completions 請求，並刪除該實例。

---

**1. 預覽 instance 佈署**

`/instance/previews` 端點會預覽模型所有有效佈署方式。

```bash
curl "http://localhost:52415/instance/previews?model_id=llama-3.2-1b"
```

範例回應：

```json
{
  "previews": [
    {
      "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "sharding": "Pipeline",
      "instance_meta": "MlxRing",
      "instance": {...},
      "memory_delta_by_node": {"local": 729808896},
      "error": null
    }
    // ...possibly more placements...
  ]
}
```

這會回傳該模型所有可用佈署。請選擇你想要的配置。
若要選第一個，可透過 `jq` 管線：

```bash
curl "http://localhost:52415/instance/previews?model_id=llama-3.2-1b" | jq -c '.previews[] | select(.error == null) | .instance' | head -n1
```

---

**2. 建立模型 instance**

對 `/instance` 發送 POST，並在 `instance` 欄位放入你要的佈署（完整 payload 必須符合 `CreateInstanceParams` 型別），可從步驟 1 複製：

```bash
curl -X POST http://localhost:52415/instance \
  -H 'Content-Type: application/json' \
  -d '{
    "instance": {...}
  }'
```


範例回應：

```json
{
  "message": "Command received.",
  "command_id": "e9d1a8ab-...."
}
```

這個指令是非同步的。送出推論請求前，先等 API 能看到此模型的新 instance：

```bash
curl -N "http://localhost:52415/instance/await?model_id=mlx-community/Llama-3.2-1B-Instruct-4bit"
```

這個端點會回傳 SSE stream。成功等待會送出含
`"type": "ready"` 與對應 instance 的訊息；逾時則送出 `"type": "timeout"`。
預設會無限等待。可設定 `timeout_seconds` 為正值限制等待時間。

---

**3. 傳送 chat completion**

接著對 `/v1/chat/completions` 發送 POST（格式與 OpenAI API 相同）：

```bash
curl -N -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "What is Llama 3.2 1B?"}
    ],
    "stream": true
  }'
```

---

**4. 刪除 instance**

完成後，透過 ID 刪除 instance（可從 `/state` 或 `/instance` 端點取得）：

```bash
curl -X DELETE http://localhost:52415/instance/YOUR_INSTANCE_ID
```

### Claude Messages API Compatibility

使用 `/v1/messages` 端點採用 Claude Messages API 格式：

```bash
curl -N -X POST http://localhost:52415/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 1024,
    "stream": true
  }'
```

### OpenAI Responses API Compatibility

使用 `/v1/responses` 端點採用 OpenAI Responses API 格式：

```bash
curl -N -X POST http://localhost:52415/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": true
  }'
```

### Ollama API Compatibility

exo 支援 Ollama API 端點，以相容 OpenWebUI 等工具：

```bash
# Ollama chat
curl -X POST http://localhost:52415/ollama/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": false
  }'

# List models (Ollama format)
curl http://localhost:52415/ollama/api/tags
```

### Custom Model Loading from HuggingFace

你可以從 HuggingFace hub 新增自訂模型：

```bash
curl -X POST http://localhost:52415/models/add \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "mlx-community/my-custom-model"
  }'
```

**Security Note:**

為了安全性，若自訂模型在設定中需要 `trust_remote_code`，必須明確啟用（預設為 false）。僅在你信任該模型遠端程式碼執行時才啟用。模型會從 HuggingFace 取得，並以自訂模型卡形式儲存在本機。

**Other useful API endpoints*:*

- 列出所有模型：`curl http://localhost:52415/models`
- 只列出已下載模型：`curl http://localhost:52415/models?status=downloaded`
- 搜尋 HuggingFace：`curl "http://localhost:52415/models/search?query=llama&limit=10"`
- 檢查 instance ID 與部署狀態：`curl http://localhost:52415/state`

更多細節請參考：

- [docs/api.md](docs/api.md) 中的 API 文件。
- [src/exo/master/api.py](src/exo/master/api.py) 中的 API 型別與端點。

---

## Benchmarking

`exo-bench` 工具可測量不同佈署配置下的模型 prefill 與 token 生成速度。這有助於你最佳化模型效能並驗證改進。

**Prerequisites:**
- 執行基準測試前，節點應先以 `uv run exo` 啟動
- 工具使用 `/bench/chat/completions` 端點

**Basic usage:**

```bash
uv run bench/exo_bench.py \
  --model Llama-3.2-1B-Instruct-4bit \
  --pp 128,256,512 \
  --tg 128,256
```

**Key parameters:**

- `--model`：要測試的模型（短 ID 或 HuggingFace ID）
- `--pp`：Prompt 大小提示（以逗號分隔的整數）
- `--tg`：生成長度（以逗號分隔的整數）
- `--max-nodes`：將佈署限制在 N 個節點（預設：4）
- `--instance-meta`：以 `ring`、`jaccl` 或 `both` 篩選（預設：both）
- `--sharding`：以 `pipeline`、`tensor` 或 `both` 篩選（預設：both）
- `--repeat`：每個配置重複次數（預設：1）
- `--warmup`：每個佈署的暖機次數（預設：0）
- `--json-out`：結果輸出檔案（預設：bench/results.json）

**Example with filters:**

```bash
uv run bench/exo_bench.py \
  --model Llama-3.2-1B-Instruct-4bit \
  --pp 128,512 \
  --tg 128 \
  --max-nodes 2 \
  --sharding tensor \
  --repeat 3 \
  --json-out my-results.json
```

工具會輸出效能指標，包含每秒 prompt token（prompt_tps）、每秒生成 token（generation_tps），以及各配置的峰值記憶體使用量。

---

## Hardware Accelerator Support

在 macOS 上，exo 使用 GPU。在 Linux 上，exo 目前以 CPU 執行。我們正持續擴充硬體加速器支援。若你希望支援新的硬體平台，請先[搜尋既有功能請求](https://github.com/exo-explore/exo/issues)，並按讚讓我們了解社群重視哪些硬體。

---

## Contributing

如何為 exo 貢獻，請見 [CONTRIBUTING.md](CONTRIBUTING.md)。

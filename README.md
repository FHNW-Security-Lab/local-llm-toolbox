<div align="center">
  <picture align="center">
    <source srcset="./assets/logo.png">
    <img alt="Local LLM Toolbox Icon" src="./assets/logo.png" height="100" style="max-width: 100%;">
  </picture>
    <div>
      <h1>Local LLM Toolbox</h1><br>
      <p>A tool for experimenting with local LLM hosting and management tools.</p>
    </div>
</div>

## Features

- **Multiple Backends**: Switch between llama.cpp and Microsoft Foundry
- **Web Dashboard**: Browser-based UI for managing backends and models
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API at `/v1`
- **RPC Clustering**: Distribute inference across multiple machines (llama.cpp)

## Supported Backends

| Backend | Platform | Model Format |
|---------|----------|--------------|
| **llama.cpp** | macOS, Linux | GGUF |
| **Foundry** | macOS, Windows | ONNX |

## Quick Start

### Prerequisites

- [Nix](https://nixos.org/download.html) package manager
- For llama.cpp: `llama-server` in PATH (installed via Nix flake)
- For Foundry: `foundry` CLI and `foundry-local-sdk` (macOS/Windows only)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/local-llm-toolbox.git
cd local-llm-toolbox

# Enter the development environment (auto-detects GPU)
./dev

# Start the toolbox
./toolbox serve
```

The `./dev` script auto-detects your GPU and selects the appropriate environment:
- **macOS**: Metal (Apple Silicon)
- **Linux with NVIDIA**: CUDA
- **Linux with AMD/Intel**: Vulkan
- **No GPU**: CPU-only with BLAS

You can also select a specific environment directly:

```bash
./dev nvidia   # CUDA (NVIDIA GPU)
./dev vulkan   # Vulkan (AMD/Intel)
./dev cpu      # CPU only (BLAS)
```

Or use Nix directly:

```bash
nix develop           # Default (Vulkan on Linux, Metal on macOS)
nix develop .#nvidia  # CUDA
nix develop .#vulkan  # Vulkan
nix develop .#cpu     # CPU only
```

The dashboard will be available at http://localhost:8090

### Adding Models

**For llama.cpp (GGUF models):**

Download GGUF files to `~/.local/share/models/`:

```bash
# Example: Download from HuggingFace
cd ~/.local/share/models
wget https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
```

**For Foundry (ONNX models):**

Use the download command or the web dashboard:

```bash
./toolbox download phi-4 foundry
```

## Usage

### Web Dashboard

Start the toolbox and open http://localhost:8090 in your browser:

```bash
./toolbox serve
```

From the dashboard you can:
- Start/stop backends
- Browse and load models
- Chat with loaded models
- Configure RPC clustering (llama.cpp)
- Monitor system resources

### CLI Commands

```bash
# Start the dashboard + API server
./toolbox serve
./toolbox serve --debug    # Enable verbose logging

# Backend management
./toolbox start llama      # Start llama.cpp backend
./toolbox start foundry    # Start Foundry backend
./toolbox stop             # Stop the active backend
./toolbox status           # Show current status

# Model management
./toolbox models           # List available models
./toolbox models llama     # List models for specific backend
./toolbox load <model>     # Load a model
./toolbox unload           # Unload current model
./toolbox download <model> # Download a model (Foundry only)

# RPC worker (run on worker machines for distributed inference)
./toolbox rpc llama        # Start as RPC worker
```

### OpenAI-Compatible API

The API is available at `http://localhost:8090/v1` and supports:

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `POST /v1/completions` - Text completions
- `POST /v1/embeddings` - Embeddings (if supported by backend)

Example with curl:

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Example with OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="not-needed"  # No auth required
)

response = client.chat.completions.create(
    model="qwen2.5-1.5b-instruct-q4_k_m",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Configuration

Copy `config.env.example` to `.env` and adjust as needed:

```bash
cp config.env.example .env
```

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHBOARD_PORT` | 8090 | Web dashboard and API port |
| `LLAMA_MODELS_DIR` | ~/.local/share/models | Where to look for GGUF models |
| `LLAMA_PORT` | 8080 | llama-server port |
| `LLAMA_CTX_SIZE` | 8192 | Context window size |
| `LLAMA_GPU_LAYERS` | 99 | Layers to offload to GPU |
| `FOUNDRY_PORT` | 5273 | Foundry service port |

See `config.env.example` for all available options.

## RPC Clustering (llama.cpp)

Distribute inference across multiple machines using llama.cpp's RPC feature.

### Setup

**1. On each worker machine:**

Clone the repo and start the RPC worker:

```bash
git clone https://github.com/your-org/local-llm-toolbox.git
cd local-llm-toolbox
./dev
./toolbox rpc llama
```

The worker will start:
- RPC server on port 50052 (tensor offload)
- Control API on port 50053 (management)

**2. On the main machine:**

Configure workers in `.env`:

```bash
LLAMA_RPC_WORKERS=192.168.1.10,192.168.1.11
```

Then start normally:

```bash
./toolbox serve
./toolbox start llama
./toolbox load my-model
```

The main node will automatically reset all workers before loading each model, ensuring a clean state.

### How It Works

- Workers expose GPU/CPU compute via llama.cpp's `rpc-server`
- The main node's control API resets workers before each model load
- Model weights are distributed across all nodes proportionally to available memory
- Tensor caching (`-c` flag) speeds up repeated model loads

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_RPC_WORKERS` | (empty) | Comma-separated worker IPs/hostnames |
| `LLAMA_RPC_CONTROL_PORT` | 50053 | Control API port on workers |

### Notes

- Workers must be started manually on each machine
- If the main node crashes, workers may need a restart to clear leaked GPU memory
- Use `./toolbox rpc llama --help` for worker options

# llama.cpp

Local LLM inference using [llama.cpp](https://github.com/ggerganov/llama.cpp). Provides `llama-server` with automatic GPU detection via Nix, supporting router mode (multi-model management with built-in web UI).

## Prerequisites

- [Nix](https://nixos.org/download.html) with flakes enabled
- For Linux GPU setup, see the [Hardware Setup Guide](../SETUP.md)

## Quick Start

```bash
# Enter the dev environment (auto-detects GPU)
./dev

# Start llama-server in router mode with web UI
llama-server --models-dir ~/models -ngl 99 -c 8192
```

The web UI is available at http://localhost:8080.

### Select a specific GPU backend

```bash
./dev rocm      # ROCm (AMD)
./dev nvidia    # CUDA (NVIDIA)
./dev vulkan    # Vulkan (AMD/Intel)
./dev cpu       # CPU only (BLAS)
```

Or use Nix directly:

```bash
nix develop           # Default (Vulkan on Linux, Metal on macOS)
nix develop .#rocm    # ROCm
nix develop .#nvidia  # CUDA
nix develop .#cpu     # CPU only
```

## Router Mode

llama-server's router mode manages multiple GGUF models from a directory. Models are loaded on demand and evicted via LRU when the limit is reached.

```bash
llama-server \
  --models-dir ~/models \
  --models-max 4 \
  -ngl 99 \
  -c 8192 \
  --port 8080
```

| Flag | Description |
|------|-------------|
| `--models-dir PATH` | Directory containing GGUF files (scanned on startup) |
| `--models-max N` | Max models loaded simultaneously (default: 4, LRU eviction) |
| `-ngl N` | GPU layers to offload (99 = all) |
| `-c N` | Context size (default: 2048) |
| `--port N` | Server port (default: 8080) |

### API

The server exposes an OpenAI-compatible API:

```bash
# List available models
curl http://localhost:8080/v1/models

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-35B-A3B-UD-Q8_K_XL",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Works with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="Qwen3.5-35B-A3B-UD-Q8_K_XL",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Downloading Models

Place GGUF files in your models directory (default: `~/models` or `~/.local/share/models`).

### Bulk download

The included script downloads a curated set of models from HuggingFace:

```bash
./download-models.sh
```

This downloads models to `~/.local/share/models/`, including multi-shard models into subdirectories. Edit the script to customize which models to download.

### Manual download

```bash
cd ~/models
wget https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
```

## RPC Distributed Inference

Split model inference across multiple machines using llama.cpp's RPC protocol. Useful for models that exceed a single machine's memory.

### On each worker machine

```bash
cd llama-cpp && ./dev
llama-rpc-server --host 0.0.0.0 --port 50053
```

### On the main machine

```bash
cd llama-cpp && ./dev
llama-server \
  --models-dir ~/models \
  -ngl 99 \
  -c 8192 \
  --rpc worker1:50053,worker2:50053
```

See the [Hardware Setup Guide](../SETUP.md) for network configuration and detailed RPC cluster setup.

## Nix Flake Details

The `flake.nix` provides dev shells for each GPU backend, all with `rpcSupport = true`:

| Shell | GPU Acceleration |
|-------|-----------------|
| `default` | Vulkan (Linux) / Metal (macOS) |
| `nvidia` | CUDA |
| `vulkan` | Vulkan |
| `rocm` | ROCm with Flash Attention (rocWMMA) |
| `cpu` | CPU only (BLAS) |

The flake also includes `systemConfigs` for setting up `/run/opengl-driver` on non-NixOS Linux via [nix-system-graphics](https://github.com/soupglasses/nix-system-graphics).

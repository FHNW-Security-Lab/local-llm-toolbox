<div align="center">
  <picture align="center">
    <source srcset="./assets/logo.png">
    <img alt="Local LLM Toolbox Icon" src="./assets/logo.png" height="100" style="max-width: 100%;">
  </picture>
    <div>
      <h1>Local LLM Toolbox</h1><br>
      <p>Unified interface for running local LLM backends.</p>
    </div>
</div>

## Features

- **Single CLI** to manage all backends
- **Router** provides one OpenAI-compatible endpoint regardless of backend
- **Auto-detection** of running backends
- Works with Open WebUI or any OpenAI-compatible client

## Backends

| Backend | Description |
|---------|-------------|
| `llama-cluster` | llama.cpp with distributed RPC support |
| `ollama` | Ollama server |
| `vllm` | vLLM server |

## Quick Start

```bash
# Enter nix shell
nix develop

# Start a backend
./toolbox start llama-cluster

# Check status
./toolbox status

# Start router (point Open WebUI here)
./toolbox router

# Switch backends
./toolbox start ollama --force

# Stop
./toolbox stop
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Open WebUI   │────▶│   Router     │────▶│ active backend  │
│ (or any      │     │   :5000      │     │ (auto-detected) │
│  client)     │     │              │     │                 │
└──────────────┘     └──────────────┘     └─────────────────┘
```

The router auto-detects which backend is running and proxies `/v1/*` requests to it.

## Configuration

Copy `config.env.example` to `config.env` and adjust as needed.

Backend-specific config lives in `backends/<name>/config.env`.

## Documentation

- [llama-cluster setup](./backends/llama-cluster/docs/setup.md)
- [FHNW hardware setup](./backends/llama-cluster/docs/fhnw-setup.md)

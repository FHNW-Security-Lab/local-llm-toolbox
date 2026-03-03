# Local LLM Toolbox

A collection of lightweight setups for running LLMs locally.

Each subfolder is a self-contained environment for a specific inference backend. Pick the one that fits your hardware.

## Backends

| Backend | Platform | Model Format | Description |
|---------|----------|--------------|-------------|
| [**llama.cpp**](llama-cpp/) | macOS, Linux | GGUF | Nix-based dev environment with GPU auto-detection, router mode, RPC clustering |
| [**Foundry Local**](foundry/) | macOS, Windows | ONNX | Microsoft's ONNX Runtime optimized inference |

## Hardware Setup

See the [Hardware Setup Guide](SETUP.md) for GPU driver configuration, AMD memory tuning, and distributed inference networking.

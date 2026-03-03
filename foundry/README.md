# Microsoft Foundry Local

Local LLM inference using [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local). Runs ONNX-optimized models with ONNX Runtime on macOS and Windows.

## Prerequisites

- macOS or Windows
- Foundry CLI installed

## Installation

**macOS:**

```bash
brew install microsoft/foundrylocal/foundrylocal
```

**Windows:**

```powershell
winget install Microsoft.FoundryLocal
```

## Quick Start

```bash
# List available models
foundry model list

# Download a model
foundry model download phi-4

# Start the service and load a model
foundry model run phi-4
```

The server starts automatically and exposes an OpenAI-compatible API.

## API

Foundry Local provides an OpenAI-compatible endpoint:

```bash
# Check the service endpoint
foundry service status

# Chat completion
curl http://localhost:5273/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Works with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5273/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="phi-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Model Management

```bash
foundry model list              # List catalog models
foundry model list --cached     # List downloaded models
foundry model download <model>  # Download a model
foundry model run <model>       # Load and run a model
```

## Python SDK

For programmatic access:

```bash
pip install foundry-local-sdk
```

```python
from foundry_local import FoundryLocalManager

manager = FoundryLocalManager()
manager.download_model("phi-4")
manager.load_model("phi-4")

print(f"API endpoint: {manager.endpoint}")
```

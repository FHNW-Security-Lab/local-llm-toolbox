# Chat Templates for llama-server

Override templates for models whose GGUF-embedded templates are broken or
missing tool-calling support.

## Usage

```bash
llama-server --model-preset configs/templates/model-presets.conf
```

## Model Status

| Model | Template | Status |
|-------|----------|--------|
| Qwen3.5-35B-A3B | `qwen3.5-tools.jinja` | Official template has broken `arguments \| items` iteration |
| Qwen3.5-122B-A10B | `qwen3.5-tools.jinja` | Same bug as above |
| Qwen3.5-397B-A17B | `qwen3.5-tools.jinja` | Same bug as above |
| GLM-4.7 | `glm4-tools.jinja` | Template doesn't inject tool descriptions |
| GLM-4.7-Flash-REAP | `glm4-tools.jinja` | Same family, same bug |
| DeepSeek-V3.1-Terminus | `deepseek-tools.jinja` | Uses special DeepSeek tokens, needs llama.cpp override |
| Devstral-Small-2 | `devstral-tools.jinja` | Role alternation + tool call parsing bugs |
| Llama-4-Scout | `llama4-tools.jinja` | No native handler in llama.cpp, pythonic format |
| Qwen3-Coder-Next | *embedded* | Unsloth GGUF has fixed template |
| Qwen3-Coder-30B-A3B | *embedded* | Unsloth GGUF has fixed template |
| Nemotron-3-Nano | *embedded* | Native llama.cpp support |
| MiniMax-M2.5 | *embedded* | Works with generic `--jinja` handler |

## Sources

- Qwen3.5: [barubary/qwen3.5-barubary-attuned-chat-template](https://huggingface.co/barubary/qwen3.5-barubary-attuned-chat-template)
- GLM-4: [unsloth/GLM-4.5-Air-GGUF/discussions/9](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/discussions/9)
- DeepSeek: [llama.cpp/models/templates/llama-cpp-deepseek-r1.jinja](https://github.com/ggml-org/llama.cpp/blob/master/models/templates/llama-cpp-deepseek-r1.jinja)
- Devstral: [wonderfuldestruction/devstral-small-2-template-fix](https://github.com/wonderfuldestruction/devstral-small-2-template-fix) + Unsloth fixes
- Llama-4: [vllm-project/vllm tool_chat_template_llama4_pythonic.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama4_pythonic.jinja)

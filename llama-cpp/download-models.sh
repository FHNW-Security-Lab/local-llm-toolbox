#!/usr/bin/env bash
#
# Download GGUF models into ~/.local/share/models/
# Multi-shard models go into subdirectories, single files go directly.
#
# Requires: curl (with resume support via -C -)
#
set -e

MODELS_DIR="${HOME}/.local/share/models"
HF="https://huggingface.co"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

download() {
    local url="$1"
    local dest="$2"

    if [[ -f "$dest" ]]; then
        echo -e "  ${GREEN}✓${NC} $(basename "$dest") (already exists)"
        return
    fi

    echo -e "  ${CYAN}↓${NC} $(basename "$dest")"
    curl -L -C - --progress-bar -o "${dest}.part" "$url"
    mv "${dest}.part" "$dest"
}

echo ""
echo -e "${CYAN}=== Model Downloader ===${NC}"
echo -e "Target: ${MODELS_DIR}"
echo ""

mkdir -p "$MODELS_DIR"

# ── Single file: Qwen3.5-35B-A3B-UD-Q8_K_XL (~39 GB) ───────────────────────

echo -e "${YELLOW}[1/7] Qwen3.5-35B-A3B-UD-Q8_K_XL (~39 GB)${NC}"
download \
    "${HF}/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf" \
    "${MODELS_DIR}/Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf"
echo ""

# ── Multi-shard: Qwen3.5-122B-A10B-UD-Q8_K_XL (~155 GB, 4 shards) ──────────

echo -e "${YELLOW}[2/7] Qwen3.5-122B-A10B-UD-Q8_K_XL (~155 GB)${NC}"
DIR="${MODELS_DIR}/Qwen3.5-122B-A10B-UD-Q8_K_XL"
mkdir -p "$DIR"
for i in $(seq -w 1 4); do
    download \
        "${HF}/unsloth/Qwen3.5-122B-A10B-GGUF/resolve/main/UD-Q8_K_XL/Qwen3.5-122B-A10B-UD-Q8_K_XL-0000${i}-of-00004.gguf" \
        "${DIR}/Qwen3.5-122B-A10B-UD-Q8_K_XL-0000${i}-of-00004.gguf"
done
echo ""

# ── Multi-shard: Qwen3-Coder-Next-BF16 (~159 GB, 4 shards) ─────────────────

echo -e "${YELLOW}[3/7] Qwen3-Coder-Next-BF16 (~159 GB)${NC}"
DIR="${MODELS_DIR}/Qwen3-Coder-Next-BF16"
mkdir -p "$DIR"
for i in $(seq -w 1 4); do
    download \
        "${HF}/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/BF16/Qwen3-Coder-Next-BF16-0000${i}-of-00004.gguf" \
        "${DIR}/Qwen3-Coder-Next-BF16-0000${i}-of-00004.gguf"
done
echo ""

# ── Multi-shard: Qwen3.5-397B-A17B-IQ4_XS (~223 GB, 6 shards) ──────────────

echo -e "${YELLOW}[4/7] Qwen3.5-397B-A17B-IQ4_XS (~223 GB)${NC}"
DIR="${MODELS_DIR}/Qwen3.5-397B-A17B-IQ4_XS"
mkdir -p "$DIR"
for i in $(seq -w 1 6); do
    download \
        "${HF}/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/IQ4_XS/Qwen3.5-397B-A17B-IQ4_XS-0000${i}-of-00006.gguf" \
        "${DIR}/Qwen3.5-397B-A17B-IQ4_XS-0000${i}-of-00006.gguf"
done
echo ""

# ── Multi-shard: GLM-4.7-UD-Q4_K_XL (~205 GB, 5 shards) ────────────────────

echo -e "${YELLOW}[5/7] GLM-4.7-UD-Q4_K_XL (~205 GB)${NC}"
DIR="${MODELS_DIR}/GLM-4.7-UD-Q4_K_XL"
mkdir -p "$DIR"
for i in $(seq -w 1 5); do
    download \
        "${HF}/unsloth/GLM-4.7-GGUF/resolve/main/UD-Q4_K_XL/GLM-4.7-UD-Q4_K_XL-0000${i}-of-00005.gguf" \
        "${DIR}/GLM-4.7-UD-Q4_K_XL-0000${i}-of-00005.gguf"
done
echo ""

# ── Single file: GLM-4.7-Flash-REAP-23B-A3B-BF16 (~46 GB) ──────────────────

echo -e "${YELLOW}[6/7] GLM-4.7-Flash-REAP-23B-A3B-BF16 (~46 GB)${NC}"
download \
    "${HF}/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF/resolve/main/GLM-4.7-Flash-REAP-23B-A3B-BF16.gguf" \
    "${MODELS_DIR}/GLM-4.7-Flash-REAP-23B-A3B-BF16.gguf"
echo ""

# ── Multi-shard: Qwen3-Coder-30B-A3B-Instruct-BF16 (~61 GB, 2 shards) ──────

echo -e "${YELLOW}[7/7] Qwen3-Coder-30B-A3B-Instruct-BF16 (~61 GB)${NC}"
DIR="${MODELS_DIR}/Qwen3-Coder-30B-A3B-Instruct-BF16"
mkdir -p "$DIR"
for i in $(seq -w 1 2); do
    download \
        "${HF}/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-0000${i}-of-00002.gguf" \
        "${DIR}/Qwen3-Coder-30B-A3B-Instruct-BF16-0000${i}-of-00002.gguf"
done
echo ""

echo -e "${GREEN}=== All downloads complete ===${NC}"
echo ""
echo "Models directory:"
ls -1 "$MODELS_DIR"

#!/usr/bin/env bash
#
# Download Gemma 4 GGUF models into ~/.local/share/models/
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
echo -e "${CYAN}=== Gemma 4 Model Downloader ===${NC}"
echo -e "Target: ${MODELS_DIR}"
echo ""

mkdir -p "$MODELS_DIR"

# ── Multi-shard: gemma-4-26B-A4B-it-BF16 (2 shards) ───────────────────────

echo -e "${YELLOW}[1/4] gemma-4-26B-A4B-it-BF16${NC}"
DIR="${MODELS_DIR}/gemma-4-26B-A4B-it-BF16"
mkdir -p "$DIR"
for i in $(seq -w 1 2); do
    download \
        "${HF}/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/BF16/gemma-4-26B-A4B-it-BF16-0000${i}-of-00002.gguf" \
        "${DIR}/gemma-4-26B-A4B-it-BF16-0000${i}-of-00002.gguf"
done
echo ""

# ── Multi-shard: gemma-4-31B-it-BF16 (2 shards) ───────────────────────────

echo -e "${YELLOW}[2/4] gemma-4-31B-it-BF16${NC}"
DIR="${MODELS_DIR}/gemma-4-31B-it-BF16"
mkdir -p "$DIR"
for i in $(seq -w 1 2); do
    download \
        "${HF}/unsloth/gemma-4-31B-it-GGUF/resolve/main/BF16/gemma-4-31B-it-BF16-0000${i}-of-00002.gguf" \
        "${DIR}/gemma-4-31B-it-BF16-0000${i}-of-00002.gguf"
done
echo ""

# ── Single file: gemma-4-E4B-it-BF16 ──────────────────────────────────────

echo -e "${YELLOW}[3/4] gemma-4-E4B-it-BF16${NC}"
download \
    "${HF}/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-BF16.gguf" \
    "${MODELS_DIR}/gemma-4-E4B-it-BF16.gguf"
echo ""

# ── Single file: gemma-4-E2B-it-BF16 ──────────────────────────────────────

echo -e "${YELLOW}[4/4] gemma-4-E2B-it-BF16${NC}"
download \
    "${HF}/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-BF16.gguf" \
    "${MODELS_DIR}/gemma-4-E2B-it-BF16.gguf"
echo ""

echo -e "${GREEN}=== All downloads complete ===${NC}"
echo ""
echo "Models directory:"
ls -1 "$MODELS_DIR"

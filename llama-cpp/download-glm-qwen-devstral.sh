#!/usr/bin/env bash
#
# Download GLM-4.7 / Qwen3 / Devstral GGUF models into ~/.local/share/models/
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

download_shards() {
    local dir="$1"
    local url_base="$2"
    local name="$3"
    local total="$4"

    mkdir -p "$dir"
    local pad=$(printf "%05d" "$total")
    for i in $(seq 1 "$total"); do
        local shard=$(printf "%05d" "$i")
        download \
            "${url_base}/${name}-${shard}-of-${pad}.gguf" \
            "${dir}/${name}-${shard}-of-${pad}.gguf"
    done
}

echo ""
echo -e "${CYAN}=== GLM / Qwen3 / Devstral Model Downloader ===${NC}"
echo -e "Target: ${MODELS_DIR}"
echo ""

mkdir -p "$MODELS_DIR"

# ── 1. GLM-4.7-UD-Q3_K_XL (4 shards) ─────────────────────────────────────

echo -e "${YELLOW}[1/8] GLM-4.7-UD-Q3_K_XL${NC}"
download_shards \
    "${MODELS_DIR}/GLM-4.7-UD-Q3_K_XL" \
    "${HF}/unsloth/GLM-4.7-GGUF/resolve/main/UD-Q3_K_XL" \
    "GLM-4.7-UD-Q3_K_XL" 4
echo ""

# ── 2. GLM-4.7-UD-Q2_K_XL (3 shards) ─────────────────────────────────────

echo -e "${YELLOW}[2/8] GLM-4.7-UD-Q2_K_XL${NC}"
download_shards \
    "${MODELS_DIR}/GLM-4.7-UD-Q2_K_XL" \
    "${HF}/unsloth/GLM-4.7-GGUF/resolve/main/UD-Q2_K_XL" \
    "GLM-4.7-UD-Q2_K_XL" 3
echo ""

# ── 3. GLM-4.7-Flash-BF16 (2 shards) ──────────────────────────────────────

echo -e "${YELLOW}[3/8] GLM-4.7-Flash-BF16${NC}"
download_shards \
    "${MODELS_DIR}/GLM-4.7-Flash-BF16" \
    "${HF}/unsloth/GLM-4.7-Flash-GGUF/resolve/main/BF16" \
    "GLM-4.7-Flash-BF16" 2
echo ""

# ── 4. GLM-4.7-Flash-REAP-23B-A3B-BF16 (single file) ─────────────────────

echo -e "${YELLOW}[4/8] GLM-4.7-Flash-REAP-23B-A3B-BF16${NC}"
download \
    "${HF}/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF/resolve/main/GLM-4.7-Flash-REAP-23B-A3B-BF16.gguf" \
    "${MODELS_DIR}/GLM-4.7-Flash-REAP-23B-A3B-BF16.gguf"
echo ""

# ── 5. Qwen3-Coder-480B-A35B-Instruct-Q2_K (4 shards) ────────────────────

echo -e "${YELLOW}[5/8] Qwen3-Coder-480B-A35B-Instruct-Q2_K${NC}"
download_shards \
    "${MODELS_DIR}/Qwen3-Coder-480B-A35B-Instruct-Q2_K" \
    "${HF}/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/resolve/main/Q2_K" \
    "Qwen3-Coder-480B-A35B-Instruct-Q2_K" 4
echo ""

# ── 6. Qwen3-Coder-480B-A35B-Instruct-UD-IQ1_M (single file) ─────────────

echo -e "${YELLOW}[6/8] Qwen3-Coder-480B-A35B-Instruct-UD-IQ1_M${NC}"
download \
    "${HF}/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/resolve/main/Qwen3-Coder-480B-A35B-Instruct-UD-IQ1_M.gguf" \
    "${MODELS_DIR}/Qwen3-Coder-480B-A35B-Instruct-UD-IQ1_M.gguf"
echo ""

# ── 7. Qwen3-235B-A22B-UD-Q5_K_XL (4 shards) ─────────────────────────────

echo -e "${YELLOW}[7/8] Qwen3-235B-A22B-UD-Q5_K_XL${NC}"
download_shards \
    "${MODELS_DIR}/Qwen3-235B-A22B-UD-Q5_K_XL" \
    "${HF}/unsloth/Qwen3-235B-A22B-GGUF/resolve/main/UD-Q5_K_XL" \
    "Qwen3-235B-A22B-UD-Q5_K_XL" 4
echo ""

# ── 8. Devstral-2-123B-Instruct-2512-Q8_0 (3 shards) ─────────────────────

echo -e "${YELLOW}[8/8] Devstral-2-123B-Instruct-2512-Q8_0${NC}"
download_shards \
    "${MODELS_DIR}/Devstral-2-123B-Instruct-2512-Q8_0" \
    "${HF}/unsloth/Devstral-2-123B-Instruct-2512-GGUF/resolve/main/Q8_0" \
    "Devstral-2-123B-Instruct-2512-Q8_0" 3
echo ""

echo -e "${GREEN}=== All downloads complete ===${NC}"
echo ""
echo "Models directory:"
ls -1 "$MODELS_DIR"

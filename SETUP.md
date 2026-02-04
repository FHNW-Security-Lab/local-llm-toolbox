# Hardware Setup Guide

One-time setup steps for running Local LLM Toolbox with GPU acceleration.

## Our Hardware

Two identical machines:

| Component | Specification                                 |
| --------- | --------------------------------------------- |
| Device    | GMKtec NucBox EVO-X2                          |
| CPU       | AMD Ryzen AI MAX+ 395 (16 cores / 32 threads) |
| GPU       | Integrated Radeon 8060S (shares system RAM)   |
| RAM       | 128GB                                         |
| Storage   | 2TB NVMe (Phison ESR02TBYCCA4)                |
| OS        | Ubuntu 24.04.3 LTS Server                     |
| Kernel    | 6.14.0-37-generic                             |
| Network   | `eno1` (Ethernet), `wlp195s0` (WiFi)          |

---

## General Setup

These steps apply to all backends and configurations.

### Step 1: Install Ubuntu Server

Install Ubuntu 24.04 LTS Server. No desktop environment needed.

### Step 2: SSH Setup
```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl enable ssh

# From your local machine
ssh-copy-id ubuntu@<node-ip>
```

### Step 3: Install Nix
```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

Logout and login again, then verify with `nix --version`.

### Step 4: Install GPU Drivers

Ubuntu 24.04 includes AMD GPU drivers by default:
```bash
lsmod | grep amdgpu
```

If not loaded:
```bash
sudo apt update && sudo apt install linux-firmware && sudo reboot
```

Verify: `ls /dev/dri/` should show `card0` and `renderD128`.

### Step 5: GPU Access Permissions
```bash
sudo usermod -aG render $USER
sudo usermod -aG video $USER
```

**Logout and login again.**

Symptoms if skipped: `llama-server --list-devices` shows nothing, permission denied errors.

### Step 6: AMD GPU Memory Configuration

AMD APUs share system memory. The default GTT size is too small for large models.
```bash
sudo nano /etc/default/grub
```

Set:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432"
```

Apply:
```bash
sudo update-grub && sudo reboot
```

Verify: `cat /sys/class/drm/card0/device/mem_info_gtt_total` should show ~137438953472 (128GB).

### Step 7: Clone and Enter Toolbox
```bash
git clone https://github.com/FHNW-Security-Lab/local-llm-toolbox ~/local-llm-toolbox
cd ~/local-llm-toolbox
nix develop
```

Verify GPU:
```bash
llama-server --list-devices
vulkaninfo --summary 2>&1 | grep -A5 "GPU"
```

---

## Backend: llama.cpp Distributed Inference

Optional setup for splitting models across multiple machines using RPC.

### Network Topology
```
┌─────────────────────┐     Ethernet      ┌─────────────────────┐
│  Node 1 (Main)      │◄─────────────────►│  Node 2 (Worker)    │
│  192.168.100.1      │     (eno1)        │  192.168.100.2      │
│  Runs: toolbox serve│                   │  Runs: toolbox rpc  │
└─────────────────────┘                   └─────────────────────┘
```

### Configure Static IPs

Create `/etc/netplan/99-local-llm.yaml`:

**Node 1 (main):**
```yaml
network:
  version: 2
  ethernets:
    eno1:
      addresses:
        - 192.168.100.1/24
```

**Node 2 (worker):**
```yaml
network:
  version: 2
  ethernets:
    eno1:
      addresses:
        - 192.168.100.2/24
```

Apply: `sudo netplan apply`

### Firewall (if enabled)

On worker node:
```bash
sudo ufw allow from 192.168.100.1 to any port 50052  # RPC server
sudo ufw allow from 192.168.100.1 to any port 50053  # Control API
```

### Running the Cluster

**Worker node (node2):**
```bash
cd ~/local-llm-toolbox && nix develop
./toolbox rpc llama
```

**Main node (node1):**
```bash
cd ~/local-llm-toolbox && nix develop
export LLAMA_RPC_WORKERS=192.168.100.2
./toolbox serve
```

### Manual Test
```bash
llama-server --host 0.0.0.0 --port 8080 \
  --model ~/.local/share/models/your-model.gguf \
  --n-gpu-layers 99 \
  --rpc 192.168.100.2:50052
```

Expected: `using device RPC0 (192.168.100.2:50052) - 127945 MiB free`



## Backend: Microsoft Foundry

_tba_

## Backend: vLLM

*requires CUDA gpu*

## Backend: SGLang

*requires CUDA gpu*

---

## Troubleshooting

| Problem                      | Check                                                        |
| ---------------------------- | ------------------------------------------------------------ |
| GPU not detected             | `groups` should include `render`; `ls -la /dev/dri/`         |
| RPC connection failed        | Worker running? `ss -tlnp \| grep 50052`; Ping works?        |
| Out of memory                | `cat /proc/cmdline`; `cat /sys/class/drm/card0/device/mem_info_gtt_total` |
| Vulkan "INCOMPATIBLE_DRIVER" | Permission denied - check `render` group membership          |

|      |      |
|---------|-------|
|      |      |
|      |      |
|      |      |
|      |      |

# Hardware Setup Guide

One-time setup steps for running local LLMs with GPU acceleration.

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
| Kernel    | 6.17.0-14-generic                              |
| Network   | Thunderbolt 5 (40 Gbps), `eno1` (Ethernet), `wlp195s0` (WiFi) |

---

## General Setup

These steps apply to all backends and configurations.

### Step 1: Install Ubuntu Server

Install Ubuntu 24.04 LTS Server. No desktop environment needed.

### Step 2: Extend LVM Volume

Ubuntu Server's installer creates a small (~100GB) root logical volume by default, even on large disks. Extend it to use all available space:
```bash
sudo vgs                             # Check free space in volume group
sudo lvextend -l +100%FREE /dev/mapper/ubuntu--vg-ubuntu--lv
sudo resize2fs /dev/mapper/ubuntu--vg-ubuntu--lv
df -h /                              # Verify full disk is now available
```

### Step 3: SSH Setup
```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl enable ssh

# From your local machine
ssh-copy-id ubuntu@<node-ip>
```

### Step 4: Install Nix
```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

Logout and login again, then verify with `nix --version`.

### Step 5: Install GPU Drivers

Ubuntu 24.04 includes AMD GPU drivers by default:
```bash
lsmod | grep amdgpu
```

If not loaded:
```bash
sudo apt update && sudo apt install linux-firmware && sudo reboot
```

Verify: `ls /dev/dri/` should show `card0` and `renderD128`.

### Step 6: GPU Access Permissions

Handled automatically by `./dev` on first run (prompts to add your user to `render` and `video` groups).

Symptoms if missing: `llama-server --list-devices` shows nothing, permission denied errors.

### Step 7: AMD GPU Memory Configuration

AMD APUs share system memory. The default GTT size is too small for large models.
Per [AMD's official recommendation](https://community.amd.com/t5/ai/running-a-1-trillion-parameter-llm-locally-with-amd-ryzen-ai-max/ba-p/741744), configure 120GB GTT (leaving ~8GB for OS).

```bash
sudo nano /etc/default/grub
```

Set:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=on iommu=pt amdgpu.gttsize=120000 ttm.pages_limit=30720000"
```

- `amd_iommu=on iommu=pt` — required for Thunderbolt networking and PCI passthrough
- `amdgpu.gttsize=120000` — 120GB GTT (AMD recommended for 128GB systems)
- `ttm.pages_limit=30720000` — matching page limit for TTM

Apply:
```bash
sudo update-grub && sudo reboot
```

Verify: `cat /sys/class/drm/card0/device/mem_info_gtt_total` should show ~125829120000 (~120GB).

### Step 8: Clone and Enter Dev Environment
```bash
git clone https://github.com/FHNW-Security-Lab/local-llm-toolbox ~/local-llm-toolbox
cd ~/local-llm-toolbox/llama-cpp
./dev
```

The `./dev` script detects your GPU and selects the appropriate Nix dev shell:
- **ROCm** — recommended for AMD GPUs (enables Flash Attention via rocWMMA)
- **Vulkan** — fallback for AMD GPUs
- **CUDA** — for NVIDIA GPUs

On first run, `./dev` will also check system configuration and prompt you to fix:
- **GPU group membership** — adds your user to `render` and `video` groups
- **GPU driver visibility** — runs [system-manager](https://github.com/numtide/system-manager) to create the `/run/opengl-driver` symlink (via [nix-system-graphics](https://github.com/soupglasses/nix-system-graphics))

Follow the prompts, then verify GPU:
```bash
llama-server --list-devices
```

---

## Distributed Inference (llama.cpp RPC)

Optional setup for splitting models across multiple machines using llama.cpp's RPC.
Based on [AMD's guide for distributed inference with Ryzen AI MAX](https://community.amd.com/t5/ai/running-a-1-trillion-parameter-llm-locally-with-amd-ryzen-ai-max/ba-p/741744).

### Network Topology
```
┌─────────────────────┐  Thunderbolt 5    ┌─────────────────────┐
│  Node 1 (Main)      │◄───(40 Gbps)────►│  Node 2 (Worker)    │
│  192.168.200.1      │  (thunderbolt0)   │  192.168.200.2      │
│  Runs: toolbox serve│                   │  Runs: toolbox rpc  │
└─────────────────────┘                   └─────────────────────┘
         │                                          │
         │  Ethernet (backup)                       │
         │  192.168.100.1 (eno1)   192.168.100.2 (eno1)
```

### Configure Static IPs

Create `/etc/netplan/99-local-llm.yaml`:

**Node 1 (main):**
```yaml
network:
  version: 2
  ethernets:
    thunderbolt0:
      addresses:
        - 192.168.200.1/24
    eno1:
      addresses:
        - 192.168.100.1/24
```

**Node 2 (worker):**
```yaml
network:
  version: 2
  ethernets:
    thunderbolt0:
      addresses:
        - 192.168.200.2/24
    eno1:
      addresses:
        - 192.168.100.2/24
```

Apply: `sudo netplan apply`

Verify: `ping 192.168.200.2` from node 1. For bandwidth test: `iperf3` (expect ~13.6 Gbps over Thunderbolt 5).

> **Note:** Thunderbolt requires `amd_iommu=on iommu=pt` in GRUB (see Step 7).

### Firewall (if enabled)

On worker node:
```bash
sudo ufw allow from 192.168.200.1 to any port 50053  # RPC data port
sudo ufw allow from 192.168.200.1 to any port 50054  # Control API
```

### Running the Cluster

**Worker node (node2):**
```bash
cd ~/local-llm-toolbox/llama-cpp && ./dev
llama-rpc-server --host 0.0.0.0 --port 50053
```

**Main node (node1):**
```bash
cd ~/local-llm-toolbox/llama-cpp && ./dev
llama-server --host 0.0.0.0 --port 8080 \
  --models-dir ~/models \
  -ngl 99 --no-mmap -fa \
  --rpc 192.168.200.2:50053
```

Expected output: `using device RPC0 (192.168.200.2:50053) - ~120000 MiB free`

---

## Troubleshooting

| Problem                      | Check                                                        |
| ---------------------------- | ------------------------------------------------------------ |
| GPU not detected             | `groups` should include `render`; `ls -la /dev/dri/`         |
| Nix programs can't find GPU  | Run `ls -la /run/opengl-driver`; if missing, run system-manager (Step 9) |
| RPC connection failed        | Worker running? `ss -tlnp \| grep 50053`; Ping works?        |
| Out of memory                | `cat /proc/cmdline`; `cat /sys/class/drm/card0/device/mem_info_gtt_total` |
| Disk full / LLVM error       | `df -h /`; extend LVM if needed (see Step 2)                |
| Vulkan "INCOMPATIBLE_DRIVER" | Permission denied - check `render` group membership          |

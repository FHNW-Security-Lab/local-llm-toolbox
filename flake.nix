{
  description = "Local LLM Toolbox - unified interface for LLM backends";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixgl = {
      url = "github:nix-community/nixGL";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, nixgl }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

    in {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          nixglPkgs = nixgl.packages.${system};
          isLinux = pkgs.stdenv.isLinux;

          llamaVulkan = pkgs.llama-cpp.override { vulkanSupport = true; rpcSupport = true; };
          llamaCuda = pkgs.llama-cpp.override { cudaSupport = true; rpcSupport = true; };
          llamaMetal = pkgs.llama-cpp.override { rpcSupport = true; };  # Metal is default on Darwin
          llamaCpu = pkgs.llama-cpp.override { blasSupport = true; rpcSupport = true; };

          # Base packages for all shells
          basePkgs = [ pkgs.python312 pkgs.uv ];

          # Shell hook with optional Vulkan setup for non-NixOS Linux
          mkShellHook = accel: vulkanSetup: ''
            # Create/activate venv if not exists
            if [[ ! -d .venv ]]; then
              uv venv
            fi
            source .venv/bin/activate

            # Sync deps
            if [[ -f pyproject.toml ]]; then
              uv sync
            fi

            export PYTHONPATH="$PWD:$PYTHONPATH"

            ${vulkanSetup}

            echo ""
            echo "=== Local LLM Toolbox ==="
            echo "GPU: ${accel}"
            echo ""
            ./toolbox help
          '';

          # Vulkan environment setup for non-NixOS Linux (sources nixGL's env vars)
          vulkanEnvSetup = if isLinux then ''
            # Set up Vulkan environment for non-NixOS systems
            # This replicates what nixVulkanIntel does
            if [[ -f /etc/os-release ]] && ! grep -q "ID=nixos" /etc/os-release 2>/dev/null; then
              export VK_LAYER_PATH="${nixglPkgs.nixVulkanIntel}/share/vulkan/explicit_layer.d''${VK_LAYER_PATH:+:$VK_LAYER_PATH}"
              NIXGL_MESA_ICD=$(cat ${nixglPkgs.nixVulkanIntel}/share/vulkan/nixgl_mesa_icd 2>/dev/null || echo "")
              if [[ -n "$NIXGL_MESA_ICD" ]]; then
                export VK_ICD_FILENAMES="$NIXGL_MESA_ICD''${VK_ICD_FILENAMES:+:$VK_ICD_FILENAMES}"
              fi
              export LD_LIBRARY_PATH="${nixglPkgs.nixVulkanIntel}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
          '' else "";

        in {
          # Default: Vulkan on Linux, Metal on macOS
          default = pkgs.mkShell {
            packages = (if isLinux then [ llamaVulkan nixglPkgs.nixVulkanIntel ] else [ llamaMetal ]) ++ basePkgs;
            shellHook = mkShellHook (if isLinux then "Vulkan (AMD/Intel)" else "Metal") vulkanEnvSetup;
          };

          # Named shells for explicit selection
          nvidia = pkgs.mkShell {
            packages = (if isLinux then [ llamaCuda ] else [ llamaMetal ]) ++ basePkgs;
            shellHook = mkShellHook (if isLinux then "CUDA (NVIDIA)" else "Metal") "";
          };

          vulkan = pkgs.mkShell {
            packages = (if isLinux then [ llamaVulkan nixglPkgs.nixVulkanIntel ] else [ llamaMetal ]) ++ basePkgs;
            shellHook = mkShellHook (if isLinux then "Vulkan (AMD/Intel)" else "Metal") vulkanEnvSetup;
          };

          cpu = pkgs.mkShell {
            packages = [ llamaCpu ] ++ basePkgs;
            shellHook = mkShellHook "CPU (BLAS)" "";
          };
        }
      );
    };
}


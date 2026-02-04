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

      # Shared shell hook
      shellHook = accel: ''
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

        echo ""
        echo "=== Local LLM Toolbox ==="
        echo "GPU: ${accel}"
        echo ""
        ./toolbox help
      '';

      # Create dev shell with specified llama-cpp config
      mkShell = pkgs: packages: accel: pkgs.mkShell {
        inherit packages;
        shellHook = shellHook accel;
      };

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

          # NixGL wrapper for Vulkan on Linux (needed for non-NixOS systems)
          vulkanPkgs = if isLinux
            then [ llamaVulkan nixglPkgs.nixVulkanIntel ] ++ basePkgs
            else [ llamaMetal ] ++ basePkgs;

        in {
          # Default: Vulkan on Linux (with nixGL), Metal on macOS
          default = mkShell pkgs vulkanPkgs
            (if isLinux then "Vulkan (AMD/Intel) - use 'nixVulkanIntel llama-server' on non-NixOS" else "Metal");

          # Named shells for explicit selection
          nvidia = mkShell pkgs
            (if isLinux then [ llamaCuda ] ++ basePkgs else [ llamaMetal ] ++ basePkgs)
            (if isLinux then "CUDA (NVIDIA)" else "Metal");

          vulkan = mkShell pkgs vulkanPkgs
            (if isLinux then "Vulkan - use 'nixVulkanIntel llama-server' on non-NixOS" else "Metal");

          cpu = mkShell pkgs ([ llamaCpu ] ++ basePkgs) "CPU (BLAS)";
        }
      );
    };
}

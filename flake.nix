{
  description = "Local LLM Toolbox - unified interface for LLM backends";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
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
      mkShell = pkgs: llama: accel: pkgs.mkShell {
        packages = [ llama pkgs.python312 pkgs.uv ];
        shellHook = shellHook accel;
      };

    in {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          isLinux = pkgs.stdenv.isLinux;

          llamaVulkan = pkgs.llama-cpp.override { vulkanSupport = true; };
          llamaCuda = pkgs.llama-cpp.override { cudaSupport = true; };
          llamaMetal = pkgs.llama-cpp;  # Metal is default on Darwin
          llamaCpu = pkgs.llama-cpp.override { blasSupport = true; };
        in {
          # Default: Vulkan on Linux, Metal on macOS
          default = mkShell pkgs
            (if isLinux then llamaVulkan else llamaMetal)
            (if isLinux then "Vulkan (AMD/Intel)" else "Metal");

          # Named shells for explicit selection
          nvidia = mkShell pkgs
            (if isLinux then llamaCuda else llamaMetal)
            (if isLinux then "CUDA (NVIDIA)" else "Metal");

          vulkan = mkShell pkgs
            (if isLinux then llamaVulkan else llamaMetal)
            (if isLinux then "Vulkan" else "Metal");

          cpu = mkShell pkgs llamaCpu "CPU (BLAS)";
        }
      );
    };
}

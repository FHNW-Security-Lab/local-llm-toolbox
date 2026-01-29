{
  description = "Local LLM Toolbox - unified interface for LLM backends";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          isLinux = pkgs.stdenv.isLinux;

          # On Linux use Vulkan, on macOS use Metal (default)
          llama = if isLinux then
            pkgs.llama-cpp.override { vulkanSupport = true; }
          else
            pkgs.llama-cpp;
        in {
          default = pkgs.mkShell {
            packages = [
              llama
              pkgs.python312
              pkgs.uv
            ];

            shellHook = ''
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
              echo ""
              echo "Backend:"
              echo "  ./toolbox start <backend>   Start a backend"
              echo "  ./toolbox stop              Stop active backend"
              echo "  ./toolbox status            Show status"
              echo ""
              echo "Models:"
              echo "  ./toolbox models            List available models"
              echo "  ./toolbox load <model>      Load a model"
              echo "  ./toolbox unload            Unload current model"
              echo ""
              echo "Services:"
              echo "  ./toolbox router            Start the API router"
              echo "  ./toolbox dashboard         Start the web dashboard"
              echo ""
              echo "Backends: llama (GGUF), foundry (ONNX), vllm, sglang"
              echo ""
            '';
          };
        }
      );
    };
}

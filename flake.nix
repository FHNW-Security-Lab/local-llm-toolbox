{
  description = "Local LLM Toolbox - unified interface for LLM backends";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    system-manager = {
      url = "github:numtide/system-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nix-system-graphics = {
      url = "github:soupglasses/nix-system-graphics";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, system-manager, nix-system-graphics }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

    in {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          isLinux = pkgs.stdenv.isLinux;

          llamaVulkan = pkgs.llama-cpp.override { vulkanSupport = true; rpcSupport = true; };
          llamaCuda = pkgs.llama-cpp.override { cudaSupport = true; rpcSupport = true; };
          llamaMetal = pkgs.llama-cpp.override { rpcSupport = true; };  # Metal is default on Darwin
          llamaCpu = pkgs.llama-cpp.override { blasSupport = true; rpcSupport = true; };

          # Base packages for all shells
          basePkgs = [ pkgs.python312 pkgs.uv ];

          mkShellHook = accel: ''
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

        in {
          # Default: Vulkan on Linux, Metal on macOS
          default = pkgs.mkShell {
            packages = (if isLinux then [ llamaVulkan ] else [ llamaMetal ]) ++ basePkgs;
            shellHook = mkShellHook (if isLinux then "Vulkan (AMD/Intel)" else "Metal");
          };

          # Named shells for explicit selection
          nvidia = pkgs.mkShell {
            packages = (if isLinux then [ llamaCuda ] else [ llamaMetal ]) ++ basePkgs;
            shellHook = mkShellHook (if isLinux then "CUDA (NVIDIA)" else "Metal");
          };

          vulkan = pkgs.mkShell {
            packages = (if isLinux then [ llamaVulkan ] else [ llamaMetal ]) ++ basePkgs;
            shellHook = mkShellHook (if isLinux then "Vulkan (AMD/Intel)" else "Metal");
          };

          cpu = pkgs.mkShell {
            packages = [ llamaCpu ] ++ basePkgs;
            shellHook = mkShellHook "CPU (BLAS)";
          };
        }
      );

      # System-level configuration for non-NixOS Linux machines.
      # Creates /run/opengl-driver symlink so Nix-built programs find GPU drivers.
      #
      # One-time activation:
      #   sudo nix run 'github:numtide/system-manager' -- switch --flake '.'
      systemConfigs = let
        linuxSystems = [ "x86_64-linux" "aarch64-linux" ];
      in nixpkgs.lib.genAttrs linuxSystems (system: {
        default = system-manager.lib.makeSystemConfig {
          modules = [
            nix-system-graphics.systemModules.default
            ({ ... }: {
              config = {
                nixpkgs.hostPlatform = system;
                system-manager.allowAnyDistro = true;
                system-graphics.enable = true;
              };
            })
          ];
        };
      });
    };
}

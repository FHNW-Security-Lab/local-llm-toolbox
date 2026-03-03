{
  description = "llama-router - llama.cpp with auto GPU setup";

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
          llamaRocm = (pkgs.llama-cpp.override {
            rocmSupport = true;
            rpcSupport = true;
            rocmGpuTargets = [ "gfx1151" ];
          }).overrideAttrs (prev: {
            cmakeFlags = prev.cmakeFlags ++ [
              "-DGGML_HIP_ROCWMMA_FATTN=ON"
              "-Drocwmma_ROOT=${pkgs.rocmPackages.rocwmma}"
              "-DCMAKE_HIP_FLAGS=-I${pkgs.rocmPackages.rocwmma}/include"
            ];
            buildInputs = prev.buildInputs ++ [
              pkgs.rocmPackages.rocwmma
            ];
          });
          llamaCuda = pkgs.llama-cpp.override { cudaSupport = true; rpcSupport = true; };
          llamaMetal = pkgs.llama-cpp.override { rpcSupport = true; };
          llamaCpu = pkgs.llama-cpp.override { blasSupport = true; rpcSupport = true; };

          mkShellHook = accel: ''
            echo ""
            echo "=== llama-router ==="
            echo "GPU: ${accel}"
            echo ""
            echo "Start server (router mode with web UI):"
            echo "  llama-server --models-dir ~/models -ngl 99 -c 8192"
            echo ""
            echo "Options:"
            echo "  --models-dir PATH    Directory with GGUF files (scanned on startup)"
            echo "  --models-max N       Max models loaded at once (default: 4, LRU eviction)"
            echo "  -ngl N               GPU layers (99 = all)"
            echo "  -c N                 Context size (default: 2048)"
            echo "  --port N             Port (default: 8080)"
            echo ""
            echo "Start RPC worker (distributed inference):"
            echo "  llama-rpc-server --host 0.0.0.0 --port 50053"
            echo ""
            echo "Connect to RPC workers from the server:"
            echo "  llama-server --models-dir ~/models -ngl 99 --rpc host1:50053,host2:50053"
            echo ""
            echo "Web UI:  http://localhost:8080"
            echo "Models:  curl http://localhost:8080/models"
            echo ""
          '';

        in {
          default = pkgs.mkShell {
            packages = if isLinux then [ llamaVulkan ] else [ llamaMetal ];
            shellHook = mkShellHook (if isLinux then "Vulkan (AMD/Intel)" else "Metal");
          };

          nvidia = pkgs.mkShell {
            packages = if isLinux then [ llamaCuda ] else [ llamaMetal ];
            shellHook = mkShellHook (if isLinux then "CUDA (NVIDIA)" else "Metal");
          };

          vulkan = pkgs.mkShell {
            packages = if isLinux then [ llamaVulkan ] else [ llamaMetal ];
            shellHook = mkShellHook (if isLinux then "Vulkan (AMD/Intel)" else "Metal");
          };

          rocm = pkgs.mkShell {
            packages = if isLinux then [ llamaRocm ] else [ llamaMetal ];
            shellHook = mkShellHook (if isLinux then "ROCm (AMD)" else "Metal");
          };

          cpu = pkgs.mkShell {
            packages = [ llamaCpu ];
            shellHook = mkShellHook "CPU (BLAS)";
          };
        }
      );

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

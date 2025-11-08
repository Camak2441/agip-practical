{
  description = "";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    additionalPythonPackages = {
      url = "github:Camak2441/additional-python-packages";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self, nixpkgs, additionalPythonPackages, ...
  } @ inputs:
  let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    forAllSystems = nixpkgs.lib.genAttrs systems;
  in
  {
    devShells = forAllSystems (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        addPyPkgs = additionalPythonPackages.packages.${system};
      in
      {
        default = pkgs.mkShell rec {
          packages = with pkgs; [
            suitesparse
            (python3.withPackages (ps: [
              ps.numpy
              ps.scipy
              ps.matplotlib
              ps.scikit-image
              ps.cython
              addPyPkgs.scikit-sparse
            ]))
          ];
        };
      }
    );
  };
}
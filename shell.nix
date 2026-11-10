{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    (python313.withPackages (
      ps: with ps; [
        torch
        numpy
        pandas
        accelerate
        ipython
      ]
    ))
  ];
}

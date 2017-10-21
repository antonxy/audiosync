let
    pkgs = import <nixpkgs> {};
    stdenv = pkgs.stdenv;
    pypkgs = pkgs.python36Packages;
in rec {
    audiosync = stdenv.mkDerivation rec {
        name = "audiosync";
        src = ./.;
        nativeBuildInputs = [ pkgs.pkgconfig ];
        buildInputs = [
          pypkgs.numpy
          pypkgs.tkinter
          pkgs.ffmpeg

#Debug
          pypkgs.matplotlib
        ];
    };
}

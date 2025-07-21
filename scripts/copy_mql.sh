#!/usr/bin/bash
# Arg1 is root source MQL5 folder, this script needs to be run from the build folder or the scripts folder beneath the repository root
for ff in `find "$1" -type d -name tempus -printf "%P\n"`; do cp -Rvu "$1/$ff/"* ../mql/MQL5/$ff/; done


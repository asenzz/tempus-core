#!/usr/bin/bash
# Arg1 is root source MQL5 folder
for ff in `find "$1" -type d -name tempus -printf "%P\n"`; do cp -Rvu "$1/$ff/"* ../mql/MQL5/$ff/; done


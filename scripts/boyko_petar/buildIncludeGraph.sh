#!/usr/bin/env bash

INC_DIRS=""

for id in ../SVRRoot/*/include
do
    INC_DIRS=$INC_DIRS,$id
done

cat << EOF
Building inclusion graph for the project.
            Output: ../build/includeGraph.dot
Source directories: ../SVRRoot
      Include dirs: $INC_DIRS
EOF

./cinclude2dot.pl --src ../SVRRoot --include $INC_DIRS > ../build/includeGraph.dot


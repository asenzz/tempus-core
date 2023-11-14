#!/bin/bash

INPUT=""
OUTPUT=""
INCLUDE_DIRS=""

while getopts ":i:o:I:" opt; do
  case $opt in
    i)
      INPUT=$OPTARG
      ;;
    o)
      OUTPUT=$OPTARG
      ;;
    I)
      INCLUDE_DIRS=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# copy the configuration file to the build folder
cp $INPUT/config.json $OUTPUT

# write all templates here
TEMPLATES="$INPUT/MainLayout.tmpl"

# process templates to срр
cppcms_tmpl_cc $TEMPLATES *.view -o $INPUT/SVRWebSkin.cpp

# collect templates to the library
g++ -shared -fPIC $INPUT/SVRWebSkin.cpp -I../include ${INCLUDE_DIRS} -o $OUTPUT/libSVRWebSkin.so -lcppcms -lbooster
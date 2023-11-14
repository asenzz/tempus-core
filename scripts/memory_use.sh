#!/bin/bash
echo Extracting process memory use in megabytes from log file at $1 to clipboard . . .
sed -n 's/\[\(.*\)\] \[0x0.* process memory RSS \(.*\) MB/\1\t\2/gp' $1 | xclip -selection clipboard
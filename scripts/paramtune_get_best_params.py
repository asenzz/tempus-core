SCORE_FIELD = 18

# Usage: paramtune_get_best_params.py paramtune_log_file number_of_levels

import sys
import os

input_file = sys.argv[1];
temp_grep = "/tmp/filtered.out"
print("Input file is " + input_file)
os.system('grep Loss ' + input_file + ' > ' + temp_grep)
levels = int(sys.argv[2])

for level in range(levels):
    temp_grep_level = temp_grep + "_" + str(level)
    os.system("grep 'level " + str(level) + "' " + temp_grep + " > " + temp_grep_level)
    file = open(temp_grep_level)
    lines = file.readlines()
    if len(lines) < 1: continue
    lines.sort(key=lambda line: float(line.split()[SCORE_FIELD]))
    print("Level " + str(level) + " best score:\n" + lines[0] + "\n")
    file.close()
    os.system("rm -f " + temp_grep_level)

os.system("rm -f " + temp_grep)
print("Done.")
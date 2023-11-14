import csv
import sys
if len(sys.argv) > 1:
    QUIET = 1
    MAX_LINES = int(sys.argv[1])
else:
    QUIET = 0
    MAX_LINES = 1000000

highs = []
lows = []
line_ct = 0
with open('/home/zarko/repo/tempus-db/dbscripts/0.98_eurusd_avg_3600_new_ohlc_data.sql') as csvfile:
    valreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
    for row in valreader:
        if len(row) < 6: continue
        highs.append(float(row[4]))
        lows.append(float(row[5]))
        line_ct += 1
        if line_ct >= MAX_LINES + 1: break

print("len(highs) " + str(len(highs)) + " len(lows) " + str(len(lows)))
avg_range = 0.
for i in range(1, len(highs)):
    avg_range += max(highs[i] - lows[i], 0)
avg_range /= len(highs) - 1

print("Average range " + str(avg_range) + " of " + str(len(highs) - 1) + " rows.")
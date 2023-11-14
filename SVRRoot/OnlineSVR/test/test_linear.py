import csv
import sys
from collections import OrderedDict

import dateutil.parser as parser
from dateutil import tz

ENABLE_CONSEQUENT_HACK=False
SECOND_PREDICTION=True

if len(sys.argv) > 1:
    QUIET = 1
    MAX_LINES = int(sys.argv[1])
else:
    QUIET = 0
    MAX_LINES = 10000

FROM_ZONE=tz.gettz("Europe/Zurich")
TO_ZONE=tz.gettz("UTC")
ADD_DST=False

# os.system("sed -n 's/.*compare_by_value_mean_error_ohlc.*Time \([^,]\+\), position .* predicted \([^,]\+\).*/\1\t\2/p' /var/log/svrbacktest.log > /var/tmp/predictions.tsv")




if SECOND_PREDICTION: ENABLE_CONSEQUENT_HACK = False

predictions = OrderedDict()
line_ct = 0
with open('/var/tmp/predictions_second.tsv') as csvfile:
    valreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
    for row in valreader:
        if len(row) < 2: continue
        row_time = parser.parse(row[0])
        predictions[row_time] = float(row[1])
        line_ct += 1
        if line_ct >= MAX_LINES + 1: break

avgs = OrderedDict()
line_ct = 0
with open('/home/zarko/repo/tempus-db/dbscripts/0.98_eurusd_avg_3600_new_avg_data.sql') as csvfile:
    valreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
    for row in valreader:
        if len(row) < 4: continue
        row_time = parser.parse(row[0])
        avgs[row_time] = float(row[3])
        line_ct += 1
        if line_ct >= MAX_LINES + 1: break

highs = OrderedDict()
lows = OrderedDict()
closes = OrderedDict()
with open('/home/zarko/repo/tempus-db/dbscripts/0.98_eurusd_avg_3600_new_ohlc_data.sql') as csvfile:
    valreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
    for row in valreader:
        if len(row) < 6: continue
        row_time = parser.parse(row[0])
        highs[row_time] = float(row[4])
        lows[row_time] = float(row[5])
        closes[row_time] = float(row[6])

print("len(highs) " + str(len(highs)) + " len(lows) " + str(len(lows)) + " len(avgs) " + str(len(avgs)) + " len(predictions) " + str(len(predictions)))

hit_count = 0
total_count = 0
mae = 0
avg_times = list(avgs.keys())
for i in range(2, len(avg_times)):
    if SECOND_PREDICTION:
        avg_pred_time = avg_times[i - 2]
    else:
        avg_pred_time = avg_times[i - 1]
    pred_avg = avgs[avg_pred_time]
    pred_close = closes[avg_pred_time]
    try:
        if lows[avg_times[i]] < pred_close < highs[avg_times[i]]:
            hit_count += 1
        total_count += 1
        mae += abs(avgs[avg_times[i]] - pred_close)
    except:
        print("Time " + str(avg_times[i]) + " not found in OHLC data.")

mae = mae / total_count
hit_rate = 100. * hit_count / total_count
if QUIET:
    print(str(hit_rate))
else:
    print("Hit rate of last close is " + str(hit_rate) + " and MAE is " + str(mae) + " out of " + str(
        total_count) + " comparisons.")

hit_count = 0
hit_count_close = 0
total_count = 0
mae = 0
mae_close = 0
pred_times = list(predictions.keys())
maes = []
print("Pos\tOpen UTC\tAbsolute Error\tMAE\tMA10E\tHit-rate %\tTWAP1H\tPredicted TWAP1H\tHigh pc\tLow pc\tClose pc\tAE Close\tMAE Close\tHit-rate close %")
for i in range(2, len(pred_times)):
    #print ("Pred time " + str(pred_times[i - 1]))
    # Hack
    pred = predictions[pred_times[i]]
    fin_pred = pred
    if SECOND_PREDICTION:
        close_time = pred_times[i - 2]
    else:
        close_time = pred_times[i - 1]
    close_pred = closes[close_time]
    try:
        if lows[pred_times[i]] < fin_pred < highs[pred_times[i]]:
            hit_count += 1
        if lows[pred_times[i]] < close_pred < highs[pred_times[i]]:
            hit_count_close += 1
        total_count += 1
    except:
        print("Time " + str(pred_times[i]) + " not found in OHLC data.")
        continue

    # Row stats
    ae = abs(avgs[pred_times[i]] - fin_pred)
    ae_close = abs(avgs[pred_times[i]] - close_pred)
    mae += ae
    mae_close += ae_close
    maes.append(ae)
    if len(maes) > 10:
        mae10 = sum(maes[-10:]) / 10.
    else:
        mae10 = 0
    mae_now = mae / total_count
    mae_close_now = mae_close / total_count
    hit_rate_now = 100. * hit_count / total_count
    hit_rate_close_now = 100. * hit_count_close / total_count

    # Convert time zone
    conv_row_datetime = pred_times[i]
    conv_row_datetime = conv_row_datetime.replace(tzinfo=FROM_ZONE)
    conv_row_datetime.astimezone(TO_ZONE)
    if (ADD_DST): conv_row_datetime = conv_row_datetime + conv_row_datetime.dst() # Add DST if not present uncomment this line

    # Print row
    print(f'{i}\t{conv_row_datetime}\t{ae}\t{mae_now}\t{mae10}\t{hit_rate_now}\t{avgs[pred_times[i]]}\t{fin_pred}\t{highs[pred_times[i]]}\t{lows[pred_times[i]]}\t{closes[pred_times[i]]}\t{ae_close}\t{mae_close_now}\t{hit_rate_close_now}')

mae = mae / total_count
hit_rate = 100. * hit_count / total_count
if QUIET:
    print(str(hit_rate))
else:
    print("Hit rate of predictions is " + str(hit_rate) + " and MAE is " + str(mae) + " out of " + str(
        total_count) + " comparisons.")

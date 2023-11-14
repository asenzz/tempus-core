import psycopg2
first_n_predictions = -8 * 115
last_n_predictions = -4 * 115
conn = psycopg2.connect(database="svrwave",
                        host="/var/run/postgresql",
                        user="svrwave",
                        password="svrwave")

cur = conn.cursor()
cur.execute("SELECT multival_results.value_time, SIGN(multival_results.value - q_svrwave_xauusd_avg_1.xauusd_avg_bid) AS forecast_direction FROM multival_results JOIN q_svrwave_xauusd_avg_1 ON multival_results.value_time = q_svrwave_xauusd_avg_1.value_time + interval '361 seconds' WHERE q_svrwave_xauusd_avg_1.value_time IN (SELECT value_time - interval '361 seconds' FROM multival_results ORDER BY value_time ASC);")
# cur.execute("SELECT multival_results.value_time, SIGN(multival_results.value - q_svrwave_xauusd_avg_1.xauusd_avg_bid) AS forecast_direction FROM multival_results JOIN q_svrwave_xauusd_avg_1 ON multival_results.value_time = q_svrwave_xauusd_avg_1.value_time + interval '361 seconds' WHERE q_svrwave_xauusd_avg_1.value_time IN (SELECT value_time - interval '361 seconds' FROM multival_results WHERE value_time >= '2023.04.18 00:00:00' AND value_time <= '2023.5.15 23:59:59' ORDER BY value_time ASC);")
res = cur.fetchall()
forecasted_direction = dict()
for r in res:
    forecasted_direction[r[0]] = r[1]
    # print(forecasted_direction)

actual_direction = dict()
# cur.execute("SELECT q_svrwave_xauusd_avg_3600.value_time, SIGN(q_svrwave_xauusd_avg_3600.xauusd_avg_bid - q_svrwave_xauusd_avg_1.xauusd_avg_bid) AS actual_direction FROM q_svrwave_xauusd_avg_3600 JOIN q_svrwave_xauusd_avg_1 ON q_svrwave_xauusd_avg_3600.value_time = q_svrwave_xauusd_avg_1.value_time + interval '361 seconds' WHERE q_svrwave_xauusd_avg_1.value_time IN (SELECT value_time - interval '361 seconds' FROM q_svrwave_xauusd_avg_3600 ORDER BY value_time ASC);")
cur.execute("SELECT q_svrwave_xauusd_avg_3600.value_time, SIGN(q_svrwave_xauusd_avg_3600.xauusd_avg_bid - q_svrwave_xauusd_avg_1.xauusd_avg_bid) AS actual_direction FROM q_svrwave_xauusd_avg_3600 JOIN q_svrwave_xauusd_avg_1 ON q_svrwave_xauusd_avg_3600.value_time = q_svrwave_xauusd_avg_1.value_time + interval '361 seconds' WHERE q_svrwave_xauusd_avg_1.value_time IN (SELECT value_time - interval '361 seconds' FROM q_svrwave_xauusd_avg_3600 WHERE value_time > '2020.01.01 00:00:00' ORDER BY value_time ASC);")
res = cur.fetchall()
for r in res:
    actual_direction[r[0]] = r[1]
    # print(actual_direction)

cur.close()
conn.close()

actual_times = sorted(actual_direction.keys())
forecasted_times = sorted(forecasted_direction.keys())[first_n_predictions:last_n_predictions]
best_corrpct = 0
best_skip_hour = 0
best_ct = 0
best_errpct = 0
best_corrpct = 0
best_skip_weekday = 0
for skip_weekday in range(0, 6):
    best_corrpct = 0
    for skip_hour in range(0, 24):
        err = 0
        ct = 0
        rmct = 0
        for valtime in forecasted_times:
            if valtime.weekday() >= skip_weekday and valtime.hour >= skip_hour: continue
            if valtime in actual_direction and valtime in forecasted_direction:
                ct += 1
                if actual_direction[valtime] != forecasted_direction[valtime]:
                    rmct += 1
                    err += 1
                    #if rmct < 10:
                    #    print("UPDATE multival_results SET value = value + " + str(10 * actual_direction[valtime]) + " WHERE value_time = '" + str(valtime) + "';")
        if ct < 1:
            print("No result, aborting, for " + str(skip_weekday) + "wd:" + str(skip_hour) + "h")
            # exit(1)
        else:
            errpct = 100. * err / ct
            corrpct = 100. - errpct
            if corrpct > best_corrpct:
                best_corrpct = corrpct
                best_errpct = errpct
                best_ct = ct
                best_skip_hour = skip_hour
                best_skip_weekday = skip_weekday

    print("Best corrpct for " + str(skip_weekday) + "wd: " + str(best_skip_hour) + "h, total " + str(best_ct) + ", trading alpha " + str(best_corrpct) + "%, ratio " + str(best_corrpct/best_errpct) + ", starting " + str(next(iter(forecasted_times))) + ", until " + str(list(forecasted_times)[-1]))

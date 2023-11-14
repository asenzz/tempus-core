#!/usr/bin/python3

import os

first = True # Set to True to SKIP ERASING FIR coefs on first cycle
c_week_len = 115
offset = 115
offlimit = 8 * 115
os.environ["PGPASSWORD"] = "svrwave"
os.environ["PGUSER"] = "svrwave"
os.environ["PGDATABASE"] = "svrwave"
os.environ["PGHOST"] = "/var/run/postgresql"
os.system("psql -c 'delete from multival_results'")
os.chdir("/mnt/faststore/repo/tempus-core/build")
while offset < offlimit:
    os.system("pkill SVRDaemon")
    if not first: 
        res = os.system("rm -f /mnt/faststore/repo/tempus-core/libs/oemd_fir_masks_xauusd_1s/*")
        if res: print("Got error result ", str(res))
    else:
        first = False
    res = os.system("psql -c 'delete from svr_parameters; delete from dq_scaling_factors'")
    if res: print("Got error result ", str(res))
    os.environ["SVRWAVE_TEST_WINDOW"] = str(offset)
    res = os.system("/bin/bash ../scripts/run_daemon.sh >> /mnt/slowstore/var/log/svrdaemon_generate_forecasts.log 2>&1")
    if res: print("Got error result ", str(res))
    offset += c_week_len

#/bin/bash
cmake -N -LA . | tail -n+2 | sed -r 's/([A-Za-z_0-9]+):([A-Z]+)=(.*)/set(\1 "\3" CACHE \2 "")/' > cmake-init.txt

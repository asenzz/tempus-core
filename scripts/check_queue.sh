#!/bin/bash

# Example usage ../scripts/check_queue.sh q_svrwave_eurusd_avg_1 eurusd_avg_bid 100

export PGHOST=/var/run/postgresql
export DB=svrwave
export PGPASSWORD=$DB

echo Rows prior to which is a gap:
psql -U $DB -d $DB -h $PGHOST -c "select value_time from (select value_time, $2, lag(value_time) over client_window as prev_value_time from $1 window client_window as (order by value_time asc)) as lagged_time where (select max_gap from datasets where datasets.id = $3) < lagged_time.value_time - lagged_time.prev_value_time or lagged_time.$2 = double precision 'NaN'"

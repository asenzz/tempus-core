#!/bin/bash
grep 'Actual values at' $1 | sed -r 's/.*Actual values at (.*): (.*)/\1, \2/g' > /tmp/actual_values.csv
grep 'Predicted values at ' $1 | sed -r 's/.*Predicted values at (.*): (.*)/\1, \2/g' > /tmp/predicted_values.csv


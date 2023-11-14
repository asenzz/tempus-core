#!/bin/bash
grep ', position 0, error ' $1 | perl -n -e'/Time ([^,]+), position 0, error ([^,]+), predicted ([^,]+), etalon high ([^,]+), etalon low([^,]+)/ && print "$1;$2;$3;$4;$5\n"' > position_0_error_high_low.csv
#grep ', position 1, error ' $1 | perl -n -e'/Time ([^,]+), position 1, error ([^,]+), predicted ([^,]+), etalon high ([^,]+), etalon low([^,]+)/ && print "$1;$2;$3;$4;$5\n"' > position_1_error_high_low.csv
#grep ', position 2, error ' $1 | perl -n -e'/Time ([^,]+), position 2, error ([^,]+), predicted ([^,]+), etalon high ([^,]+), etalon low([^,]+)/ && print "$1;$2;$3;$4;$5\n"' > position_2_error_high_low.csv
#grep ', position 3, error ' $1 | perl -n -e'/Time ([^,]+), position 3, error ([^,]+), predicted ([^,]+), etalon high ([^,]+), etalon low([^,]+)/ && print "$1;$2;$3;$4;$5\n"' > position_3_error_high_low.csv
#grep ', position 4, error ' $1 | perl -n -e'/Time ([^,]+), position 4, error ([^,]+), predicted ([^,]+), etalon high ([^,]+), etalon low([^,]+)/ && print "$1;$2;$3;$4;$5\n"' > position_4_error_high_low.csv
#grep 'Position 0, error ' $1 | perl -n -e'/Position 0, error ([^,]+), predicted ([^,]+), etalon avg ([^,]+)/ && print "$1;$2;$3\n"' > position_0_error_mean.csv
#grep 'Position 1, error ' $1 | perl -n -e'/Position 1, error ([^,]+), predicted ([^,]+), etalon avg ([^,]+)/ && print "$1;$2;$3\n"' > position_1_error_mean.csv
#grep 'Position 2, error ' $1 | perl -n -e'/Position 2, error ([^,]+), predicted ([^,]+), etalon avg ([^,]+)/ && print "$1;$2;$3\n"' > position_2_error_mean.csv
#grep 'Position 3, error ' $1 | perl -n -e'/Position 3, error ([^,]+), predicted ([^,]+), etalon avg ([^,]+)/ && print "$1;$2;$3\n"' > position_3_error_mean.csv
#grep 'Position 4, error ' $1 | perl -n -e'/Position 4, error ([^,]+), predicted ([^,]+), etalon avg ([^,]+)/ && print "$1;$2;$3\n"' > position_4_error_mean.csv

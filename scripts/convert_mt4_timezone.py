import csv
import sys

import dateutil.parser as parser
from dateutil import tz

# arg 1 file name
# arg 2 from time zone eg. 'UTC'
# arg 3 to time zone eg. 'Europe/Zurich'
# arg 4 write 'DST' if DST is not added to data by provider


from_zone = tz.gettz(sys.argv[2])
to_zone = tz.gettz(sys.argv[3])
add_dst = sys.argv[4] == 'DST'

with open(sys.argv[1]) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_datetime = parser.parse(row[0] + " " + row[1])  # datetime.strptime(row[0] + " " + row[1], '%Y.%m.%d %H:%M')
        row_datetime = row_datetime.replace(tzinfo=from_zone)
        conv_row_datetime = row_datetime.astimezone(to_zone)
        if (add_dst): conv_row_datetime = conv_row_datetime + conv_row_datetime.dst()  # Add DST if not present
        conv_str = conv_row_datetime.strftime('%Y.%m.%d,%H:%M')
        print(f'{conv_str},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}')


#!/usr/bin/env python3

import psycopg2
from config import config
from datetime import datetime, timedelta

TIME_PERIOD=4
PARAMS = config()
SQL_STATEMENT_AVG_INSERT = """(value_time, update_time, tick_volume, bid) VALUES(%s, %s, %s, %s);"""
SQL_STATEMENT_OHLC_INSERT = """(value_time, update_time, tick_volume, open, high, low, close) VALUES(%s, %s, %s, %s, %s, %s, %s);"""
SQL_STATEMENT_AVG_SELECT = """SELECT * from {} order by value_time desc"""

def get_next_time(last):
    next_time = last + timedelta(hours=-TIME_PERIOD)
    if next_time.weekday() > 4:
        next_time += timedelta(days=-2)
    return next_time    
        
def count_of_inserts_for_palindrome(input):
    input_length = len(input)
    last = input[input_length-1]
    for i, elem in enumerate(input):
        if elem[-1] == last[-1] and elem[-2] == last[-2]:
            return i
    return input_length-1

def fill_to_palindrome(ct_inserts, input):
    fill_data = []
    input_len = len(input)
    next_time = input[input_len-1][0]
    i = ct_inserts-1
    while i >= 0:	
        next_time = get_next_time(next_time)
        fill_data.append([next_time, datetime.now()] + list(input[i][2:]))
        i -= 1
    return fill_data

def get_queue_data():
    conn = None
    try:
        conn = psycopg2.connect(host=PARAMS['host'],database=PARAMS['database'], user=PARAMS['user'], password=PARAMS['password'])
		
        cur = conn.cursor()
        
        cur.execute(SQL_STATEMENT_AVG_SELECT.format(PARAMS['queue']))

        db_version = cur.fetchall()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return db_version

def upload_data(new_rows):
    try:
        conn = psycopg2.connect(host=PARAMS['host'],database=PARAMS['database'], user=PARAMS['user'], password=PARAMS['password'])                                                                                                                                                                          
        cur = conn.cursor()
        for elem in new_rows:
            print("Adding to queue " + str(elem))
            if "avg" in PARAMS['queue']:
                sql_statement = SQL_STATEMENT_AVG_INSERT
            else:
                sql_statement = SQL_STATEMENT_OHLC_INSERT
            print(sql_statement)
            cur.execute("INSERT INTO " + PARAMS['queue'] + sql_statement, tuple(elem))
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def get_missing_rows(queue_data):
    ct_rows = count_of_inserts_for_palindrome(queue_data)
    if ct_rows > 0:
        new_rows = fill_to_palindrome(ct_rows, queue_data)
        return new_rows

if __name__ == '__main__':
    queue_data = get_queue_data()
    new_rows = get_missing_rows(queue_data)
    if new_rows:
        upload_data(new_rows)

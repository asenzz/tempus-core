#!/usr/bin/bash

default_db=${1:-"svrwave"}
export PGPASSWORD= #'Qg3UfdGxAq2KpTuP'
export PGHOST=/tmp/pg
export PGUSER=asenzz
psql -d postgres -c "create database ${default_db}"
psql -d postgres -c "create role ${default_db}" 
psql -d postgres -c "grant all privileges on database ${default_db} to ${default_db}"
psql -d postgres -c "ALTER ROLE ${default_db} WITH LOGIN"
psql -d ${default_db} < dbschema.sql

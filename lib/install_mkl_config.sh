#!/usr/bin/bash

for ff in `find $ONEAPI_ROOT -type f -name MKLConfig.cmake`; do 
	sudo cp "$ff" "${ff}.backup"
	sudo cp -f MKLConfig.cmake "$ff"; 
done

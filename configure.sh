#!/bin/bash

function calcVars {
	APACHE_USER="www-data"
	SVRWEB_USER=$USER
	FASTCGI_DIR="/var/svrweb"
	FASTCGI_FIL="svrweb_fcgi"
	FASTCGI_FILENAME=$FASTCGI_DIR/$FASTCGI_FIL
	BASE_DIR=`pwd`
	SVRWEB_BINARY="$BASE_DIR/build/SVRWeb"
}

function printParams {
	CL='\033[0;30m'
	NC='\033[0m'

	echo -e $1 $CL"APACHE_USER=$APACHE_USER"
	echo -e $1 "SVRWEB_USER=$SVRWEB_USER"
	echo -e $1 "FASTCGI_FILENAME=$FASTCGI_FILENAME"$NC
}

function printShortHelp {
	echo 	"This is an automatic runtime and build configuration script"
	echo 	"Usage:./configure [OPTIONS] PARAMS..."

	echo -e "\tOPTIONS:"
	echo -e "\t\t-r: configure runtime only"
	echo -e	"\tPARAMS:All parameters are passed to cmake."
	echo 	"Current variables:"
	printParams "\t"
}

function configureRuntime {
 	
	if [ ! -d "$FASTCGI_DIR" ]; then
		sudo mkdir $FASTCGI_DIR
		sudo chown $SVRWEB_USER $FASTCGI_DIR
	fi

	if [ ! -f "$FASTCGI_FILENAME" ]; then
		touch $FASTCGI_FILENAME
	fi
 	
	sed -ir "s|/tmp/svrweb/svrweb_fcgi|$FASTCGI_FILENAME|g" config/config_fastcgi.json

	sudo usermod -a -G $SVRWEB_USER $APACHE_USER

	chmod g+rw $FASTCGI_FILENAME

	sed -ri "s|[$]SVRWEB_BINARY|$SVRWEB_BINARY|g" config/apache2/mods-available/fastcgi.conf
}

function configureBuild {

	local last_cpu=$((`lscpu -p | egrep "^[^#].*$" | wc -l` -1))

	cd $BASE_DIR
	mkdir build || rm -rf build/*
	cd build
	cmake "$@" ..
}

calcVars

printShortHelp

configureRuntime

if [ $# -gt 0 ]; then
	if [ $1="-r" ]; then
		exit
	fi
fi

configureBuild


#!/usr/bin/bash
read -r -d '' BACKUP_FILES << EOM

/home/zarko/.profile
/home/zarko/.bashrc
/home/zarko/.bash_history
/home/zarko/.gdbinit
/home/zarko/.gdbinit.local
/home/zarko/.psqlrc
/home/zarko/.psql_history*
/home/zarko/.gdb_history*
/home/zarko/.vimrc
/home/zarko/.tmux
/home/zarko/.gitconfig
/etc/malloc.conf
/etc/rc0.d/K01preparegpus
/etc/profile.d/*
/etc/profile
EOM

tar -cvf - ${BACKUP_FILES} | bzip2 -9cvf - > ../lib/env_backup.tar.bz2


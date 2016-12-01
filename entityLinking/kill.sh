pkill -9 python;
ps -ef |grep defunct |grep -v grep | cut -b8-20 | xargs kill -9

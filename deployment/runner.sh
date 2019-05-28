#!/bin/bash
# Runs a set of commands on a set of nodes

handler()
{
  echo "Killed"
        # kill all background processes
	while read -u 11 p; do
		echo "Terminating on $p..."
		# kill twice for all python subprocesses to die
		ssh  $p "pkill -f $killcmd" < /dev/tty
		ssh  $p "pkill -f $killcmd" < /dev/tty
#		echo $p | taktuk -c "oarsh" -f - broadcast exec [ pkill -f "python $exec"  ]
  done 11< $nodes

  exit -1
}

trap handler SIGINT

if [ "$1" = "" ] || [ "$2" = "" ] || [ "$3" == "" ]; then
  echo -e "usage: \nfrontend: $ $0 <commands_file> <node_file> <command_pattern>"
  echo -e "<commands_file>: \ncat <commands_file>\nexport name=TEST && echo \$name,echo \$hostname"
  echo -e "<node_file>: \nuniq \$OAR_NODEFILE > <nodes_file>"
  echo -e "<command_pattern>: /path/to/driver.py"
  exit 1
fi

killcmd=$3
nodes=$2 # avoid ambigous redirect issues when killing
i=1 # first line-command to deploy
END=$(wc -l < "$1") # last line
while ((i <= $END)); do
  while read -u 10 p; do
    # check if job is running on this node
    ssh $p "pgrep -f $killcmd" > /dev/null
    if [ $? -eq 0 ]
    then
      echo "Job already running on: "$p
    else
      echo "Deploying job "$i " on: "$p
   
      IFS=, read loc remote <<< `sed -n "$i p" $1` # read command
      eval $loc # execute local one
      remoteEval=$(echo $remote | envsubst)
      echo $remoteEval
      ssh $p $remoteEval & # execute remote
      
      i=$((i+1))
      if ((i > $END)); then
        break
      fi
  
    fi
  done 10<$nodes
  
  wait -n # wait for one of the processes to finish
  date

done

wait # wait for all processes to finish
echo "Finito"

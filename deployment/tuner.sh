#!/bin/bash
# Usefull for hyperparameter tuning

handler()
{
  echo "Killed"
        # kill all background processes
	while read p; do
		echo "Terminating on $p..."
		# kill twice for all python subprocesses to die
		ssh  $p "pkill -f /path/to/driver.py" < /dev/tty
		ssh  $p "pkill -f /path/to/driver.py" < /dev/tty
#		echo $p | taktuk -c "oarsh" -f - broadcast exec [ pkill -f "python $exec"  ]
  done < $tmpdir/nodes

  # clean up
  rm -r $tmpdir
  exit -1
}

trap handler SIGINT

if [ "$1" = "" ] || [ "$2" = "" ] || [ "$3" = "" ]; then
  echo -e "usage: \nfrontend: $ $0 <parameters_file> <output_dir> <node_file>"
  echo -e "Example: \ncat <parameters_file>\n0.14,0.02,200\n0.414,0.02,22\n0.21314,0.202,322"
  echo -e "Example: \nuniq \$OAR_NODEFILE > <nodes_file>"
  exit 1
fi


outdir=$2
mkdir -p $outdir
#tmpdir=`mktemp -d`
tmpdir=~/$(basename $(mktemp -u))
mkdir $tmpdir
echo $tmpdir
#uniq $OAR_NODEFILE > $tmpdir/nodes
cat $3 > $tmpdir/nodes

i=1 # first line of parameters_file to deploy job with
END=$(wc -l < "$1") # last line 
while ((i <= $END)); do
  while read -u 10 p; do
    # check if job is running on this node
    ssh $p "pgrep -f /path/to/driver.py" > /dev/null
    if [ $? -eq 0 ]
    then
      echo "Job already running on: "$p
    else
      echo "Deploying job "$i " on: "$p
   
      # READ PARAMETERS
      IFS=, read reg lrate factors <<< `sed -n "$i p" $1`
      echo "Deploying parameters: $reg | $lrate | $factors"

      tmp=`mktemp`
      sed "s/rec.factor.number=.*/rec.factor.number=$factors/g" \
        ~/discriminate_recom/config/pmf.properties > $tmp
     
      tmp2=`mktemp`
      sed "s/rec.user.regularization=.*/rec.user.regularization=$reg/g" \
        $tmp > $tmp2
     
      tmp3=`mktemp`
      sed "s/rec.item.regularization=.*/rec.item.regularization=$reg/g" \
        $tmp2 > $tmp3
     
      tmp4=`mktemp`
      sed "s/rec.iterator.learnrate=.*/rec.iterator.learnrate=$lrate/g" \
        $tmp3 > $tmp4
  
      sed "s/rec.iterator.learnrate.maximum=.*/rec.iterator.learnrate.maximum=$lrate/g" \
        $tmp4 > $tmpdir/pmf${i}.properties
      
      rm $tmp $tmp1 $tmp2 $tmp3 $tmp4
  
#      cat $tmpdir/pmf${i}.properties
      
      threads=$(ssh $p "grep -c ^processor /proc/cpuinfo")
      ssh $p "python ~/discriminate_recom/git/driver.py \
      --pickleSavePath $outdir/$i.pickle \
      --known_dataset ~/discriminate_recom/datasets/ML100K.csv \
      --config $tmpdir/pmf${i}.properties \
      --proc $((threads-4)) \
      --validSize 2000 --testSize 2000 > $outdir/${i}.out 2>&1" &
  
      i=$((i+1))
      if ((i > $END)); then
        break
      fi
  
    fi
  done 10<$tmpdir/nodes
  
  wait -n # wait for one of the processes to finish
  date

done

# clean up
rm -r $tmpdir

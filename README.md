# RecRank 
Tool for exploiting black-box recommenders for algorithm selection.

## Main Components

### [recom.py](recom.py)
Executes recommendation algorithms from  and outputs the recommendations list

* Example execution:
	```bash
	python recom.py --topN_list 5 --dataset /fullPath/u.data --testSize 1000 --validSize 9000  --surprise_algo SVD --pickleSavePath /fullPath/list.pickle
	```

### [rec2graph.py](rec2graph.py)
Parses the output of a recommendation algorithm to a graph

* Example execution:
  ```bash
  python rec2graph.py --pickleLoadPath ./list.pickle --output ./graphs/graph.graph
  ```


### [distCalc.py](distCalc.py)
Calculates and visualizes the distance of recom algorithms

* Example execution:
  ```bash
  python distCalc.py --folder_paths  ./graphs/ --manual
  ``` 


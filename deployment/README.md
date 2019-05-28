## Deployment scripts

### [runner.sh](runner.sh)
Spawns parallel jobs for a list of input commands on g5k nodes

### [commandsCreator.py](commandsCreator.py)
Creates the input file for [runner.sh](runner.sh).

### [tuner.sh](tuner.sh)
Spawns parallel jobs for different hyperparameter values on g5k nodes

### [randomParams.py](randomParams.py)
Creates random tuples for hyperparameter tuning

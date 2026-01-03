Run the following commands to create the environment and install dependencies:

# 1. Create virtual environment
```
python -m venv venv
```

# 2. Activate environment
# Windows:
```
.\venv\Scripts\activate
```
# Mac/Linux:
```
source venv/bin/activate
```

# 3. Install requirements
```
pip install -r requirements.txt
```


Instruction for how to run each RL algo (all code is located in the algorithms folder):

1 - SAC

Code for models is all located in adequately named separate python files. To view results or run experiments, all code is located in the testSAC.ipynb notebook. 
It creates an environment, trains and evaluates. The model that were tested are located in the models folders so can easilty be loaded to replicated experiments. Other files within the SAC folder contain the results of various experiments.


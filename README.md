# CCPO: Conservatively Constrained Policy Optimization using State Augmentation

## About 


## Installation 

The following installation commands were tested in Ubuntu 16.04.

Our implementation is based on the Open AI safety starter agents(https://github.com/openai/safety-starter-agents) and SAUTE(https://github.com/huawei-noah/HEBO/tree/master/SAUTE). So make sure that you have installed the packages before you start running codes.



## Running 

We release a few tested safe environments, which can be evaluated using main.py. The file takes two arguments: the experiment identifier and the number of experiments for a particular algorithm to run in parallel. For instance, 

```console 
python main.py --experiment 14 --num-exps 5
```

The above command runs a program of CCPO-PPO over environment Pendulum.

The complete list of the experiment identifiers can be found in the main.py file and the comments section.


## Output 

The specific output path can be modified according to ‘exps/envs/algo.py‘. 

By default no checkpoints are saved, but the results are tracked in the tensorboard.


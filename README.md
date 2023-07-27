# CCPO: Conservatively Constrained Policy Optimization using State Augmentation

## About 

How to satisfy safety constraints almost surely (or with probability one) is becoming an emerging research issue for safe reinforcement learning (RL) algorithms in safety-critical domains. For instance, self-driving cars are expected to ensure that the driving strategy they adopt will never do harm to pedestrians and themselves. However, existing safe RL algorithms suffer from either risky and unstable constraint satisfaction or slow convergence. To tackle these two issues, we propose Conservatively Constrained Policy Optimization (CCPO) using state augmentation. CCPO designs a simple yet effective penalized reward function by introducing safety states and adaptive penalty factors under Safety Augmented MDP framework. Specifically, a novel Safety Promotion Function (SPF) is proposed to make the agent being more concentrated on constraint satisfaction with faster convergence by reshaping a more conservative constrained optimization objective. Moreover, we theoretically prove the convergence of CCPO. To validate both the effectiveness and efficiency of CCPO, comprehensive experiments are conducted in both single-constraint and more challenging multi-constraint environments. The experimental results demonstrate that the safe RL algorithms augmented by CCPO satisfy the predefined safety constraints almost surely and gain almost equivalent cumulative reward with faster convergence.

## Installation 

The following installation commands were tested in Ubuntu 16.04.

Our implementation is based on the Open AI safety starter agents(https://github.com/openai/safety-starter-agents) and SAUTE(https://github.com/huawei-noah/HEBO/tree/master/SAUTE (old) or https://github.com/huawei-noah/HEBO) (new). So make sure that you have installed the packages before you start running codes.



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


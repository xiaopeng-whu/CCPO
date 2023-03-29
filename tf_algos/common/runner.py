from typing import Dict
import numpy as np

import multiprocessing as mp
from common.base_runner import BaseRunner
from common.utils import set_overrides
import os
from tf_algos.safety_starter_agents.test_agents import evaluate_run

class TFRunner(BaseRunner):
    """Main runner class for tf-based algorithms."""
    def setup_algo(
                self,
                agent_cfg_overrides:Dict,
                env_cfg_overrides:Dict,
                exp_dir:str=''
    ):
        """
        Sets up the algorithm for training and testing 

        :param agent_cfg_overrides: dictionary with ovverides for the agent config files, 
        :param env_cfg_overrides: dictionary with ovverides for the environment config files, 
        :param exp_dir: directory for logging the experiment,
        
        :returns: two functions for training and testing the algorithms.
        """
        from tf_algos.safety_starter_agents.tf_ppo import ppo, saute_ppo, ppo_lagrangian, ccpo_ppo
        from tf_algos.safety_starter_agents.tf_trpo import trpo, saute_trpo, trpo_lagrangian, ccpo_trpo, ccpo_trpo2, trpo2, saute_trpo2, trpo_lagrangian2, ccpo_trpo_fixed_lamda, ccpo_trpo_without_z
        from tf_algos.safety_starter_agents.tf_cpo import cpo
        from tf_algos.safety_starter_agents.run_agents import polopt_cfg
        
        from tf_algos.safety_starter_agents.sac_utils import mlp_actor, mlp_critic
        from tf_algos.safety_starter_agents.tf_sac import vanilla_sac, saute_sac, lagrangian_sac, sac_cfg, ccpo_sac
        if 'PPO' in self.agent_name:
            if 'Saute' in self.agent_name:
               agent_cfg_overrides['env_name'] = 'Sauted' +  agent_cfg_overrides['env_name'] 
            agent_cfg = set_overrides(polopt_cfg, agent_cfg_overrides)
            train_env_fn, test_env_fn, agent_cfg, env_cfg = self.create_env(agent_cfg, env_cfg_overrides)
            self.writer, self.train_dir, self.test_dir = self.setup_log(exp_dir=exp_dir, agent_cfg=agent_cfg, env_cfg=env_cfg)
            agent_cfg['logger_kwargs'] = dict(output_dir=self.train_dir, output_fname='logs.txt')
            if self.agent_name == 'VanillaPPO':
                train_algo = lambda : ppo(
                    train_env_fn=train_env_fn,  
                    writer=self.writer, 
                    **agent_cfg
                )
            elif self.agent_name == 'LagrangianPPO':    
                train_algo = lambda : ppo_lagrangian(
                    train_env_fn=train_env_fn,  
                    writer=self.writer, 
                    **agent_cfg
                )
            elif self.agent_name == 'SautePPO':
                train_algo = lambda : saute_ppo(
                    train_env_fn=train_env_fn,  
                    writer=self.writer, 
                    **agent_cfg
                )
            elif self.agent_name == 'CPPO':
                train_algo = lambda : ccpo_ppo(
                    train_env_fn=train_env_fn,
                    writer=self.writer,
                    **agent_cfg
                )
            else:
                raise NotImplementedError(f"Agent {self.agent_name} is not implemented")
        elif 'TRPO' in self.agent_name:
            if 'Saute' in self.agent_name:# == 'SauteTRPO'
               agent_cfg_overrides['env_name'] = 'Sauted' +  agent_cfg_overrides['env_name']             
            agent_cfg = set_overrides(polopt_cfg, agent_cfg_overrides)
            train_env_fn, test_env_fn, agent_cfg, env_cfg = self.create_env(agent_cfg, env_cfg_overrides)
            self.writer, self.train_dir, self.test_dir = self.setup_log(exp_dir=exp_dir, agent_cfg=agent_cfg, env_cfg=env_cfg)
            agent_cfg['logger_kwargs'] = dict(output_dir=self.train_dir, output_fname='logs.txt')
            if self.agent_name == 'VanillaTRPO':
                train_algo = lambda : trpo(
                    train_env_fn=train_env_fn,  
                    writer=self.writer, 
                    **agent_cfg
                )
            elif self.agent_name == 'VanillaTRPO2':
                train_algo = lambda : trpo2(
                    train_env_fn=train_env_fn,  
                    writer=self.writer, 
                    **agent_cfg
                )
            elif self.agent_name == 'LagrangianTRPO':    
                train_algo = lambda : trpo_lagrangian(
                    train_env_fn=train_env_fn,  
                    writer=self.writer, 
                    **agent_cfg
                )
            elif self.agent_name == 'LagrangianTRPO2':    
                train_algo = lambda : trpo_lagrangian2(
                    train_env_fn=train_env_fn,  
                    writer=self.writer, 
                    **agent_cfg
                )
            elif self.agent_name == 'SauteTRPO':    
                train_algo = lambda : saute_trpo(
                    train_env_fn=train_env_fn,  
                    writer=self.writer, 
                    **agent_cfg
                )     
            elif self.agent_name == 'SauteTRPO2':    
                train_algo = lambda : saute_trpo2(
                    train_env_fn=train_env_fn,  
                    writer=self.writer, 
                    **agent_cfg
                )       
            elif self.agent_name == 'CTRPO':
                train_algo = lambda : ccpo_trpo(
                    train_env_fn=train_env_fn,
                    writer=self.writer,
                    **agent_cfg
                )
            elif self.agent_name == 'CTRPO2':
                train_algo = lambda : ccpo_trpo2(
                    train_env_fn=train_env_fn,
                    writer=self.writer,
                    **agent_cfg
                )
            elif self.agent_name == 'CTRPO_fixed_lamda':
                train_algo = lambda : ccpo_trpo_fixed_lamda(
                    train_env_fn=train_env_fn,
                    writer=self.writer,
                    **agent_cfg
                )
            elif self.agent_name == 'CTRPO_without_z':
                train_algo = lambda : ccpo_trpo_without_z(
                    train_env_fn=train_env_fn,
                    writer=self.writer,
                    **agent_cfg
                )
            else:
                raise NotImplementedError(f"Agent {self.agent_name} is not implemented")
        elif self.agent_name == 'CPO':    
            agent_cfg = set_overrides(polopt_cfg, agent_cfg_overrides)
            train_env_fn, test_env_fn, agent_cfg, env_cfg = self.create_env(agent_cfg, env_cfg_overrides)
            self.writer, self.train_dir, self.test_dir = self.setup_log(exp_dir=exp_dir, agent_cfg=agent_cfg, env_cfg=env_cfg)
            agent_cfg['logger_kwargs'] = dict(output_dir=self.train_dir, output_fname='logs.txt')
            train_algo = lambda : cpo(
                train_env_fn=train_env_fn,  
                writer=self.writer, 
                    **agent_cfg
            )
        elif 'SAC' in self.agent_name:  
            if 'Saute' in self.agent_name:
               agent_cfg_overrides['env_name'] = 'Sauted' +  agent_cfg_overrides['env_name'] 
            agent_cfg = set_overrides(sac_cfg, agent_cfg_overrides)
            train_env_fn, test_env_fn, agent_cfg, env_cfg = self.create_env(agent_cfg, env_cfg_overrides)
            self.writer, self.train_dir, self.test_dir = self.setup_log(exp_dir=exp_dir, agent_cfg=agent_cfg, env_cfg=env_cfg)
            agent_cfg['logger_kwargs'] = dict(output_dir=self.train_dir, output_fname='logs.txt')
            if self.agent_name == 'VanillaSAC':  
                train_algo = lambda: vanilla_sac(
                        train_env_fn=train_env_fn, 
                        test_env_fn=test_env_fn,  
                        writer=self.writer, 
                        **agent_cfg
                        )
            elif self.agent_name == 'LagrangianSAC':  
                train_algo = lambda: lagrangian_sac(
                        train_env_fn=train_env_fn, 
                        test_env_fn=test_env_fn,  
                        writer=self.writer, 
                        **agent_cfg
                        )
            elif self.agent_name == 'SauteSAC':  
                train_algo = lambda: saute_sac(
                        train_env_fn=train_env_fn, 
                        test_env_fn=test_env_fn,  
                        writer=self.writer, 
                        **agent_cfg
                        )
            elif self.agent_name == 'CSAC':  
                train_algo = lambda: ccpo_sac(
                        train_env_fn=train_env_fn, 
                        test_env_fn=test_env_fn,  
                        writer=self.writer, 
                        **agent_cfg
                        )                      
            else:
                raise NotImplementedError(f"Agent {self.agent_name} is not implemented")
        else:
            raise NotImplementedError(f"Agent {self.agent_name} is not implemented")
        test_algo = lambda last_only: evaluate_run(self.train_dir, env_fn=test_env_fn, evaluations=agent_cfg['n_test_episodes'], last_only=last_only)
        return train_algo, test_algo 

    def run_experiment(
            self,
            train:bool=False, 
            test:bool=False, 
            evaluate_last_only:bool=False,
            data_filename:str="test_results.csv",
            num_exps:int=1
    ):
        """
        Main run file for the algorithm  training and testing 

        :param train: if true train policy,
        :param test: if true test an already trained policy,
        :param evaluate_last_only: evaluates only the last iteration,
        :param data_filename: csv file containing the experimental data,
        :param num_exps: number of experiments to run simulatenously.
        """    
        agent_overrides, env_overrides, experiment_paths = self.set_all_overrides()

        if train:
            def _train_exps(count):
                train_algo, _ = self.setup_algo(
                    agent_cfg_overrides=agent_overrides[count],
                    env_cfg_overrides=env_overrides[count],
                    exp_dir=experiment_paths[count]
                )
                train_algo()  
            self._parallel_run(_train_exps, n_threads=num_exps, n_exps=len(experiment_paths))    
        if test:
            def _test_exps(count):
                    _, test_algo = self.setup_algo(
                        agent_cfg_overrides=agent_overrides[count],
                        env_cfg_overrides=env_overrides[count],
                        exp_dir=experiment_paths[count]
                    )
                    df = test_algo(evaluate_last_only)
                    df.to_csv(os.path.join(experiment_paths[count], "test", data_filename))                        
                        
            self._parallel_run(_test_exps, n_threads=num_exps, n_exps=len(experiment_paths))                  

    @staticmethod
    def _parallel_run(func, n_threads, n_exps):
        """
        Script for a parallel run of experiments,
        :param func: a function to run,
        :param n_threads: number of experiments to run simulatenously, 
        :param n_exps: total number of experiments to run.
        """
        if n_threads > 1:
            n_loops = int(np.ceil(n_exps / n_threads))
            for loop_idx in range(n_loops):
                cur_range = np.arange(loop_idx * n_threads, min((loop_idx + 1) * n_threads, n_exps))
                processes = []
                print("-----------------------------------")
                print(f"-------Starting Loop {loop_idx+1} / {n_loops}-------")
                print("-----------------------------------")
                for count in cur_range:
                    p = mp.Process(target=func, args=(count,))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
        else:
            for count in range(n_exps):
                func(count)

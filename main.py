from common.argument_parser import GeneralArgumentParser


if __name__ == "__main__":

    exp_parser = GeneralArgumentParser()
    args = exp_parser.parse_args()

    current_experiment = args.experiment    
### Single Pendulum        
    if current_experiment == 10:	#'VanillaSAC', 'LagrangianSAC', 'SauteSAC'
        from exps.single_pendulum.tf_sac_pendulum_v1 import run_tf_sac_pendulum_v1          
        run_tf_sac_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)    
    elif current_experiment == 11:	#'VanillaPPO', 'SautePPO', 'LagrangianPPO'
        from exps.single_pendulum.tf_ppo_pendulum_v1 import run_tf_ppo_pendulum_v1          
        run_tf_ppo_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)    
    elif current_experiment == 12:	#'CPO', 'SauteTRPO', 'VanillaTRPO', 'LagrangianTRPO'
        from exps.single_pendulum.tf_trpo_pendulum_v1 import run_tf_trpo_pendulum_v1          
        run_tf_trpo_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)        
    elif current_experiment == 13:	#CCPO-SAC
        from exps.single_pendulum.tf_ccpo_pendulum_v1 import run_tf_csac_pendulum_v1
        run_tf_sacrs_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)			
    elif current_experiment == 14:	#CCPO-PPO
        from exps.single_pendulum.tf_ccpo_pendulum_v1 import run_tf_cppo_pendulum_v1
        run_tf_ppors_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 15:	#CCPO-TRPO(1-logz, 2-z, 1/z)
        from exps.single_pendulum.tf_ccpo_pendulum_v1 import run_tf_ctrpo_pendulum_v1
        run_tf_trpors_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 16:	#'CCPO-TRPO_fixed_lamda'
        from exps.single_pendulum.tf_trpo_pendulum_v3 import run_tf_trpo_pendulum_v3          
        run_tf_trpo_pendulum_v3(num_exps=args.num_exps, smoketest=args.smoketest) 
    elif current_experiment == 17:	#'CCPO-TRPO_without_z'
        from exps.single_pendulum.tf_trpo_pendulum_v4 import run_tf_trpo_pendulum_v4          
        run_tf_trpo_pendulum_v4(num_exps=args.num_exps, smoketest=args.smoketest) 
    elif current_experiment == 18:	#'SauteTRPO'
        from exps.single_pendulum.tf_trpo_pendulum_v5 import run_tf_trpo_pendulum_v5          
        run_tf_trpo_pendulum_v5(num_exps=args.num_exps, smoketest=args.smoketest) 
### Reacher
    elif current_experiment == 30:	#'LagrangianTRPO', 'SauteTRPO', 'VanillaTRPO',  'CPO'
        from exps.reacher.tf_trpo_reacher_v1 import run_tf_trpo_reacher_v1
        run_tf_trpo_reacher_v1(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 31:	#CCPO-TRPO
        from exps.reacher.tf_trpo_reacher_v2 import run_tf_trpo_reacher_v2
        run_tf_trpo_reacher_v2(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 32:	#CCPO-TRPO for two-constraints env
        from exps.reacher.tf_trpo_reacher_v3 import run_tf_trpo_reacher_v3
        run_tf_trpo_reacher_v3(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 33:	#VanillaTRPO for two-constraints env
        from exps.reacher.tf_trpo_reacher_v4 import run_tf_trpo_reacher_v4
        run_tf_trpo_reacher_v4(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 34:	#SauteTRPO for two-constraints env
        from exps.reacher.tf_trpo_reacher_v5 import run_tf_trpo_reacher_v5
        run_tf_trpo_reacher_v5(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 35:	#LagrangianTRPO for two-constraints env
        from exps.reacher.tf_trpo_reacher_v6 import run_tf_trpo_reacher_v6
        run_tf_trpo_reacher_v6(num_exps=args.num_exps, smoketest=args.smoketest)
### Safety Gym
    elif current_experiment == 40:	#'VanillaTRPO'
        from exps.safety_gym.tf_trpo_safety_gym_v1 import run_tf_trpo_safety_gym_v1
        run_tf_trpo_safety_gym_v1(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 41:	#'LagrangianTRPO'
        from exps.safety_gym.tf_trpo_safety_gym_v2 import run_tf_trpo_safety_gym_v2
        run_tf_trpo_safety_gym_v2(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 42:	#'CPO'
        from exps.safety_gym.tf_trpo_safety_gym_v3 import run_tf_trpo_safety_gym_v3
        run_tf_trpo_safety_gym_v3(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 43:	#'SauteTRPO'
        from exps.safety_gym.tf_trpo_safety_gym_v4 import run_tf_trpo_safety_gym_v4
        run_tf_trpo_safety_gym_v4(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 44:	#CCPO-TRPO
        from exps.safety_gym.tf_trpo_safety_gym_v5 import run_tf_trpo_safety_gym_v5
        run_tf_trpo_safety_gym_v5(num_exps=args.num_exps, smoketest=args.smoketest)

    else:
        raise NotImplementedError 
 
    print("done.") 
     
    
   

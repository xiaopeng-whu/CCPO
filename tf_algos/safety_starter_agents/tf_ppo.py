from tf_algos.safety_starter_agents.agents import PPOAgent
from tf_algos.safety_starter_agents.run_agents import run_polopt_agent

# added env_name parameter for easy of overriding
def ppo(**kwargs):   
    """Set up to run Vanilla PPO."""     
    ppo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=False,
                    learn_penalty=False,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = PPOAgent(**ppo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)

def saute_ppo(**kwargs):    
    """Set up to run Saute PPO."""     
    ppo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=False,
                    learn_penalty=False,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = PPOAgent(**ppo_kwargs)
    kwargs['saute_constraints'] = True
    run_polopt_agent(agent=agent, **kwargs)

def ccpo_ppo(**kwargs):
    """Set up to run CCPO PPO."""
    ppo_kwargs = dict(
                    reward_penalized=True,
                    objective_penalized=False,
                    learn_penalty=True,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = PPOAgent(**ppo_kwargs)
    kwargs['saute_constraints'] = True
    kwargs['rs'] = True
    run_polopt_agent(agent=agent, **kwargs)

def ppo_lagrangian(**kwargs):
    """Set up to run PPO Lagrangian."""     
    # Objective-penalized form of Lagrangian PPO.
    ppo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=True,
                    learn_penalty=True,
                    penalty_param_loss=True
                    )
    agent = PPOAgent(**ppo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)


from safe_rl.pg.agents import TRPOAgent
from tf_algos.safety_starter_agents.run_agents import run_polopt_agent
from tf_algos.safety_starter_agents.run_agents2 import run_polopt_agent2


def trpo(**kwargs):
    """Run Vanilla TRPO."""  
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=False,
                    learn_penalty=False,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = TRPOAgent(**trpo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)

def trpo2(**kwargs):
    """Run Vanilla TRPO for multi-constraint env."""  
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=False,
                    learn_penalty=False,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = TRPOAgent(**trpo_kwargs)
    run_polopt_agent2(agent=agent, **kwargs)


def saute_trpo(**kwargs):
    """Run Saute TRPO."""  
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=False,
                    learn_penalty=False,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = TRPOAgent(**trpo_kwargs)        
    kwargs['saute_constraints'] = True
    run_polopt_agent(agent=agent, **kwargs)

def saute_trpo2(**kwargs):
    """Run Saute TRPO."""  
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=False,
                    learn_penalty=False,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = TRPOAgent(**trpo_kwargs)        
    kwargs['saute_constraints'] = True
    run_polopt_agent2(agent=agent, **kwargs)

def ccpo_trpo(**kwargs):
    """Run Saute TRPO RS."""  
    trpo_kwargs = dict(
                    reward_penalized=True,
                    objective_penalized=False,
                    learn_penalty=True,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = TRPOAgent(**trpo_kwargs)        
    kwargs['saute_constraints'] = True
    kwargs['rs'] = True
    run_polopt_agent(agent=agent, **kwargs)

def ccpo_trpo2(**kwargs):
    """Run Saute TRPO RS for multi-constraint env."""  
    trpo_kwargs = dict(
                    reward_penalized=True,
                    objective_penalized=False,
                    learn_penalty=True,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = TRPOAgent(**trpo_kwargs)        
    kwargs['saute_constraints'] = True
    kwargs['rs'] = True
    run_polopt_agent2(agent=agent, **kwargs)

def ccpo_trpo_fixed_lamda(**kwargs):
    """Run Saute TRPO RS with fixed lamda."""  
    trpo_kwargs = dict(
                    reward_penalized=True,
                    objective_penalized=False,
                    learn_penalty=False,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = TRPOAgent(**trpo_kwargs)        
    kwargs['saute_constraints'] = True
    kwargs['fixed_lamda'] = True
    run_polopt_agent(agent=agent, **kwargs)

def ccpo_trpo_without_z(**kwargs):
    """Run Saute TRPO RS without (1-z)."""  
    trpo_kwargs = dict(
                    reward_penalized=True,
                    objective_penalized=False,
                    learn_penalty=True,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = TRPOAgent(**trpo_kwargs)        
    kwargs['saute_constraints'] = True
    kwargs['without_z'] = True
    run_polopt_agent(agent=agent, **kwargs)


def trpo_lagrangian(**kwargs):
    """Run TRPO Lagrangian."""  
    # Objective-penalized form of Lagrangian TRPO.
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=True,
                    learn_penalty=True,
                    penalty_param_loss=True,
                    backtrack_iters=kwargs['backtrack_iters']
                    )
    agent = TRPOAgent(**trpo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)

def trpo_lagrangian2(**kwargs):
    """Run TRPO Lagrangian."""  
    # Objective-penalized form of Lagrangian TRPO.
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=True,
                    learn_penalty=True,
                    penalty_param_loss=True,
                    backtrack_iters=kwargs['backtrack_iters']
                    )
    agent = TRPOAgent(**trpo_kwargs)
    run_polopt_agent2(agent=agent, **kwargs)



"""
Main run file copied from safety starter agents with the following modifications:
- added a capability for Sauteing Vanilla and Lagrangian methods
- added a capability to use CVaR constraints
"""
import numpy as np
from tensorboardX.writer import SummaryWriter
import tensorflow as tf
import pandas as pd
import time
import safe_rl.pg.trust_region as tro
from tf_algos.safety_starter_agents.agents import PPOAgent
from safe_rl.pg.buffer import CPOBuffer
from safe_rl.pg.network import count_vars, \
                               get_vars, \
                               mlp_actor_critic,\
                               placeholders, \
                               placeholders_from_spaces
from safe_rl.pg.utils import values_as_sorted_list
from safe_rl.utils.logx import EpochLogger
from safe_rl.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from safe_rl.utils.mpi_tools import *
from collections import deque
import math

polopt_cfg = dict(
    # from our torch runners
    log=True, 
    log_updates=False, 
    seed=0,
    render=False,
    saute_constraints=False,
    saute_lagrangian=False,
    # Experience collection:
    steps_per_epoch=4000, 
    epochs=200, 
    # max_ep_len=0,  # removed as it should be taken from env_cfg
    # Discount factors:
    discount_factor=0.99, 
    lam=0.97,
    safety_discount_factor=0.99, 
    safety_lam=0.97, 
    safety_budget=1.0, 
    # Policy learning:
    ent_reg=0.,
    # Cost constraints / penalties:
    penalty_init=1.,
    penalty_lr=5e-2,
    # KL divergence:
    target_kl=0.01, 
    # Value learning:
    value_fn_lr=1e-3,
    gradient_steps=80, 
    # Logging:
    checkpoint_frequency=1,
    n_test_episodes=100,
    n_train_episodes=100,
    backtrack_iters=15
    )

class CVaREpochLogger(EpochLogger):
    def __init__(self, risk=0.9, **kwargs):
        super(CVaREpochLogger, self).__init__(**kwargs)
        self.risk = risk

    def get_stats_cvar(self, key):
        """
        Lets an algorithm ask the logger for CVaR at risk=alpha.
        """
        v = self.epoch_dict[key]
        # Each episode cost in a list, for each process
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
        # Assuming only one process, so all the data of cum costs in one list, calculate CVaR
        n = len(vals)
        # Calculate k for removed smallest k values from list
        k = n-int(n*self.risk)
        # Sort with k smalles values first then remove these and keep n-k largest costs
        vals = np.partition(vals, k)[k:]
        return mpi_statistics_scalar(vals)

def run_polopt_agent(train_env_fn,
                     env_name:str='',
                     log:bool=True, 
                     log_updates:bool=False, 
                     agent=PPOAgent(),
                     actor_critic=mlp_actor_critic, 
                     ac_kwargs=dict(), 
                     seed=0,
                     render=False,
                     saute_constraints=False,
                     saute_lagrangian=False,
                     rs=False,
                     fixed_lamda=False,
                     without_z=False,
                     # Experience collection:
                     steps_per_epoch=4000, 
                     epochs=50, 
                     max_ep_len=200,
                     # Discount factors:
                     discount_factor=0.99, 
                     lam=0.97,
                     safety_discount_factor=0.99, 
                     safety_lam=0.97, 
                     # Policy learning:
                     ent_reg=0.,
                     # Cost constraints / penalties:
                     safety_budget=25,
                     penalty_init=1.,
                     penalty_lr=5e-2,
                     # KL divergence:
                     target_kl=0.01, 
                     # Value learning:
                     value_fn_lr=1e-3,
                     gradient_steps=80, 
                     # TB logging 
                     writer:SummaryWriter=None,
                     # Logging:
                     logger=None, 
                     logger_kwargs=dict(), 
                     checkpoint_frequency=1,
                     CVaR=False,
                     risk=0.9,
                     n_test_episodes=10,
                     n_train_episodes=10,
                     algo_name='ppo',
                     backtrack_iters=10
                     ):
    #=========================================================================#
    #  Prepare logger, seed, and environment in this process                  #
    #=========================================================================#
    if CVaR:
        # MUST HAVE --cpu set to 1 or number of cpus set to 1 otherwise it won't work!
        logger = CVaREpochLogger(risk=risk, **logger_kwargs)

    else:
        logger = EpochLogger(**logger_kwargs) if logger is None else logger
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = train_env_fn()
    observation_space = env.observation_space 
    
    agent.set_logger(logger)

    #=========================================================================#
    #  Create computation graph for actor and critic (not training routine)   #
    #=========================================================================#

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph from environment spaces
    x_ph, a_ph = placeholders_from_spaces(observation_space, env.action_space)

    # Inputs to computation graph for batch data
    adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph = placeholders(*(None for _ in range(5)))

    # Inputs to computation graph for special purposes
    surr_cost_rescale_ph = tf.placeholder(tf.float32, shape=())
    cur_cost_ph = tf.placeholder(tf.float32, shape=())

    # Outputs from actor critic
    ac_outs = actor_critic(x_ph, a_ph, **ac_kwargs)
    pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent, v, vc = ac_outs

    # Organize placeholders for zipping with data from buffer on updates
    buf_phs = [x_ph, a_ph, adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph]
    buf_phs += values_as_sorted_list(pi_info_phs)

    # Organize symbols we have to compute at each step of acting in env
    get_action_ops = dict(pi=pi, 
                          v=v, 
                          logp_pi=logp_pi,
                          pi_info=pi_info)

    # If agent is reward penalized, it doesn't use a separate value function
    # for costs and we don't need to include it in get_action_ops; otherwise we do.
    if not(agent.reward_penalized):
        get_action_ops['vc'] = vc

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'vf', 'vc'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n'%var_counts)
    # print('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n'%var_counts)

    # Make a sample estimate for entropy to use as sanity check
    approx_ent = tf.reduce_mean(-logp)


    #=========================================================================#
    #  Create replay buffer                                                   #
    #=========================================================================#

    # Obs/act shapes
    obs_shape = observation_space.shape
    act_shape = env.action_space.shape

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    pi_info_shapes = {k: v.shape.as_list()[1:] for k,v in pi_info_phs.items()}
    buf = CPOBuffer(local_steps_per_epoch,
                    obs_shape, 
                    act_shape, 
                    pi_info_shapes, 
                    discount_factor, 
                    lam,
                    safety_discount_factor,
                    safety_lam)


    #=========================================================================#
    #  Create computation graph for penalty learning, if applicable           #
    #=========================================================================#
    #penalty_init=2.5
    if agent.use_penalty:
        with tf.variable_scope('penalty'):
            # param_init = np.log(penalty_init)
            param_init = np.log(max(np.exp(penalty_init)-1, 1e-8))
            penalty_param = tf.get_variable('penalty_param',
                                          initializer=float(param_init),
                                          trainable=agent.learn_penalty,
                                          dtype=tf.float32)
        # penalty = tf.exp(penalty_param)
        penalty = tf.nn.softplus(penalty_param)

    if agent.learn_penalty:
        if agent.penalty_param_loss:
            penalty_loss = -penalty_param * (cur_cost_ph - safety_budget)
        else:
            penalty_loss = -penalty * (cur_cost_ph - safety_budget)
            #penalty_loss = -penalty * (cur_cost_ph - 30)
        train_penalty = MpiAdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)


    #=========================================================================#
    #  Create computation graph for policy learning                           #
    #=========================================================================#

    # Likelihood ratio
    ratio = tf.exp(logp - logp_old_ph)

    # Surrogate advantage / clipped surrogate advantage
    if agent.clipped_adv:
        min_adv = tf.where(adv_ph>0, 
                           (1+agent.clip_ratio)*adv_ph, 
                           (1-agent.clip_ratio)*adv_ph
                           )
        surr_adv = tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    else:
        surr_adv = tf.reduce_mean(ratio * adv_ph)

    # Surrogate cost
    surr_cost = tf.reduce_mean(ratio * cadv_ph)

    # Create policy objective function, including entropy regularization
    pi_objective = surr_adv + ent_reg * ent

    # Possibly include surr_cost in pi_objective
    if agent.objective_penalized:
        pi_objective -= penalty * surr_cost
        pi_objective /= (1 + penalty)

    # Loss function for pi is negative of pi_objective
    pi_loss = -pi_objective

    # Optimizer-specific symbols
    if agent.trust_region:

        # Symbols needed for CG solver for any trust region method
        pi_params = get_vars('pi')
        flat_g = tro.flat_grad(pi_loss, pi_params)
        v_ph, hvp = tro.hessian_vector_product(d_kl, pi_params)
        if agent.damping_coeff > 0:
            hvp += agent.damping_coeff * v_ph

        # Symbols needed for CG solver for CPO only
        flat_b = tro.flat_grad(surr_cost, pi_params)

        # Symbols for getting and setting params
        get_pi_params = tro.flat_concat(pi_params)
        set_pi_params = tro.assign_params_from_flat(v_ph, pi_params)

        training_package = dict(flat_g=flat_g,
                                flat_b=flat_b,
                                v_ph=v_ph,
                                hvp=hvp,
                                get_pi_params=get_pi_params,
                                set_pi_params=set_pi_params)

    elif agent.first_order:

        # Optimizer for first-order policy optimization
        train_pi = MpiAdamOptimizer(learning_rate=agent.pi_lr).minimize(pi_loss)

        # Prepare training package for agent
        training_package = dict(train_pi=train_pi)

    else:
        raise NotImplementedError

    # Provide training package to agent
    training_package.update(dict(pi_loss=pi_loss, 
                                 surr_cost=surr_cost,
                                 d_kl=d_kl, 
                                 target_kl=target_kl,
                                 cost_lim=safety_budget))
    agent.prepare_update(training_package)

    #=========================================================================#
    #  Create computation graph for value learning                            #
    #=========================================================================#

    # Value losses
    v_loss = tf.reduce_mean((ret_ph - v)**2)
    vc_loss = tf.reduce_mean((cret_ph - vc)**2)

    # If agent uses penalty directly in reward function, don't train a separate
    # value function for predicting cost returns. (Only use one vf for r - p*c.)
    if agent.reward_penalized:
        total_value_loss = v_loss
    else:
        total_value_loss = v_loss + vc_loss

    # Optimizer for value learning
    train_vf = MpiAdamOptimizer(learning_rate=value_fn_lr).minimize(total_value_loss)


    #=========================================================================#
    #  Create session, sync across procs, and set up saver                    #
    #=========================================================================#

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(
        sess, 
        inputs={'x': x_ph}, 
        outputs={'pi': pi, 'v': v, 'vc': vc}
        )


    #=========================================================================#
    #  Provide session to agent                                               #
    #=========================================================================#
    agent.prepare_session(sess)


    #=========================================================================#
    #  Create function for running update (called at end of each epoch)       #
    #=========================================================================#

    def update(epoch:int):
        if CVaR:
            cur_cost = logger.get_stats_cvar('EpCost')[0]
        else:
            cur_cost = logger.get_stats('EpCost')[0]
            #cur_cost = logger.get_stats('EpCost1')[0]
        c = cur_cost - safety_budget
        #c = cur_cost - 30
        #print("cur_cost:", cur_cost)
        if c > 0 and agent.cares_about_cost:
            logger.log('Warning! Safety constraint is already violated.', 'red')

        #=====================================================================#
        #  Prepare feed dict                                                  #
        #=====================================================================#

        inputs = {k:v for k,v in zip(buf_phs, buf.get())}
        inputs[surr_cost_rescale_ph] = logger.get_stats('EpLen')[0]
        inputs[cur_cost_ph] = cur_cost

        #=====================================================================#
        #  Make some measurements before updating                             #
        #=====================================================================#

        measures = dict(LossPi=pi_loss,
                        SurrCost=surr_cost,
                        SurrAdv=surr_adv,
                        LossV=v_loss,
                        Entropy=ent)
        if not(agent.reward_penalized):
            measures['LossVC'] = vc_loss
            # if writer is not None:
            #     writer.add_scalar('opt_info/LossVC',vc_loss, epoch)
        if agent.use_penalty:
            measures['Penalty'] = penalty
            # if writer is not None:
            #     writer.add_scalar('opt_info/Penalty',penalty, epoch)

        pre_update_measures = sess.run(measures, feed_dict=inputs)
        logger.store(**pre_update_measures)
        # if writer is not None:
        #     writer.add_scalar('opt_info/LossPi',pi_loss, epoch)
        #     writer.add_scalar('opt_info/SurrCost',surr_cost, epoch)
        #     writer.add_scalar('opt_info/LossV',v_loss, epoch)
        #     writer.add_scalar('opt_info/Entropy',ent, epoch)
        #=====================================================================#
        #  Update penalty if learning penalty                                 #
        #=====================================================================#
        if agent.learn_penalty:
            sess.run(train_penalty, feed_dict={cur_cost_ph: cur_cost})

        #=====================================================================#
        #  Update policy                                                      #
        #=====================================================================#
        agent.update_pi(inputs)

        #=====================================================================#
        #  Update value function                                              #
        #=====================================================================#
        for _ in range(gradient_steps):
            sess.run(train_vf, feed_dict=inputs)

        #=====================================================================#
        #  Make some measurements after updating                              #
        #=====================================================================#

        del measures['Entropy']
        measures['KL'] = d_kl

        post_update_measures = sess.run(measures, feed_dict=inputs)
        deltas = dict()
        for k in post_update_measures:
            if k in pre_update_measures:
                deltas['Delta'+k] = post_update_measures[k] - pre_update_measures[k]
        logger.store(KL=post_update_measures['KL'], **deltas)




    #=========================================================================#
    #  Run main environment interaction loop                                  #
    #=========================================================================#

    start_time = time.time()
    o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    ep_cost1 = 0
    cur_penalty = 0
    cum_cost = 0
    if saute_constraints:
        true_ep_ret = 0
    training_rewards = deque([0], maxlen=n_train_episodes)  
    training_costs = deque([0], maxlen=n_train_episodes)  
    df = pd.DataFrame()
    robot_pos = []
    record_pos = False
    start_record = False
    for epoch in range(epochs):
        pos = []
        if agent.use_penalty:
            cur_penalty = sess.run(penalty)

        for t in range(local_steps_per_epoch):

            # Possibly render
            if render and proc_id()==0 and t < 1000:
                env.render()
            
            # Get outputs from policy
            get_action_outs = sess.run(get_action_ops, 
                                       feed_dict={x_ph: o[np.newaxis]})
            a = get_action_outs['pi']
            v_t = get_action_outs['v']
            vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
            logp_t = get_action_outs['logp_pi']
            pi_info_t = get_action_outs['pi_info']

            # Step in environment
            o2, r, d, info = env.step(a)
            
            # if saute_constraints and (objective_penalized and penalty_param_loss and ):

            # Include penalty on cost
            c = info.get('cost', 0)

            # Track cumulative cost over training
            cum_cost += c
            if saute_lagrangian:
                r = info['true_reward']
            # save and log
            if agent.reward_penalized:
                if rs:
                    if info['next_safety_state'] > 0:
                        #r_total = r - cur_penalty*(float(1/(info['next_safety_state']))) * c
                        #r_total = r - cur_penalty*(1-math.log(info['next_safety_state'])) * c
                        r_total = r - cur_penalty*(2-info['next_safety_state']) * c
                    else:
                        r_total = 0
                        #r_total = -100	
                    buf.store(o, a, r_total, v_t, c, vc_t, logp_t, pi_info_t)
                elif fixed_lamda:
                    if info['next_safety_state'] > 0:
                        #r_total = r - 0.1*(2-info['next_safety_state']) * c
                        #r_total = r - 10*(2-info['next_safety_state']) * c
                        #r_total = r - 0.1*(1-math.log(info['next_safety_state'])) * c
                        r_total = r - 10*(1-math.log(info['next_safety_state'])) * c
                    else:
                        r_total = 0
                    buf.store(o, a, r_total, v_t, c, vc_t, logp_t, pi_info_t)
                elif without_z:
                    if info['next_safety_state'] > 0:
                        r_total = r - cur_penalty*c
                    else:
                        r_total = 0
                    buf.store(o, a, r_total, v_t, c, vc_t, logp_t, pi_info_t)
            else:
                r_total = r
                #if info['next_safety_state'] < 0:
                    #r_total = 0
                #else:
                    #r_total = r
                #buf.store(o, a, r, v_t, c, vc_t, logp_t, pi_info_t)
                buf.store(o, a, r_total, v_t, c, vc_t, logp_t, pi_info_t)
            logger.store(VVals=v_t, CostVVals=vc_t)

            o = o2
            ep_ret += r
            if saute_constraints:
                true_ep_ret += info['true_reward']
            ep_cost += c
            if rs:
                #ep_cost1 += (2-info['next_safety_state']) * c
                if info['next_safety_state'] > 0:
                    ep_cost1 += (2-info['next_safety_state']) * c
                    #ep_cost1 += (1-math.log(info['next_safety_state'])) * c
                    #ep_cost1 += (float(1/(info['next_safety_state']))) * c
                else:
                    ep_cost1 += c
            else:
                ep_cost1 = ep_cost
            ep_len += 1

            if (record_pos and start_record):
                p = info['pos']
                pos.append(p)

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):

                # If trajectory didn't reach terminal state, bootstrap value target(s)
                if d and not(ep_len == max_ep_len):
                    # Note: we do not count env time out as true terminal state
                    last_val, last_cval = 0, 0
                else:
                    feed_dict={x_ph: o[np.newaxis]}
                    if agent.reward_penalized:
                        last_val = sess.run(v, feed_dict=feed_dict)
                        last_cval = 0
                    else:
                        last_val, last_cval = sess.run([v, vc], feed_dict=feed_dict)
                buf.finish_path(last_val, last_cval)

                # Only save EpRet / EpLen if trajectory finished
                if terminal:
                    if rs:
                        logger.store(EpRet=true_ep_ret, EpLen=ep_len, EpCost=ep_cost, EpCost1=ep_cost1)
                        training_rewards.extend([true_ep_ret])
                    if fixed_lamda:
                        logger.store(EpRet=true_ep_ret, EpLen=ep_len, EpCost=ep_cost, EpCost1=ep_cost1)
                        training_rewards.extend([true_ep_ret])
                    elif saute_constraints:
                        logger.store(EpRet=true_ep_ret, EpLen=ep_len, EpCost=ep_cost)
                        training_rewards.extend([true_ep_ret])
                    else:
                        logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                        training_rewards.extend([ep_ret])
                    training_costs.extend([ep_cost]) 

                else:
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)

                if (record_pos and start_record):
                    robot_pos.append(pos)
                    pos = []
                # Reset environment
                o, r, d, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0
                ep_cost1 = 0
                if saute_constraints:
                    true_ep_ret = 0
        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch) * max_ep_len
        # Save model
        if (checkpoint_frequency and (epoch % checkpoint_frequency == 0)) or (epoch == epochs-1):
            logger.save_state({'env': env}, epoch)
            df = df.append(pd.DataFrame({
                    "episode_return": training_rewards, 
                    "episode_cost": training_costs,
                    "accumulated_cost": cumulative_cost,
                    "cost_rate": cost_rate,
                    "epoch": epoch,
                    "run": np.arange(len(training_rewards))
                }))            
            df.to_csv(os.path.join(logger.output_dir, "train_results.csv"))

        #=====================================================================#
        #  Run RL update                                                      #
        #=====================================================================#
        update(epoch=epoch)

        # cumulative_cost = mpi_sum(cum_cost)
        # cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)

        #=====================================================================#
        #  Log performance and stats                                          #
        #=====================================================================#

        logger.log_tabular('Epoch', epoch)

        # Performance stats
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CumulativeCost', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)

        # Value function values
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('CostVVals', with_min_and_max=True)

        # Pi loss and change
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)

        # Surr adv and change
        logger.log_tabular('SurrAdv', average_only=True)
        logger.log_tabular('DeltaSurrAdv', average_only=True)

        # Surr cost and change
        logger.log_tabular('SurrCost', average_only=True)
        logger.log_tabular('DeltaSurrCost', average_only=True)

        # V loss and change
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        true_objective = logger.log_current_row['AverageVVals']

        if writer is not None:
            # optimization infos
            writer.add_scalar('train_info/LossPi', logger.log_current_row['LossPi'], epoch)
            writer.add_scalar('train_info/DeltaLossPi', logger.log_current_row['DeltaLossPi'], epoch)
            writer.add_scalar('train_info/SurrAdv', logger.log_current_row['SurrAdv'], epoch)
            writer.add_scalar('train_info/DeltaSurrAdv', logger.log_current_row['DeltaSurrAdv'], epoch)
            writer.add_scalar('train_info/std_V_values', logger.log_current_row['StdVVals'], epoch)
            writer.add_scalar('train_info/mean_V_values', logger.log_current_row['AverageVVals'], epoch)
            writer.add_scalar('train_info/std_CostVVals', logger.log_current_row['StdCostVVals'], epoch)
            writer.add_scalar('train_info/mean_CostVVals', logger.log_current_row['AverageCostVVals'], epoch)
            writer.add_scalar('train_info/mean_SurrCost', logger.log_current_row['SurrCost'], epoch)
            writer.add_scalar('train_info/mean_LossV', logger.log_current_row['LossV'], epoch)
            writer.add_scalar('train_info/mean_DeltaSurrCost', logger.log_current_row['DeltaSurrCost'], epoch)
            writer.add_scalar('train_info/mean_DeltaLossV', logger.log_current_row['DeltaLossV'], epoch)
            # episode return 
            writer.add_scalar('train_return/StdEpRet', logger.log_current_row['StdEpRet'], epoch)
            writer.add_scalar('train_return/AverageEpRet', logger.log_current_row['AverageEpRet'], epoch)
            writer.add_scalar('train_return/MaxEpRet', logger.log_current_row['MaxEpRet'], epoch)
            writer.add_scalar('train_return/MinEpRet', logger.log_current_row['MinEpRet'], epoch)
            # episode cost 
            writer.add_scalar('train_cost/StdEpCost', logger.log_current_row['StdEpCost'], epoch)
            writer.add_scalar('train_cost/AverageEpCost', logger.log_current_row['AverageEpCost'], epoch)
            writer.add_scalar('train_cost/MaxEpCost', logger.log_current_row['MaxEpCost'], epoch)
            writer.add_scalar('train_cost/MinEpCost', logger.log_current_row['MinEpCost'], epoch)
            # accumulative cost
            writer.add_scalar('train_acc_cost/CumulativeCost', logger.log_current_row['CumulativeCost'], epoch)
            writer.add_scalar('train_acc_cost/CostRate', logger.log_current_row['CostRate'], epoch)

        # Vc loss and change, if applicable (reward_penalized agents don't use vc)
        if not(agent.reward_penalized):
            logger.log_tabular('LossVC', average_only=True)
            logger.log_tabular('DeltaLossVC', average_only=True)
            if writer:
                writer.add_scalar('train_info/mean_LossVC', logger.log_current_row['LossVC'], epoch)
                writer.add_scalar('train_info/mean_DeltaLossVC', logger.log_current_row['DeltaLossVC'], epoch)

        if agent.use_penalty or agent.save_penalty:
            logger.log_tabular('Penalty', average_only=True)
            logger.log_tabular('DeltaPenalty', average_only=True)
            true_objective += logger.log_current_row['Penalty'] * (safety_budget - logger.log_current_row['AverageCostVVals'])
            if writer:
                writer.add_scalar('train_info/Penalty', logger.log_current_row['Penalty'], epoch)
                writer.add_scalar('train_info/DeltaPenalty', logger.log_current_row['DeltaPenalty'], epoch)
                writer.add_scalar('train_info/True_objective', true_objective, epoch)
        else:
            logger.log_tabular('Penalty', 0)
            logger.log_tabular('DeltaPenalty', 0)
            if writer:
                writer.add_scalar('train_info/Penalty', 0, epoch)
                writer.add_scalar('train_info/DeltaPenalty', 0, epoch)
                writer.add_scalar('train_info/True_objective', true_objective, epoch)

        # Anything from the agent?
        agent.log()

        # Policy stats
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)

        # Time and steps elapsed
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)

        # Show results!
        logger.dump_tabular()

        if record_pos:
            if epoch==19:
                start_record = False
                a=np.array(robot_pos)
                np.save('robot_pos_beginning.npy',a)
                robot_pos = []
            elif epoch==999:
                start_record = True
            elif epoch==1019:
                start_record = False
                a=np.array(robot_pos)
                np.save('robot_pos_middle.npy',a)
                robot_pos = []
            elif epoch==1979:
                start_record = True
            elif epoch ==1999:
                a=np.array(robot_pos)
                np.save('robot_pos_end.npy',a)
                robot_pos = []

    sess.close() 
    tf.reset_default_graph()

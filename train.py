import argparse
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter

from unityagents import UnityEnvironment

from ddpg_agent import Agent

NUM_AGENTS = 2
STATE_SIZE = 24
ACTION_SIZE = 2

def ddpg(agent, num_agents, n_episodes=1000, max_t=1000,
         ckpt_path_prefix=None, ckpt_t=None, logger=None, print_every=100):
    """Train 2 agents to learn how to play tennis using the Deep Deterministic Policy Gradient
    (DDPG) algorithm.
    
    Params
    ======
        agent (Agent): meta agent that learns from 2 sub-agents interacting with the environment.
        num_agents (int): number of sub-agents
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        ckpt_path_prefix (string): prefix of file for saving actor and critic model parameters
        ckpt_t (int): how often (in episodes) to save checkpoint
        logger (SummaryWriter): tensorboard log directory
        print_every (float): how often to print out average score
        
    Returns
    =======
        Average score per 100 episodes corresponding to the max score per episode of the two agents
    """
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        agent_scores = np.zeros(num_agents)
        for _ in range(max_t):
            actions = agent.act(states, i_episode)
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            agent_scores += rewards

            # Save the transitions from all agents in the agent's replay buffer.
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.save(state, action, reward, next_state, done)
                         
            states = next_states
            if np.any(dones):
                break

        # We've decided to wait until after at least one agent has finished an episode.  Now 
        # update the actor/critic networks.
        agent.step(logger, i_episode)

        # Note that we take the MAX of the agents' scores.
        score = np.max(agent_scores)
        scores_deque.append(score)
        scores.append(score)
        if logger is not None:
            logger.add_scalar('agent_0_episode_rewards', agent_scores[0], i_episode)
            logger.add_scalar('agent_1_episode_rewards', agent_scores[1], i_episode)
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
            if logger is not None:
                logger.add_scalar('avg_score', np.mean(scores_deque), i_episode)
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if i_episode % ckpt_t == 0:
            torch.save(agent.actor_local.state_dict(), ckpt_path_prefix + '_actor.pt')
            torch.save(agent.critic_local.state_dict(), ckpt_path_prefix + '_critic.pt')
        if np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            break

    torch.save(agent.actor_local.state_dict(), ckpt_path_prefix + '_actor.pt')
    torch.save(agent.critic_local.state_dict(), ckpt_path_prefix + '_critic.pt')
    return scores

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-tennis_env_path", required=True, help="Path of Unity Tennis environment app")
    parser.add_argument("-ckpt_path_prefix", required=True,
                        help="Prefix of file for saving actor and critic model parameters")
    parser.add_argument('-ckpt_t', type=int, default=20, help='how often to checkpoint model parameters')
    parser.add_argument("-plot_path", default=None, help="File to save plot of score per episode")
    parser.add_argument("-tb_path", default=None, help="Tensorboard log directory")
    
    # Training parameters
    parser.add_argument('-n_episodes', type=int, default=1000, help='maximum number of training episodes')
    parser.add_argument('-max_t', type=int, default=1000, help='maximum number of timesteps per episode')
    parser.add_argument('-batch_size', type=int, default=512, help='minibatch size')
    parser.add_argument('-buffer_size', type=int, default=int(1e6), help='replay buffer size')
    parser.add_argument('-gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('-tau', type=float, default=1e-3, help='for soft update of target parameters')
    parser.add_argument('-actor_lr', type=float, default=1e-3 , help='actor network learning rate')
    parser.add_argument('-crtic_lr', type=float, default=1e-3 , help='critic network learning rate')
    parser.add_argument('-weight_decay', type=float, default=0.0, help='L2 weight decay')
    parser.add_argument('-update_size', type=int, default=20, help='number of updates to networks per agent')
    parser.add_argument('-seed', type=int, default=2, help="Random seed")
    opt = parser.parse_args()
    
    # Load the environment for simulating a tennis game.
    env = UnityEnvironment(file_name=opt.tennis_env_path)
    
    # Get the default brain
    brain_name = env.brain_names[0]
    
    # Set up tensboard to visualize losses and scores over time.
    if opt.tb_path is not None:
        logger = SummaryWriter(log_dir=opt.tb_path)
    else:
        logger = None
    
    # Run the DDPG algorithm
    env_info = env.reset(train_mode=True)[brain_name]
    agent = Agent(num_agents=NUM_AGENTS,
                  state_size=STATE_SIZE,
                  action_size=ACTION_SIZE,
                  lr_actor=opt.actor_lr,
                  lr_critic=opt.crtic_lr,
                  weight_decay=opt.weight_decay,
                  buffer_size=opt.buffer_size,
                  batch_size=opt.batch_size,
                  update_size=opt.update_size,
                  gamma=opt.gamma,
                  tau=opt.tau,
                  random_seed=opt.seed)
    scores = ddpg(agent,
                  num_agents=NUM_AGENTS,
                  n_episodes=opt.n_episodes,
                  max_t=opt.max_t,
                  ckpt_path_prefix=opt.ckpt_path_prefix,
                  ckpt_t=opt.ckpt_t)
    
    # Optionally create and save a plot of score versus episode number.
    if opt.plot_path is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(opt.plot_path)
            
    env.close()
    if logger is not None:
        logger.close()
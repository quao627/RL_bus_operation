


import time
import os
import argparse
import warnings
import pickle
warnings.filterwarnings("ignore")

import sys
import numpy as np
sys.path.append('HybridPPO')

import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from HybridPPO.hybridppo import HybridPPO
from sb3_contrib import RecurrentPPO
from sb3_contrib import MaskablePPO
from MaskedRecurrentPPO.ppo_recurrent import MaskableRecurrentPPO

# from HybridPPO.hybridppo import *
from DiscreteEnv_DR_CL_action2 import Env
# from BusBunchingEnv import Env
action_list=[]

STEP_LIMIT = 40000

class CurriculumCallback(BaseCallback):
    def __init__(self, check_freq: int, action_difficulty_thresholds: list, env, verbose=1):
        super(CurriculumCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.action_difficulty_thresholds = action_difficulty_thresholds
        self.env = env
        self.best_mean_reward = -np.inf
        self.current_num_timesteps = 0
    
    def _on_step(self) -> bool:
        self.current_num_timesteps += 1
        if self.n_calls % self.check_freq == 0:
            # Evaluate policy training performance
            vec_env = self.model.get_env()
            # mean_reward, std_reward = evaluate_policy(self.model, vec_env, n_eval_episodes=1, warn=False)
            mean_reward_sum = 0
            std_reward_sum = 0
            NUM_EVA=3
            for i in range(NUM_EVA):
                mean_reward_i, std_reward_i = evaluate_policy(self.model, vec_env, n_eval_episodes=1, warn=False)
                mean_reward_sum += mean_reward_i
                std_reward_sum += std_reward_i
            mean_reward = mean_reward_sum/NUM_EVA
            std_reward = std_reward_sum/NUM_EVA
            curr_difficulty_level = self.env.difficulty_level
            if self.verbose > 0:
                print(f"N timesteps: {self.num_timesteps} mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

            # Increase difficulty if performance threshold is exceeded
            # TODO: Do we need this or something else 
            # if mean_reward > self.best_mean_reward + self.action_difficulty_thresholds[curr_difficulty_level]:
            #     self.best_mean_reward = mean_reward
            #     self.training_env.envs[0].increase_difficulty()
            if curr_difficulty_level < 2:
                if mean_reward >= self.action_difficulty_thresholds[curr_difficulty_level] or self.current_num_timesteps>=STEP_LIMIT:
                    print('mean_reward: ', mean_reward)
                    print('self.best_mean_reward: ', self.best_mean_reward)
                    print('self.current_num_timesteps: ', self.current_num_timesteps)

                    self.best_mean_reward = mean_reward
                    self.training_env.envs[0].increase_difficulty()
                    self.current_num_timesteps = 0
        return True


def train(args, env, difficulty_level, action_values, action_difficulty_thresholds):
    model_dir = args.model_dir + args.mode
    logdir = args.log_dir

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    model = MaskableRecurrentPPO("MaskableMlpLstmPolicy", 
                    env, 
                    verbose=1, 
                    batch_size=args.batch_size, 
                    tensorboard_log=logdir,
                    learning_rate=args.learning_rate,
                    gamma=args.gamma,
                    n_steps=128,
                    n_epochs=10,
                    )
    callback = CurriculumCallback(check_freq=2000, action_difficulty_thresholds=action_difficulty_thresholds, env=env)

    model.learn(total_timesteps=args.num_steps, tb_log_name=f"ppo_lstm_difficulty_{difficulty_level}", callback=callback)

    model.save(f"ppo_recurrent_difficulty_{difficulty_level}")
    print("....")
    with open('data_DR_CL.pkl', 'wb') as f:
        pickle.dump(env.data, f)
    with open('action_list_DR_CL.pkl', 'wb') as f:
        pickle.dump(action_list, f)

    return model, env


def process_action_masks(action_masks_levels):
    action_masks_levels = action_masks_levels.split()
    action_masks = []

    for level_mask in action_masks_levels:
        level_mask = level_mask.split(',')
        level_mask = [int(action) for action in level_mask]
        action_masks.append(level_mask)
        
    return action_masks 

def setup_env(args, difficulty_level, action_values, action_difficulty_thresholds):
    # Setting up Environment
    print(f"Training on difficulty level {difficulty_level}")
    env = Env(mode=args.mode, difficulty_level=difficulty_level, action_values=action_values, action_difficulty_thresholds=action_difficulty_thresholds)
    return env


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="waiting_time_station", help="waiting_time_total, waiting_time_station, num_pax")
    parser.add_argument("--model_dir", type=str, default="models/PPO", help="model directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="log directory")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_steps", type=int, default=60000, help="number of steps")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--check_freq", type=int, default=2000, help="frequency for checking performance")
    
    # Curriculum Learning Args
    parser.add_argument("--action_difficulty_levels", type=int, default=3, help="levels of difficulty")
    parser.add_argument("--action_mask_levels", type=str, default="0,1,2,3 1,2,3 0,1,2,4", help="levels of difficulty")
    parser.add_argument("--action_difficulty_thresholds", type=str, default="0.5,0.1,0.5", help="levels of difficulty")
     

    args = parser.parse_args()
    
    action_values = process_action_masks(args.action_mask_levels)
    action_difficulty_thresholds = [float(threshold) for threshold in args.action_difficulty_thresholds.split(',')]
    print('action_values: ', action_values)
    print('difficulty_thresholds: ', action_difficulty_thresholds)
    difficulty_level = 0
    env = setup_env(args, difficulty_level, action_values, action_difficulty_thresholds)
    model, env = train(args, env, difficulty_level, action_values, action_difficulty_thresholds)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print("mean_reward: ",mean_reward )
    print("std_reward: ", std_reward)
    # print('Avg passenger waiting time: ', np.mean(env.all_pax_waiting_times))
    # print('Stdev passenger waiting time: ', np.std(env.all_pax_waiting_times))
  
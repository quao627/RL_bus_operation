import time
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
sys.path.append('HybridPPO')


import gym
from gym import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from HybridPPO.hybridppo import HybridPPO
from sb3_contrib import RecurrentPPO

# from HybridPPO.hybridppo import *
from DiscreteEnv_DR_CL import Env
# from BusBunchingEnv import Env
class CurriculumCallback(BaseCallback):
    def __init__(self, check_freq: int, difficulty_increase_thresh: float, verbose=1):
        super(CurriculumCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.difficulty_increase_thresh = difficulty_increase_thresh
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate policy training performance
            mean_reward, std_reward = evaluate_policy(self.model, self.training_env, n_eval_episodes=10)
            if self.verbose > 0:
                print(f"N timesteps: {self.num_timesteps} mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

            # Increase difficulty if performance threshold is exceeded
            if mean_reward > self.best_mean_reward + self.difficulty_increase_thresh:
                self.best_mean_reward = mean_reward
                self.training_env.envs[0].increase_difficulty()

        return True


def train(args):
    
    assert args.holding_only + args.skipping_only + args.turning_only <= 1, "Only one of the three can be true"

    for difficulty_level in range(1, 5):  # Modify this to suit the number of difficulty levels in your curriculum
        print(f"Training on difficulty level {difficulty_level}")

        config = {'holding_only': args.holding_only,
                    'skipping_only': args.skipping_only, 
                    'turning_only': args.turning_only,
                    'mode': args.mode,
                    'difficulty_level': difficulty_level}  
        env = Env(**config)

        model_dir = args.model_dir + args.mode
        logdir = args.log_dir

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        model = RecurrentPPO("MlpLstmPolicy", 
                        env, 
                        verbose=1, 
                        batch_size=args.batch_size, 
                        tensorboard_log=logdir,
                        learning_rate=args.learning_rate,
                        gamma=args.gamma,
                        n_steps=128,
                        n_epochs=10,
                        )
        callback = CurriculumCallback(check_freq=1000, difficulty_increase_thresh=10.0)

        model.learn(total_timesteps=args.num_steps, tb_log_name=f"ppo_lstm_difficulty_{difficulty_level}", callback=callback)

        model.save(f"ppo_recurrent_difficulty_{difficulty_level}")

    return model



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="waiting_time_station", help="waiting_time_total, waiting_time_station, num_pax")
    parser.add_argument("--holding_only", action="store_true", default=False, help="only holding")
    parser.add_argument("--skipping_only", action="store_true", default=False, help="only skipping")
    parser.add_argument("--turning_only", action="store_true", default=False, help="only turning")
    parser.add_argument("--model_dir", type=str, default="models/PPO", help="model directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="log directory")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_steps", type=int, default=300000, help="number of steps")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

    args = parser.parse_args()

    train(args)

    # env = gym.make('Moving-v0')
    # if recording
    # env = gym.wrappers.Monitor(env, "./video", force=True)
    # env.metadata["render.modes"] = ["human", "rgb_array"]
    
    # env.reset()

    # ACTION_SPACE = env.action_space[0].n
    # PARAMETERS_SPACE = env.action_space[1].shape[0]
    # OBSERVATION_SPACE = env.observation_space.shape[0]

    # model = PPO.load(f"model_dir/{mode}")

    # obs = env.reset()
    # while True:
    #     action = (0, 0)
    #     obs, rewards, dones, info = env.step(action)
    #     # if rendering
    #     time.sleep(0.1)

    # time.sleep(1)
    # env.close()

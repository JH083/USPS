#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from infra.logger import Logger
from infra.replay_buffer import ReplayBuffer
import infra.utils as utils

import hydra
from agent.belief import GaussianBelief

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        print(f'config: {self.cfg}')

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.step = 0

        # ---- Minimal PSRL / Belief wiring (uses defaults if not in config) ----
        # Read PSRL/belief params from cfg.agent.params when present; otherwise use safe defaults
        params = getattr(cfg.agent, 'params', cfg.agent)
        self.psrl_gamma = float(getattr(params, 'psrl_gamma', 0.99))
        self.belief_update_every = int(getattr(params, 'belief_update_every', 2000))
        self.belief_batch_size = int(getattr(params, 'belief_batch_size', 1024))

        belief_cfg = getattr(params, 'belief', None)
        if belief_cfg is not None:
            mu0 = np.array(getattr(belief_cfg, 'mu0', [0.0, 0.0]), dtype=float)
            Sigma0_diag = np.array(getattr(belief_cfg, 'Sigma0_diag', [1.0, 1.0]), dtype=float)
            Sigma0 = np.diag(Sigma0_diag)
            sigma2 = float(getattr(belief_cfg, 'sigma2', 1e-3))
            jitter = float(getattr(belief_cfg, 'jitter', 1e-6))
        else:
            # Minimal, safe default: 2D θ prior; adjust via overrides without touching sac.yaml
            mu0 = np.zeros(2, dtype=float)
            Sigma0 = np.eye(2, dtype=float)
            sigma2 = 1e-3
            jitter = 1e-6

        self.belief = GaussianBelief(mu0, Sigma0, sigma2=sigma2, jitter=jitter)
        self._psrl_theta = None
        self._steps_since_belief = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()

            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward

            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)
        return average_episode_reward

    def run(self):
        episode, episode_reward, done = 0, 0, True
        best_eval_reward = 0
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_random_steps))

                self.logger.log('train/episode_reward', episode_reward, self.step)

                # reset for new epoch
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                

                # ---- Continuing-PSRL resample gate (Bernoulli survival) ----
                # Resample θ with probability (1 - psrl_gamma), or on first use
                survive = (np.random.rand() < self.psrl_gamma)
                if (not survive) or (self._psrl_theta is None):
                    self._psrl_theta = self.belief.sample_theta()
                    # Apply parameters to env if supported
                    if hasattr(self.env, 'set_params'):
                        try:
                            self.env.set_params(self._psrl_theta)
                        except Exception:
                            pass


            # sample action for data collection
            if self.step < self.cfg.num_random_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_random_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.cfg.max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

            # ---- Periodic belief update (if env and buffer support it) ----
            self._steps_since_belief += 1
            if self._steps_since_belief >= self.belief_update_every:
                # Only run if env supplies forward model and jacobian, and buffer can sample
                if hasattr(self.env, 'forward_model') and hasattr(self.env, 'jacobian') and hasattr(self.replay_buffer, 'sample'):
                    try:
                        batch = self.replay_buffer.sample(self.belief_batch_size)
                        # Try common batch formats
                        triples = []
                        if isinstance(batch, dict) and all(k in batch for k in ('obs', 'action', 'next_obs')):
                            for o, a, no in zip(batch['obs'], batch['action'], batch['next_obs']):
                                triples.append((o, a, no))
                        elif isinstance(batch, (list, tuple)):
                            # e.g., list of namedtuples with fields obs, action, next_obs
                            for b in batch:
                                o = getattr(b, 'obs', None)
                                a = getattr(b, 'action', None)
                                no = getattr(b, 'next_obs', None)
                                if o is not None and a is not None and no is not None:
                                    triples.append((o, a, no))
                        if len(triples) > 0:
                            self.belief.update(triples,
                                               f=self.env.forward_model,
                                               jacobian=self.env.jacobian,
                                               theta_lin=self.belief.mu)
                    except Exception:
                        # Keep training even if belief update fails early
                        pass
                self._steps_since_belief = 0

            # evaluate agent periodically
            if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                print(f"Evaluating at {self.step}...")
                self.logger.log('eval/episode', episode, self.step)
                eval_reward = self.evaluate()
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.agent.save(self.work_dir)
                print(f"Ending...")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()


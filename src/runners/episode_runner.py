from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import wandb

import copy
import random
import torch as th

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.wolfpack_returns = []
        self.wolfpack_stats = {}
        self.test_wolfpack_returns = []
        self.test_wolfpack_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, qdifference_transformer, planning_transformer):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.qdiff_transformer = qdifference_transformer
        self.planning_transformer = planning_transformer

    def setup_mac_for_attack(self, mac):
        self.mac_for_attack = copy.deepcopy(mac)
        self.mac_for_attack.cuda()
    
    def setup_learner(self, learner):
        self.learner = learner

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
                wandb.log({"epsilon": self.mac.action_selector.epsilon}, step=self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def followup_agent_group_selection_l2(self, batch, t, initial_agent, agents_set):
        obs = batch["obs"][:, t]
        initial_agent_obs = obs[:, initial_agent]

        obs_l2_distances = [(i, th.sum((initial_agent_obs - obs[:, i]) ** 2)) for i in agents_set]

        obs_l2_distances.sort(key=lambda x: x[1])
        followup_agents = [obs_l2_distances[i][0] for i in range(self.args.num_followup_agents)]

        return followup_agents

    def followup_agent_group_selection(self, batch, t, initial_agent, agents_set):
        actions = batch["actions"][:, t]
        state = batch["state"][:, t]
        attacker_actions = batch["attacker_actions"][:, t]
        hidden_states = self.mac.return_hidden()

        self.mac_for_attack.agent.load_state_dict(copy.deepcopy(self.learner.mac.agent.state_dict()))
        normal_mac, attack_mac = copy.deepcopy(self.mac_for_attack), copy.deepcopy(self.mac_for_attack)
        mixer = copy.deepcopy(self.learner.mixer)
        optimizer = th.optim.Adam(list(attack_mac.parameters()), lr=self.args.lr)

        indi_attack_q_first = attack_mac.forward_q_attack(batch, t, hidden_states.detach()).detach()
        indi_normal_q = normal_mac.forward_q_attack(batch, t, hidden_states.detach())
        indi_attack_q = attack_mac.forward_q_attack(batch, t, hidden_states.detach())

        normal_q = th.gather(indi_normal_q, dim=2, index=actions).squeeze(2)    # 1,  

        do_actions = actions.clone().detach()
        for agent in initial_agent:
            do_actions[:, agent] = copy.deepcopy(attacker_actions[:, agent].detach())
        attack_q = th.gather(indi_attack_q, dim=2, index=do_actions).squeeze(2)

        if self.args.mixer == "vdn":
            normal_q = normal_q.unsqueeze(0)
            attack_q = attack_q.unsqueeze(0)
            total_normal_q = mixer(normal_q, state)
            total_attack_q = mixer(attack_q, state)
        elif self.args.mixer == "dmaq":
            total_normal_q = mixer(normal_q, state, is_v=True)
            total_attack_q = mixer(attack_q, state, is_v=True)
        else:
            total_normal_q = mixer(normal_q, state)
            total_attack_q = mixer(attack_q, state)

        loss = ((total_attack_q - total_normal_q.detach()) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        updated_attack_q = attack_mac.forward_q_attack(batch, t, hidden_states.detach())

        def normalize_q(q_values):
            q_mean = q_values.mean(dim=2, keepdim=True)
            q_std = q_values.std(dim=2, keepdim=True)
            normalized = (q_values - q_mean) / q_std
            return normalized / normalized.std(dim=2, keepdim=True)

        indi_attack_q_first_norm = th.nn.functional.softmax(normalize_q(indi_attack_q_first), dim=-1)
        updated_attack_q_norm = th.nn.functional.softmax(normalize_q(updated_attack_q), dim=-1)

        kl = [
            (i, th.sum(indi_attack_q_first_norm[:, i] * (th.log(indi_attack_q_first_norm[:, i]) - th.log(updated_attack_q_norm[:, i])), dim=-1).mean())
            for i in agents_set
        ]
        kl.sort(key=lambda x: x[1], reverse=True)

        followup_agents = [kl[i][0] for i in range(self.args.num_followup_agents)]
        
        return followup_agents

    def get_attack_step(self, pre_initial_attack_step):
        state = self.batch["state"]

        agent_id = th.eye(self.args.n_agents, device=self.batch.device).unsqueeze(0).expand(1, -1, -1)

        if self.t < self.args.trans_input_len:
            B, T, F = state.size()
            time_step_temp = th.arange(start=0, end=self.args.trans_input_len, step=1, device=state.device).reshape(1, self.args.trans_input_len)

            now_state = th.zeros(1, self.args.trans_input_len, F, device=state.device)
            time_step = th.zeros(1, self.args.trans_input_len, device=state.device)

            now_state[:, self.args.trans_input_len - (self.t+1):, :] = state[:, :self.t + 1, :]
            time_step[:, self.args.trans_input_len - (self.t+1):] = time_step_temp[:, :self.t + 1]
        else:
            now_state = state[:, self.t-self.args.trans_input_len+1:self.t+1]
            time_step = th.arange(start=self.t, end=self.t+self.args.trans_input_len, step=1, device=state.device).reshape(1, self.args.trans_input_len)

        agent_id = agent_id.unsqueeze(1).repeat(1, now_state.shape[1], 1, 1)
        predict_q_diff_list = []
        for i in range(self.args.n_agents):
            predict_q_diff = self.qdiff_transformer(time_step, now_state, agent_id[:, :, i])   
            predict_q_diff_list.append(predict_q_diff[:, -1])
        predict_q_diff_total = th.stack(predict_q_diff_list, dim=1)
        predict_q_diff, initial_agent = th.max(predict_q_diff_total, dim=1)
        predict_q_diff = predict_q_diff.squeeze(0).detach()  # [20]

        if self.t == 0:
            predict_q_diff = predict_q_diff[:]  
        else:
            predict_q_diff = predict_q_diff[:-(min(self.t-pre_initial_attack_step, self.args.attack_period-1))]
            padding_size = self.args.attack_period - predict_q_diff.shape[0]
            if padding_size > 0:
                predict_q_diff = th.nn.functional.pad(predict_q_diff, (0, padding_size), mode='constant', value=-9999999)

        predict_q_diff_softmax = th.softmax(predict_q_diff / self.args.temperature, dim=-1)
        critical_step = th.multinomial(predict_q_diff_softmax, num_samples=1)

        if critical_step == 0:
            attack_prob = 1
            initial_agent = initial_agent.squeeze(0)[0]
        else:
            attack_prob = 0
            initial_agent = 0

        return attack_prob, initial_agent

    def run_wolfpack_attacker(self, test_mode=False):
        self.reset()
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        if test_mode==True:
            attack_num = self.args.num_attack_test
        else:
            attack_num = self.args.num_attack_train

        attack_cnt = 0
        do_attack_num = 0
        initial_attack_flag = copy.deepcopy(self.args.attack_duration)
        
        pre_initial_attack_step = 0
        
        initial_agent = [self.args.n_agents]
        followup_agents = [self.args.n_agents, self.args.n_agents]

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            ori_actions, chosen_actions, random_actions, attacker_actions = self.mac.select_actions_wolfpack(self.batch,
                                                                            t_ep=self.t, t_env=self.t_env,
                                                                            test_mode=test_mode)

            hidden_states = self.mac.return_hidden()

            middle_transition_data = {
                "actions": ori_actions.to("cpu").numpy(),
                "attacker_actions": attacker_actions.to("cpu").numpy(),
                "hidden_states": hidden_states
            }
  
            self.batch.update(middle_transition_data, ts=self.t)

            if self.args.init_attack_step_method == "planner":
                attack_prob, chosen_agent = self.get_attack_step(pre_initial_attack_step)

                if self.args.init_agent_random == True:
                    initial_agent = random.sample(range(self.args.n_agents), 1)
                else:
                    initial_agent = [chosen_agent]

            do_actions = copy.deepcopy(ori_actions)
            prob = 1 / 10

            if initial_attack_flag == self.args.attack_duration:
                if self.args.init_attack_step_method == "random":
                    if random.random() < prob:
                        initial_agent = random.sample(range(self.args.n_agents), 1)
                        previous_initial_agent = copy.deepcopy(initial_agent)
                        for i in previous_initial_agent:
                            do_actions[:, i] = copy.deepcopy(attacker_actions[:, i])
                    
                    if not ori_actions.equal(do_actions):
                        pre_initial_attack_step = self.t
                        attack_cnt += 1
                        initial_attack_flag = initial_attack_flag - 1

                elif self.args.init_attack_step_method == "planner":
                    if attack_prob == 1:
                        previous_initial_agent = copy.deepcopy(initial_agent)
                        for i in previous_initial_agent:
                            do_actions[:, i] = copy.deepcopy(attacker_actions[:, i])
                    
                    if not ori_actions.equal(do_actions):
                        pre_initial_attack_step = self.t
                        attack_cnt += 1
                        initial_attack_flag = initial_attack_flag - 1
            else:  
                for i in followup_agents_eval:
                    do_actions[:, i] = copy.deepcopy(attacker_actions[:, i])

                if initial_attack_flag <= 0:
                    initial_attack_flag = copy.deepcopy(self.args.attack_duration)
                else:
                    initial_attack_flag = initial_attack_flag - 1

                attack_cnt += 1

            if attack_cnt > attack_num:
                do_actions = ori_actions

            if not ori_actions.equal(do_actions):
                do_attack_num += 1
                
            reward, terminated, env_info = self.env.step(do_actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": ori_actions.to("cpu").numpy(),
                "forced_actions": do_actions.to("cpu").numpy(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            
            self.batch.update(post_transition_data, ts=self.t)

            all_agents = set(range(self.args.n_agents))
            if initial_attack_flag == self.args.attack_duration - 1:
                excluded_agents = set(agent for agent in previous_initial_agent)
            else:
                excluded_agents = set(agent for agent in initial_agent)
            agents_set = all_agents - excluded_agents
            
            if self.args.followup_l2 == True:
                followup_agents = self.followup_agent_group_selection_l2(self.batch, self.t, initial_agent, agents_set)
            elif self.args.followup_l2 == False:
                followup_agents = self.followup_agent_group_selection(self.batch, self.t, initial_agent, agents_set)
            if initial_attack_flag == self.args.attack_duration - 1:
                followup_agents_eval = copy.deepcopy(followup_agents)
            last_transition_data = {
                "initial_agent": [initial_agent],
                "followup_agents": [followup_agents],
            }
            self.batch.update(last_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        ori_actions, chosen_actions, random_actions, attacker_actions = self.mac.select_actions_wolfpack(self.batch,
                                                                            t_ep=self.t, t_env=self.t_env,
                                                                            test_mode=test_mode)

        self.batch.update({"actions": ori_actions.to("cpu").numpy()}, ts=self.t)
        self.batch.update({"forced_actions": ori_actions.to("cpu").numpy()}, ts=self.t)

        cur_stats = self.test_wolfpack_stats if test_mode else self.wolfpack_stats
        cur_returns = self.test_wolfpack_returns if test_mode else self.wolfpack_returns
        log_prefix = "test_Wolfpack_" if test_mode else "Wolfpack_"

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            
        cur_returns.append(episode_return)

        battle_won = env_info.get("battle_won", 0)

        self._log(cur_returns, cur_stats, log_prefix)
        
        if test_mode:
            wandb.log({"test wolfpack attack num": do_attack_num}, step=self.t_env)
        else:
            wandb.log({"wolfpack attack num": do_attack_num}, step=self.t_env)

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        wandb.log({prefix + "return_mean": np.mean(returns)}, step=self.t_env)
        wandb.log({prefix + "return_std": np.std(returns)}, step=self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
                wandb.log({prefix + k + "_mean": v/stats["n_episodes"]}, step=self.t_env)
        stats.clear()

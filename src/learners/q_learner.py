import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
import wandb
from torch import optim
import random
import torch.nn.functional as F
import torch.nn as nn
import torch

class QLearner:
    def __init__(self, mac, scheme, logger, args, qdifference_transformer, planning_transformer):
        self.args = args
        self.mac = mac

        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.qdiff_transformer = qdifference_transformer
        self.optimizer_qdiff = optim.Adam(self.qdiff_transformer.parameters(), lr=args.lr)

        self.planning_transformer = planning_transformer
        self.optimizer_planning = optim.Adam(self.planning_transformer.parameters(), lr=args.lr)

    def sliding_windows(self, data, sis_list, window_size=20):
        X = []  
        for i in range(data.shape[0]):
            si = sis_list[i]
            X.append(data[i, si:si + window_size])
        X_tensor = th.stack(X, dim=0)
        return X_tensor
    
    def sliding_window_with_padding(self, tensor, window_size=20):
        if len(tensor.shape) == 3:
            B, T, F = tensor.size()
            result = th.zeros(B, T, window_size, F, device=tensor.device)
        
            for t in range(T):
                if t + 1 >= window_size:
                    result[:, t, :, :] = tensor[:, t - window_size + 1:t + 1, :]
                else:
                    result[:, t, window_size - (t+1):, :] = tensor[:, :t + 1, :]
        else:
            B, T, agent, F = tensor.size()
            result = th.zeros(B, T, window_size, agent, F, device=tensor.device)
        
            for t in range(T):
                if t + 1 >= window_size:
                    result[:, t, :, :, :] = tensor[:, t - window_size + 1:t + 1, :, :]
                else:
                    result[:, t, window_size - (t+1):, :, :] = tensor[:, :t + 1, :, :]

        return result

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        ######################################################
        # Planning Transformer training
        bs = batch["state"].shape[0]
        state = batch["state"][:, :-1].reshape(bs, -1, self.args.state_shape) 
        actions = batch["forced_actions_onehot"][:, :-1].reshape(bs, -1, self.args.n_agents, self.args.n_actions)

        next_state = batch["state"][:, 1:].reshape(bs, -1, self.args.state_shape)
        next_obs = batch["obs"][:, 1:].reshape(bs, -1, self.args.n_agents, self.args.obs_shape)
        
        hidden_states = batch["hidden_states"][:, :-1].reshape(bs, -1, self.args.n_agents, self.args.rnn_hidden_dim)

        mask_trans = batch["filled"][:, :-1].float()
        terminated_trans = batch["terminated"][:, :-1].float()
        mask_trans[:, 1:] = mask_trans[:, 1:] * (1 - terminated_trans[:, :-1])
        avail_actions = batch["avail_actions"]

        initial_agent = batch["initial_agent"][:, :-1]
        followup_agents = batch["followup_agents"][:, :-1]

        sis_list = []
        for _ in range(bs): 
            si = random.randint(0, state.shape[1] - self.args.trans_input_len)
            sis_list.append(th.tensor(si).cuda())

        state_slided = self.sliding_windows(state, sis_list, window_size=self.args.trans_input_len)    
        hidden_states_slided = self.sliding_windows(hidden_states, sis_list, window_size=self.args.trans_input_len)
        actions_slided = self.sliding_windows(actions, sis_list, window_size=self.args.trans_input_len)
        next_state_slided = self.sliding_windows(next_state, sis_list, window_size=self.args.trans_input_len)
        next_obs_slided = self.sliding_windows(next_obs, sis_list, window_size=self.args.trans_input_len)

        predict_state, predict_obs_embed = self.planning_transformer(state_slided, hidden_states_slided, actions_slided)  # [bs, 20, 1]

        agent_id = th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, state_slided.shape[1], -1, -1)

        predict_obs_list = []
        for i in range(self.args.n_agents):
            predict_obs_agent = self.planning_transformer.get_next_obs(predict_obs_embed, agent_id[:, :, i]) 
            predict_obs_list.append(predict_obs_agent)
        predict_obs = th.stack(predict_obs_list, dim=2) # [bs, 20, agent, obs]

        error_state = predict_state - next_state_slided.detach()
        error_obs = predict_obs - next_obs_slided.detach()

        mask_trans_slided = self.sliding_windows(mask_trans, sis_list, window_size=self.args.trans_input_len)

        mask_state = mask_trans_slided.expand(-1, -1, error_state.size(2))
        mask_obs = mask_trans_slided.unsqueeze(2).expand(-1, -1, error_obs.size(2), error_obs.size(3))

        masked_error_state = error_state * mask_state
        masked_error_obs = error_obs * mask_obs

        loss_state = (masked_error_state ** 2).sum() / mask_state.sum()
        loss_obs = (masked_error_obs ** 2).sum() / mask_obs.sum()

        loss_planning = loss_state + loss_obs

        self.optimizer_planning.zero_grad()
        loss_planning.backward()
        self.optimizer_planning.step()

        ######################################################
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        mac_out_temp = mac_out.clone().detach()

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        indi_q = chosen_action_qvals.clone().detach()

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        
        total_q_qmix_normal = chosen_action_qvals.clone().detach() # 32, 120, 1

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask.sum()

        ######################################################
        # Qdifference transformer training
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
        attacker_actions = batch["attacker_actions"][:, :-1]
        attacker_actions_onehot = batch["attacker_actions_onehot"][:, :-1]
        state = batch["state"][:, :-1]
        hidden_states = batch["hidden_states"][:, :-1]
        avail_actions = batch["avail_actions"]

        mac_out_detach = mac_out_temp.clone().detach()
        chosen_action_qvals_attack = th.gather(mac_out_detach[:, :-1], dim=3, index=attacker_actions).squeeze(3)  
        
        normal_total_q = th.zeros_like(total_q_qmix_normal)
        for t in range(total_q_qmix_normal.shape[1]):
            end_idx = min(t + self.args.attack_duration, total_q_qmix_normal.shape[1])  
            normal_total_q[:, t] = th.sum(total_q_qmix_normal[:, t:end_idx+1], dim=1)
        normal_total_q = self.sliding_windows(normal_total_q, sis_list, window_size=self.args.trans_input_len)
        attacked_total_q = th.zeros_like(total_q_qmix_normal).unsqueeze(2).repeat(1, 1, self.args.n_agents, 1)    
        attacked_total_q = self.sliding_windows(attacked_total_q, sis_list, window_size=self.args.trans_input_len)

        time_step = th.arange(start=0, end=state.shape[1], step=1, device=state.device).reshape(1, state.shape[1], 1).repeat(bs, 1, 1) 
        now_time_step = self.sliding_window_with_padding(time_step)
        now_time_step = self.sliding_windows(now_time_step, sis_list, window_size=self.args.trans_input_len)

        with th.no_grad():
            now_state = self.sliding_window_with_padding(state)
            now_hidden_states = self.sliding_window_with_padding(hidden_states)
            now_actions = self.sliding_window_with_padding(actions_onehot)

            now_state = self.sliding_windows(now_state, sis_list, window_size=self.args.trans_input_len)
            state_temp = copy.deepcopy(now_state)
            now_hidden_states = self.sliding_windows(now_hidden_states, sis_list, window_size=self.args.trans_input_len)
            now_actions = self.sliding_windows(now_actions, sis_list, window_size=self.args.trans_input_len)

            now_actions_onehot = self.sliding_windows(actions_onehot, sis_list, window_size=self.args.trans_input_len)
            now_attacker_actions_onehot = self.sliding_windows(attacker_actions_onehot, sis_list, window_size=self.args.trans_input_len)
            now_avail_actions = self.sliding_windows(avail_actions, sis_list, window_size=self.args.trans_input_len)
            initial_agent_t = self.sliding_windows(initial_agent, sis_list, window_size=self.args.trans_input_len)
            followup_agents_t = self.sliding_windows(followup_agents, sis_list, window_size=self.args.trans_input_len)  

            indi_q_slided = self.sliding_windows(indi_q.clone().detach(), sis_list, window_size=self.args.trans_input_len)

            batch_indices = th.arange(indi_q_slided.size(0)).unsqueeze(1).expand(-1, indi_q_slided.size(1))  # (32, 120)
            time_indices = th.arange(indi_q_slided.size(1)).unsqueeze(0).expand(indi_q_slided.size(0), -1)  # (32, 120)

            initial_agent_t = initial_agent_t.squeeze(-1)   
            chosen_values = chosen_action_qvals_attack[batch_indices, time_indices, initial_agent_t]  # (32, 120)
            indi_q_slided[batch_indices, time_indices, initial_agent_t] = chosen_values

            total_q_qmix_attack_temp = self.mixer(indi_q_slided, state_slided)

            attacked_total_q[batch_indices, time_indices, initial_agent_t] += total_q_qmix_attack_temp
            do_actions = copy.deepcopy(now_actions_onehot)   
            attacker_actions_onehot_t = copy.deepcopy(now_attacker_actions_onehot) 

            chosen_attacker_actions_onehot_t = attacker_actions_onehot_t[batch_indices, time_indices, initial_agent_t]
            do_actions[batch_indices, time_indices, initial_agent_t] = chosen_attacker_actions_onehot_t.float()

            now_actions = now_actions[:, :, :-1]   
            now_actions = th.cat((now_actions, do_actions.unsqueeze(2)), dim=2)

            agent_id = th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).unsqueeze(1).expand(bs, self.args.trans_input_len, -1, -1) 
            for _ in range(self.args.attack_duration):
                now_state = now_state.reshape(-1, self.args.trans_input_len, state.shape[-1])
                now_hidden_states = now_hidden_states.reshape(-1, self.args.trans_input_len, self.args.n_agents, hidden_states.shape[-1])
                now_actions = now_actions.reshape(-1, self.args.trans_input_len, self.args.n_agents, now_actions.shape[-1])

                predict_state, predict_obs_embed = self.planning_transformer(now_state, now_hidden_states, now_actions)  

                now_state = now_state.reshape(bs, self.args.trans_input_len, self.args.trans_input_len, -1)
                now_hidden_states = now_hidden_states.reshape(bs, self.args.trans_input_len, self.args.trans_input_len, self.args.n_agents, -1)
                now_actions = now_actions.reshape(bs, self.args.trans_input_len, self.args.trans_input_len, self.args.n_agents, -1)
                predict_state = predict_state.reshape(bs, self.args.trans_input_len, self.args.trans_input_len, -1)
                predict_obs_embed = predict_obs_embed.reshape(bs, self.args.trans_input_len, self.args.trans_input_len, -1)

                predict_state = predict_state[:, :, -1] 
                predict_obs_embed = predict_obs_embed[:, :, -1]

                obs_list = []
                for i in range(self.args.n_agents):
                    predict_obs_agent = self.planning_transformer.get_next_obs(predict_obs_embed, agent_id[:, :, i]) 
                    obs_list.append(predict_obs_agent)
                predict_obs = th.stack(obs_list, dim=2) 

                predict_indi_q = self.mac.forward_action(batch, predict_obs, now_actions[:, :, -1], now_hidden_states[:, :, -1].detach())

                predict_indi_q_action = predict_indi_q.clone().detach()
                predict_indi_q_action[now_avail_actions == 0] = -9999999
                predict_actions = predict_indi_q_action.max(dim=3, keepdim=True)[1]

                predict_indi_q_ = predict_indi_q.clone().detach()
                predict_chosen_q = th.gather(predict_indi_q_, dim=3, index=predict_actions).squeeze(-1) 

                predict_indi_q_attack_action = predict_indi_q.clone().detach()
                predict_indi_q_attack_action[now_avail_actions == 0] = +9999999
                predict_attacker_actions = predict_indi_q_attack_action.min(dim=3, keepdim=True)[1]

                predict_indi_q_attack = predict_indi_q.clone().detach()
                predict_chosen_attacker_q = th.gather(predict_indi_q_attack, dim=3, index=predict_attacker_actions).squeeze(-1) 

                do_actions = copy.deepcopy(predict_actions).squeeze(-1)

                predict_attacker_actions = predict_attacker_actions.squeeze(-1)

                indi_q_slided = predict_chosen_q.clone().detach() # 32, 20, 5

                for agent in range(self.args.num_followup_agents):
                    chosen_predict_chosen_attacker_q = predict_chosen_attacker_q[batch_indices, time_indices, followup_agents_t[:, :, agent]]

                    indi_q_slided[batch_indices, time_indices, followup_agents_t[:, :, agent]] = chosen_predict_chosen_attacker_q

                    chosen_predict_attacker_actions = predict_attacker_actions[batch_indices, time_indices, followup_agents_t[:, :, agent]]

                    do_actions[batch_indices, time_indices, followup_agents_t[:, :, agent]] = chosen_predict_attacker_actions

                total_q_qmix_attack_temp = self.mixer(indi_q_slided, predict_state)
                attacked_total_q[batch_indices, time_indices, initial_agent_t] += total_q_qmix_attack_temp

                do_actions = F.one_hot(do_actions, num_classes=self.args.n_actions).squeeze(2).float()

                predict_state = predict_state.unsqueeze(2)
                do_actions = do_actions.unsqueeze(2)

                now_state = th.cat((now_state, predict_state), dim=2)
                now_actions = th.cat((now_actions, do_actions), dim=2)

                now_state = now_state[:, :, 1:]
                now_actions = now_actions[:, :, 1:]

        attacked_total_q_agent = attacked_total_q[batch_indices, time_indices, initial_agent_t] 
        current_q_diff = (normal_total_q - attacked_total_q_agent).detach()
        
        state_temp = state_temp.reshape(-1, self.args.trans_input_len, state_temp.shape[-1]) 
        now_time_step = now_time_step.reshape(-1, self.args.trans_input_len) 
        agent_id = th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(state_temp.shape[0], self.args.trans_input_len, -1, -1)    
        predict_q_diff_list = []

        for selected_agent in range(self.args.n_agents):
            predict_q_diff = self.qdiff_transformer(now_time_step, state_temp, agent_id[:, :, selected_agent])
            predict_q_diff = predict_q_diff.reshape(bs, -1, self.args.trans_input_len, predict_q_diff.shape[-1])[:, :, -1]
            predict_q_diff_list.append(predict_q_diff)
        predict_q_diff_tensor = th.stack(predict_q_diff_list, dim=2)   
        
        predict_q_diff_agent = predict_q_diff_tensor[batch_indices, time_indices, initial_agent_t]

        now_mask_slided = self.sliding_windows(mask_trans, sis_list, window_size=self.args.trans_input_len) 
        mask_qdiff = now_mask_slided.expand(-1, -1, self.args.trans_input_len)
        current_q_diff = current_q_diff * now_mask_slided
        predict_q_diff_agent = predict_q_diff_agent * mask_qdiff

        current_q_diff = current_q_diff.squeeze(-1)
        
        error_qdiff = 0
        
        for i in range(self.args.trans_input_len):
            error_qdiff += (((predict_q_diff_agent[:, i, :self.args.trans_input_len-i] - current_q_diff[:, i:].detach()) ** 2).sum() / (self.args.trans_input_len - i))  
        loss_qdiff = error_qdiff / now_mask_slided.sum()

        self.optimizer_qdiff.zero_grad()
        loss_qdiff.backward()
        self.optimizer_qdiff.step()
        
        ######################################################
        # Optimise 
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        ######################################################

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            wandb.log({"loss_td": loss.item()}, step=t_env)
            wandb.log({"grad_norm": grad_norm}, step=t_env)

            wandb.log({"td_error_abs": (masked_td_error.abs().sum().item()/mask_elems)}, step=t_env)
            wandb.log({"q_taken_mean": (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents)}, step=t_env)
            wandb.log({"target_mean": (targets * mask).sum().item()/(mask_elems * self.args.n_agents)}, step=t_env)

            wandb.log({"loss_qdiff": loss_qdiff.item()}, step=t_env)

            wandb.log({"loss_state": loss_state.item()}, step=t_env)
            wandb.log({"loss_obs": loss_obs.item()}, step=t_env)
            wandb.log({"loss_planning": loss_planning.item()}, step=t_env)

            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.qdiff_transformer.cuda()
        self.planning_transformer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

        th.save(self.qdiff_transformer.state_dict(), "{}/qdiff_transformer.th".format(path))
        th.save(self.optimizer_qdiff.state_dict(), "{}/optimizer_qdiff.th".format(path))

        th.save(self.planning_transformer.state_dict(), "{}/planning_transformer.th".format(path))
        th.save(self.optimizer_planning.state_dict(), "{}/optimizer_planning.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

        # self.planning_transformer.load_state_dict(th.load("{}/planning_transformer.th".format(path), map_location=lambda storage, loc: storage))
        # self.optimizer_planning.load_state_dict(th.load("{}/optimizer_planning.th".format(path), map_location=lambda storage, loc: storage))

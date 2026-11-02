#!/usr/bin/env python3
"""Read-only SMAC diagnostics against a frozen sweep checkpoint.

Run with the sweep's Python environment. CUDA is explicitly disabled. A task
JSON selects the immutable source snapshot, Sacred config and checkpoint.
Metrics never update model weights. Vanilla uses the WALL scheduler as a
TRANSFER attack; it must not be described as a vanilla-trained attacker.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    cli = parser.parse_args()
    task = json.loads(cli.task.read_text())
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['SC2PATH'] = task['sc2path']
    sys.path.insert(0, str(Path(task['source']) / 'src'))
    import numpy as np
    import torch as th
    import torch.nn.functional as F
    from components.episode_buffer import EpisodeBatch
    from components.transforms import OneHot
    from components.wolfpack import joint_q
    from controllers.basic_controller import BasicMAC
    from learners.q_learner import QLearner
    from modules.step.models import Planning_Transformer, Qdifference_Transformer
    from runners import episode_runner as runner_module
    from run import args_sanity_check, seed_everything
    from utils.sequence import gather_history

    th.set_num_threads(1)
    config = json.loads(Path(task['config']).read_text())
    config.update(use_cuda=False, use_amp=False, compile_models=False, cpu_threads=1)
    args = NS(**args_sanity_check(config, logging.getLogger('audit')))
    args.device = 'cpu'
    args.test_nepisode = task.get('episodes', 32)
    if not isinstance(args.test_nepisode, int) or args.test_nepisode < 1:
        raise ValueError('episodes must be a positive integer')
    args.env_args['seed'] = task.get('env_seed', 910001)
    original_select = runner_module.select_followup

    def random_followup(mixer, args, q, state, actions, attacked, avail, initial, obs=None):
        legal = avail.bool().sum(-1) > 1
        legal.scatter_(1, initial.reshape(-1, 1), False)
        scores = th.rand(legal.shape).masked_fill(~legal, -th.inf)
        chosen = scores.argsort(-1, descending=True)[:, :args.num_followup_agents]
        return chosen.masked_fill(~legal.gather(1, chosen), -1)

    class DiagnosticRunner(runner_module.EpisodeRunner):
        mode = 'wolfpack'
        forced_time = None
        forced_agent = 0

        def reset(self):
            super().reset()
            self.hazards = []

        def _record(self, episode_return, env_info, test_mode, attacked):
            self.record = dict(return_value=float(episode_return), length=self.t,
                               win=bool(env_info.get('battle_won', False)),
                               changed_steps=int(env_info.get('attack_changed_steps', 0)),
                               bursts=int(env_info.get('attack_bursts', 0)))

        @th.no_grad()
        def get_attack_step(self, pre_initial_attack_step=None):
            start = max(0, self.t - args.trans_input_len + 1)
            states = self.batch['state'][:, start:self.t + 1]
            times = th.arange(start, self.t + 1)[None]
            scores = self.qdiff_transformer(times, states, last_only=True)[0, -1].float()
            prob = th.softmax(scores / args.temperature, -1)
            self.hazards.append(dict(t=self.t, probability=float(prob[0]),
                                     entropy=float(-(prob * prob.clamp_min(1e-30).log()).sum()),
                                     score_min=float(scores.min()), score_max=float(scores.max())))
            if self.forced_time is not None:
                return self.t == self.forced_time, self.forced_agent
            p = 1.0 / args.attack_period if self.mode == 'uniform' else float(prob[0])
            decide = bool(th.rand(()) < p)
            return decide, int(th.randint(args.n_agents, ())) if decide else -1

    class NullLogger:
        def log_stat(self, *args, **kwargs):
            pass

    seed_everything(123456)
    runner = DiagnosticRunner(args, NullLogger())
    info = runner.get_env_info()
    for key in ('n_agents', 'n_actions', 'state_shape', 'obs_shape', 'episode_limit'):
        setattr(args, key, info[key])
    scheme = {'state': {'vshape': args.state_shape},
              'obs': {'vshape': args.obs_shape, 'group': 'agents'},
              'initial_agent': {'vshape': 1, 'dtype': th.long},
              'followup_agents': {'vshape': args.num_followup_agents, 'dtype': th.long},
              'actions': {'vshape': (1,), 'group': 'agents', 'dtype': th.long},
              'forced_actions': {'vshape': (1,), 'group': 'agents', 'dtype': th.long},
              'avail_actions': {'vshape': (args.n_actions,), 'group': 'agents', 'dtype': th.int},
              'reward': {'vshape': (1,)}, 'terminated': {'vshape': (1,), 'dtype': th.uint8}}
    groups = {'agents': args.n_agents}
    preprocess = {key: (key + '_onehot', [OneHot(args.n_actions)])
                  for key in ('actions', 'forced_actions')}
    template = EpisodeBatch(scheme, groups, 1, args.episode_limit + 1, preprocess=preprocess)
    mac = BasicMAC(template.scheme, groups, args)
    planner, qdiff = Planning_Transformer(args), Qdifference_Transformer(args)
    learner = QLearner(mac, template.scheme, NullLogger(), args, qdiff, planner)
    checkpoint_hash = digest(task['checkpoint'])
    cp = th.load(task['checkpoint'], map_location='cpu', weights_only=False)
    if cp['fingerprint'] != learner._fingerprint():
        raise ValueError('Checkpoint/config fingerprint mismatch')
    mac.agent.load_state_dict(cp['agent'])
    learner.mixer.load_state_dict(cp['mixer'])
    planner.load_state_dict(cp['planning'])
    qdiff.load_state_dict(cp['qdiff'])
    learner.wall_updates = cp['wall_updates']
    scheduler_hash = None
    if task.get('scheduler_checkpoint'):
        scheduler_hash = digest(task['scheduler_checkpoint'])
        alternate = th.load(task['scheduler_checkpoint'], map_location='cpu', weights_only=False)
        if alternate['fingerprint'] != cp['fingerprint']:
            raise ValueError('Cross-seed scheduler architecture mismatch')
        planner.load_state_dict(alternate['planning'])
        qdiff.load_state_dict(alternate['qdiff'])
    if task.get('vanilla'):
        mac.load_models(task['vanilla'])
        learner.mixer.load_state_dict(th.load(str(Path(task['vanilla']) / 'mixer.th'),
                                             map_location='cpu', weights_only=True))
    for module in (mac.agent, learner.mixer, planner, qdiff):
        module.eval()
    args.init_attack_step_method = 'planner'
    args.planner_warmup_updates = 0

    def setup(r):
        r.setup(scheme, groups, preprocess, mac, qdiff, planner)
        r.setup_learner(learner)
        return r

    setup(runner)

    @th.no_grad()
    def predict_metrics(batch, length):
        records = []
        endpoints = th.arange(0, length, 3)[None]
        sh, valid, times = gather_history(batch['state'], endpoints, args.trans_input_len)
        oh = gather_history(batch['obs'], endpoints, args.trans_input_len)[0]
        ah = gather_history(batch['forced_actions_onehot'], endpoints, args.trans_input_len)[0]
        sh, oh, ah, valid, times = [x.flatten(0, 1) for x in (sh, oh, ah, valid, times)]
        original_state, original_obs = sh[:, -1].clone(), oh[:, -1].clone()
        for horizon in range(1, 4):
            ps, po, pa, pt = planner(sh, oh, ah, times, valid, return_aux=True)
            ps, po, pa, pt = ps[:, -1], po[:, -1], pa[:, -1], pt[:, -1, 0]
            idx = endpoints[0] + horizon
            active = idx <= length
            safe = idx.clamp_max(length)
            ts, to = batch['state'][0, safe], batch['obs'][0, safe]
            ta = batch['avail_actions'][0, safe].bool()
            for j in active.nonzero().flatten().tolist():
                pred_avail = pa[j] >= 0
                physical = args.state_shape - (args.n_agents * args.n_actions
                            if args.env_args.get("state_last_action", True) else 0)
                records.append(dict(horizon=horizon, t=int(endpoints[0, j]),
                    state_mse=float((ps[j] - ts[j]).square().mean()),
                    state_persistence_mse=float((original_state[j] - ts[j]).square().mean()),
                    physical_state_mse=float((ps[j, :physical] - ts[j, :physical]).square().mean()),
                    physical_state_persistence_mse=float((original_state[j, :physical] - ts[j, :physical]).square().mean()),
                    obs_mse=float((po[j] - to[j]).square().mean()),
                    obs_persistence_mse=float((original_obs[j] - to[j]).square().mean()),
                    avail_tp=int((pred_avail & ta[j]).sum()),
                    avail_fp=int((pred_avail & ~ta[j]).sum()),
                    avail_fn=int((~pred_avail & ta[j]).sum()),
                    avail_exact=float((pred_avail == ta[j]).all(-1).float().mean()),
                    term_brier=float((pt[j].sigmoid() - batch['terminated'][0, safe[j]-1, 0].float()).square())))
            # Open loop with the real trajectory's executed action sequence:
            # no future states/observations are supplied to the model.
            sh = th.cat((sh[:, 1:], ps[:, None]), 1)
            oh = th.cat((oh[:, 1:], po[:, None]), 1)
            ah = th.cat((ah[:, 1:], batch['forced_actions_onehot'][0, safe, None]), 1)
            times = th.cat((times[:, 1:], times[:, -1:] + 1), 1)
            valid = th.cat((valid[:, 1:], th.ones_like(valid[:, :1])), 1)
        return records

    @th.no_grad()
    def episode(mode, ep):
        seed_everything(task.get('attack_seed', 920001) + ep)
        runner.mode = mode
        runner_module.select_followup = random_followup if mode == 'random_followup' else original_select
        batch = runner.run(test_mode=True) if mode == 'natural' else runner.run_wolfpack_attacker(test_mode=True)
        record = dict(runner.record, episode=ep, mode=mode, hazards=runner.hazards)
        starts = batch['initial_agent'][0, :runner.t, 0] >= 0
        record['start_steps'] = starts.nonzero().flatten().tolist()
        record['initial_state_hash'] = hashlib.sha256(batch['state'][0, 0].numpy().tobytes()).hexdigest()
        return batch, record

    output = dict(task=task, checkpoint_sha256=checkpoint_hash,
                  scheduler_sha256=scheduler_hash, script_sha256=digest(__file__),
                  checkpoint_updates=learner.wall_updates, episodes=[], prediction=[], branches=[],
                  started_at=time.time(), status='running')
    try:
        if task.get('kind') == 'branch':
            args.num_attack_test = args.attack_duration + 1
            candidates = task.get('candidates', [4, 7, 10, 13, 16, 19])
            if (len(candidates) < 3 or candidates != sorted(set(candidates))
                    or candidates[0] < 0 or candidates[-1]-candidates[0] >= args.attack_period):
                raise ValueError('Use at least three increasing nonnegative candidate times within attack_period')
            for ep in range(task.get('branch_episodes', 8)):
                args.env_args['seed'] = task.get('env_seed', 910001) + ep
                runner.close_env()
                runner = setup(DiagnosticRunner(args, NullLogger()))
                base, base_record = episode('natural', ep)
                length = runner.t
                choices = [t for t in candidates if t + args.attack_duration < length]
                if len(choices) < 3:
                    continue
                origin = choices[0]
                start = max(0, origin - args.trans_input_len + 1)
                with th.no_grad():
                    scores = qdiff(th.arange(start, origin+1)[None], base['state'][:, start:origin+1])[0, -1]
                # Equal initial agent across candidate times; cycling covers
                # agents rather than selecting a favorable attacker identity.
                initial = ep % args.n_agents
                for t in choices:
                    runner.close_env()
                    runner = setup(DiagnosticRunner(args, NullLogger()))
                    runner.forced_time, runner.forced_agent = t, initial
                    attacked, record = episode('wolfpack', ep)
                    common = min(t + 1, runner.t + 1)
                    prefix_error = float((base['state'][:, :common] - attacked['state'][:, :common]).abs().max())
                    prefix_ok = common == t + 1 and prefix_error <= 1e-6
                    with th.no_grad():
                        qs = mac.forward_sequence(attacked)
                        gap = 0.0
                        for at in range(t, min(t + args.attack_duration + 1, runner.t)):
                            a = attacked['actions'][:, at].squeeze(-1)
                            forced = attacked['forced_actions'][:, at].squeeze(-1)
                            state, avail = attacked['state'][:, at], attacked['avail_actions'][:, at]
                            gap += float(joint_q(learner.mixer, args, qs[:, at], state, a, avail) -
                                         joint_q(learner.mixer, args, qs[:, at], state, forced, avail))
                    output['branches'].append(dict(episode=ep, t=t, initial=initial,
                        prediction=float(scores[t-origin]), real_q_gap=gap,
                        return_drop=base_record['return_value']-record['return_value'],
                        win_drop=int(base_record['win'])-int(record['win']),
                        prefix_ok=prefix_ok, prefix_max_error=prefix_error,
                        base=base_record, attacked=record))
                    save_json(task['output'], output)
                print('branch', ep, 'done', flush=True)
        else:
            for mode in task.get('modes', ['natural', 'uniform', 'wolfpack', 'random_followup']):
                runner.close_env()
                runner = setup(DiagnosticRunner(args, NullLogger()))
                for ep in range(args.test_nepisode):
                    batch, record = episode(mode, ep)
                    output['episodes'].append(record)
                    if mode in ('natural', 'wolfpack') and ep < task.get('prediction_episodes', 8):
                        metrics = predict_metrics(batch, runner.t)
                        output['prediction'].extend(dict(x, episode=ep, mode=mode) for x in metrics)
                    if (ep + 1) % 8 == 0:
                        save_json(task['output'], output)
                        print(mode, ep+1, '/', args.test_nepisode, flush=True)
        if scheduler_hash and digest(task['scheduler_checkpoint']) != scheduler_hash:
            raise RuntimeError('Scheduler checkpoint changed during audit')
        if digest(task['checkpoint']) != checkpoint_hash:
            raise RuntimeError('Checkpoint changed during audit')
        output.update(status='completed', finished_at=time.time())
        save_json(task['output'], output)
    finally:
        runner_module.select_followup = original_select
        runner.close_env()


if __name__ == '__main__':
    main()

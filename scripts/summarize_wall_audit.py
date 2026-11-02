#!/usr/bin/env python3
"""Summarize audit JSON with seed-level uncertainty and paired branch checks."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np


def mean(values):
    return float(np.mean(values)) if len(values) else None


def wilson(wins):
    n = len(wins)
    if not n:
        return None
    p, z = mean(wins), 1.959963984540054
    mid = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [mid-half, mid+half]


def bootstrap_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    rng = np.random.RandomState(1701)
    estimates = values[rng.randint(len(values), size=(10000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), n=len(values),
                ci95=[float(x) for x in np.percentile(estimates, [2.5, 97.5])])


def rank(values):
    values = np.asarray(values)
    return np.array([((values < x).sum() + (values <= x).sum() - 1) / 2 for x in values])


def spearman(a, b):
    a, b = rank(a), rank(b)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize(directory):
    data = []
    for manifest in (directory/'manifest.json', directory/'cross_seed/manifest.json'):
        if not manifest.exists():
            continue
        for task_path in json.loads(manifest.read_text())['tasks']:
            task = json.loads(Path(task_path).read_text())
            result = Path(task['output'])
            if result.exists():
                item = json.loads(result.read_text())
                if item['status'] == 'completed':
                    data.append(item)
    summary = dict(targets={}, branches={}, five_seed={}, completed_tasks=len(data),
                   episode_count=sum(len(d['episodes']) for d in data),
                   branch_count=sum(len(d['branches']) for d in data))
    for d in data:
        task = d['task']
        if task.get('kind') == 'branch':
            config = json.loads(Path(task['config']).read_text()) if task.get('config') else {}
            temperature = float(config.get('temperature', 0.5))
            groups = defaultdict(list)
            for row in d['branches']:
                if row['prefix_ok']:
                    groups[row['episode']].append(row)
            stats = defaultdict(list)
            for rows in groups.values():
                if len(rows) < 3:
                    continue
                pred = np.array([r['prediction'] for r in rows])
                q = np.array([r['real_q_gap'] for r in rows])
                reward = np.array([r['return_drop'] for r in rows])
                weights = np.exp((pred-pred.max())/temperature)
                weights /= weights.sum()
                rho = spearman(pred, q)
                if rho is not None:
                    stats['rank_q'].append(rho)
                rho = spearman(pred, reward)
                if rho is not None:
                    stats['rank_return'].append(rho)
                chosen = pred.argmax()
                stats['top1_q_hit'].append(float(q[chosen] >= q.max()-1e-9))
                stats['top1_return_hit'].append(float(reward[chosen] >= reward.max()-1e-9))
                stats['softmax_q_lift'].append(float(weights@q-q.mean()))
                stats['softmax_return_lift'].append(float(weights@reward-reward.mean()))
                stats['argmax_q_lift'].append(float(q[chosen]-q.mean()))
                stats['argmax_return_lift'].append(float(reward[chosen]-reward.mean()))
                stats['q_mae'].append(float(np.abs(pred-q).mean()))
                stats['actual_gap_mean'].append(float(q.mean()))
            branch_metrics = {k:bootstrap_mean(v) for k,v in stats.items()}
            for key in ('top1_q_hit', 'top1_return_hit'):
                if key in branch_metrics:
                    branch_metrics[key]['ci95'] = wilson(stats[key])
            summary['branches'][task['map']] = dict(
                tested=len(d['branches']), matched=sum(r['prefix_ok'] for r in d['branches']),
                groups=len(groups), metrics=branch_metrics)
            continue
        target = dict(map=task['map'], seed=task['seed'], step=task['checkpoint_step'],
                      vanilla=bool(task.get('vanilla')), modes={}, prediction={})
        for mode in sorted(set(e['mode'] for e in d['episodes'])):
            rows = [e for e in d['episodes'] if e['mode'] == mode]
            hazards = [h for e in rows for h in e['hazards']]
            target['modes'][mode] = dict(n=len(rows), win_rate=mean([e['win'] for e in rows]),
                win_ci95=wilson([e['win'] for e in rows]),
                reward=mean([e['return_value'] for e in rows]),
                changed_steps=mean([e['changed_steps'] for e in rows]),
                bursts=mean([e['bursts'] for e in rows]),
                first_attack_step=mean([e['start_steps'][0] for e in rows if e['start_steps']]),
                no_burst_fraction=mean([e['bursts']==0 for e in rows]) if mode != 'natural' else None,
                entropy=mean([h['entropy'] for h in hazards]),
                early_hazard=mean([h['probability'] for h in hazards if h['t'] < 6]))
        for mode in ('natural', 'wolfpack', 'combined'):
            for horizon in (1,3):
                rows = [r for r in d['prediction'] if r['horizon']==horizon and (mode=='combined' or r['mode']==mode)]
                if not rows:
                    continue
                metrics = {key:mean([r[key] for r in rows]) for key in
                    ['state_mse','state_persistence_mse','physical_state_mse','physical_state_persistence_mse',
                     'obs_mse','obs_persistence_mse','avail_exact','term_brier']}
                tp,fp,fn = [sum(r[key] for r in rows) for key in ['avail_tp','avail_fp','avail_fn']]
                metrics.update(n=len(rows),avail_f1=2*tp/max(2*tp+fp+fn,1),
                               avail_precision=tp/max(tp+fp,1),avail_recall=tp/max(tp+fn,1))
                target['prediction'][mode+'_h'+str(horizon)] = metrics
        summary['targets'][task['id']] = target
    for map_name in ('3m','8m'):
        rows = []
        for seed in range(1,6):
            key = map_name+'_seed'+str(seed)
            base = summary['targets'].get(key+'_wall')
            cross = summary['targets'].get(key+'_cross_scheduler')
            random = summary['targets'].get(key+'_random_followup')
            if not base:
                continue
            row = dict(seed=seed, **{mode:base['modes'][mode]['win_rate'] for mode in ['natural','uniform','wolfpack']})
            if cross:
                row['cross_scheduler'] = cross['modes']['wolfpack']['win_rate']
            if random:
                row['random_followup'] = random['modes']['random_followup']['win_rate']
            elif 'random_followup' in base['modes']:
                row['random_followup'] = base['modes']['random_followup']['win_rate']
            rows.append(row)
        all_keys = ['natural','uniform','wolfpack','cross_scheduler','random_followup']
        stats = {k:bootstrap_mean([r[k] for r in rows if k in r]) for k in all_keys}
        contrasts = {}
        for a,b in [('uniform','wolfpack'),('cross_scheduler','wolfpack'),('random_followup','wolfpack')]:
            contrasts[a+'_minus_'+b] = bootstrap_mean([r[a]-r[b] for r in rows if a in r and b in r])
        summary['five_seed'][map_name] = dict(rows=rows, means=stats, contrasts=contrasts)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    s = summarize(args.directory)
    (args.directory/'summary.json').write_text(json.dumps(s, indent=2, allow_nan=False))
    print(json.dumps({k:s[k] for k in ['completed_tasks','episode_count','branch_count','five_seed','branches']},indent=2))


if __name__ == '__main__':
    main()

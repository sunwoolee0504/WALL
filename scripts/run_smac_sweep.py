#!/usr/bin/env python3
"""Prepare or execute a reproducible, two-GPU WALL seed sweep.

Preparation snapshots source and defender weights. Execution is explicit and
supports memory-budgeted concurrency and adoption of existing training PIDs. Logs and status
remain under the campaign directory even when the shell disconnects.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MAPS = ['3m', '3s_vs_3z', '2s3z', '8m', '1c3s5z', 'MMM']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def gpu_inventory():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in output.splitlines():
        index, uuid, memory, utilization = [v.strip() for v in line.split(',')]
        result[int(index)] = dict(uuid=uuid, memory_mib=int(memory), utilization=int(utilization))
    return result


def make_jobs(campaign, python, maps, seeds, timesteps):
    jobs = []
    for map_name in maps:
        for seed in seeds:
            job_id = '{}_seed{}'.format(map_name, seed)
            directory = campaign / 'jobs' / job_id
            command = [python, '-u', str(campaign / 'source' / 'launch_cuda.py'),
                       '--config=wall_qmix', '--env-config=sc2', 'with',
                       'env_args.map_name=' + map_name, 'seed=' + str(seed),
                       't_max=' + str(timesteps), 'pretrain=True', 'use_cuda=True',
                       'local_results_path=' + str(directory), 'label=' + job_id]
            jobs.append(dict(id=job_id, map=map_name, seed=seed, status='pending',
                             command=command, directory=str(directory)))
    return jobs


def ready_jobs(jobs, capacity, serialize_maps=True):
    """Prefer earlier maps; optionally wait for all their seeds to finish."""
    unfinished = [job for job in jobs if job['status'] != 'completed']
    if not unfinished or any(job['status'] in ('failed', 'interrupted') for job in unfinished):
        return []
    current_map = unfinished[0]['map']
    return [job for job in unfinished if (not serialize_maps or job['map'] == current_map) and job['status'] == 'pending'][:capacity]


def prepare(campaign, python, gpus, timesteps, sc2path):
    if campaign.exists():
        raise ValueError('Campaign already exists; preparation never overwrites runs')
    if not Path(python).is_file():
        raise ValueError('Python interpreter does not exist: ' + python)
    if len(set(gpus)) != len(gpus) or not gpus:
        raise ValueError('Select distinct GPU indices')
    if timesteps < 1:
        raise ValueError('Timesteps must be positive')
    if not (Path(sc2path) / 'Versions').is_dir():
        raise ValueError('SC2PATH must contain Versions')
    inventory = gpu_inventory()
    selected = {str(index): inventory[index] for index in gpus}
    for name in MAPS:
        for filename in ('agent.th', 'mixer.th', 'opt.th'):
            if not (ROOT / 'pretrain_model/qmix' / name / filename).is_file():
                raise FileNotFoundError('Missing QMIX pretrained file: ' + name + '/' + filename)
    campaign.mkdir(parents=True)
    snapshot = campaign / 'source'
    shutil.copytree(ROOT / 'src', snapshot / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in MAPS:
        target = snapshot / 'pretrain_model/qmix' / name
        target.mkdir(parents=True)
        for filename in ('agent.th', 'mixer.th', 'opt.th'):
            shutil.copy2(ROOT / 'pretrain_model/qmix' / name / filename, target / filename)
    (snapshot / 'launch_cuda.py').write_text(
        "import os, runpy, sys\nfrom pathlib import Path\nimport torch\n"
        "if not torch.cuda.is_available() or torch.cuda.device_count() != 1:\n"
        "    raise RuntimeError('Sweep requires exactly one visible CUDA GPU; no CPU fallback')\n"
        "print('WALL_DEVICE', torch.__version__, torch.cuda.get_device_name(0), os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)\n"
        "entry = Path(__file__).resolve().parent / 'src/main.py'\n"
        "sys.path.insert(0, str(entry.parent))\nsys.argv[0] = str(entry)\n"
        "runpy.run_path(str(entry), run_name='__main__')\n")
    hashes = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(snapshot.rglob('*')) if path.is_file()}
    save(campaign / 'source_manifest.json', hashes)
    state = dict(format_version=1, created_at=now(), status='prepared', python=python,
                 project=str(ROOT), source_snapshot=str(snapshot), sc2path=sc2path,
                 algorithm='wall_qmix', maps=MAPS, seeds=[1, 2, 3, 4, 5],
                 timesteps=timesteps, gpu_indices=gpus, gpu_inventory=selected,
                 jobs=make_jobs(campaign, python, MAPS, [1, 2, 3, 4, 5], timesteps))
    save(campaign / 'status.json', state)
    return state


def process_identity(pid):
    """PID plus Linux start time prevents adopting/signalling a reused PID."""
    try:
        directory = Path('/proc') / str(pid)
        raw = (directory / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        command = [part.decode() for part in (directory / 'cmdline').read_bytes().split(b'\0') if part]
        return dict(start_ticks=int(fields[19]), command=command)
    except (FileNotFoundError, ProcessLookupError):
        return None


def validate_adoption(job, expected_uuid):
    identity = process_identity(job['pid'])
    if identity is None or identity['command'] != job['command']:
        raise ValueError('Running process does not match recorded command: ' + job['id'])
    if job.get('process_start_ticks', identity['start_ticks']) != identity['start_ticks']:
        raise ValueError('Recorded PID has been reused: ' + job['id'])
    environment = (Path('/proc') / str(job['pid']) / 'environ').read_bytes().split(b'\0')
    visible = next((part.split(b'=', 1)[1].decode() for part in environment
                    if part.startswith(b'CUDA_VISIBLE_DEVICES=')), None)
    if visible != expected_uuid or os.getpgid(job['pid']) != job['pid']:
        raise ValueError('GPU or process group differs for ' + job['id'])
    job['process_start_ticks'] = identity['start_ticks']


def process_memory():
    output = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in output.splitlines():
        pid, uuid, memory = [value.strip() for value in line.split(',')]
        if memory.isdigit():
            result[int(pid)] = dict(uuid=uuid, memory_mib=int(memory))
    return result


def read_policy(campaign):
    policy = dict(memory_target_mib=5000, headroom_mib=384, new_job_reserve_mib=1000,
                  max_jobs_per_gpu=1, max_warming_per_gpu=2,
                  launch_interval_seconds=15, serialize_maps=True, allow_shared_gpus=False)
    path = campaign / 'scheduler_policy.json'
    if path.exists():
        policy.update(json.loads(path.read_text()))
    for name in ('memory_target_mib', 'new_job_reserve_mib', 'max_jobs_per_gpu',
                 'max_warming_per_gpu', 'launch_interval_seconds'):
        if not isinstance(policy[name], int) or isinstance(policy[name], bool) or policy[name] <= 0:
            raise ValueError('Positive integer required for policy ' + name)
    if not 0 <= policy['headroom_mib'] < policy['memory_target_mib']:
        raise ValueError('Headroom must be below memory target')
    if not isinstance(policy['serialize_maps'], bool):
        raise ValueError('serialize_maps must be boolean')
    if not isinstance(policy['allow_shared_gpus'], bool):
        raise ValueError('allow_shared_gpus must be boolean')
    return policy


def estimated_memory(job, jobs, policy):
    observed = [peer.get('peak_gpu_mib', 0) for peer in jobs
                if peer['map'] == job['map'] and peer.get('learner_verified')]
    # Peak observations plus 5% growth, with an initial conservative reservation
    # until the first batch has actually trained for this map.
    return max(384, int(max(observed) * 1.05 + 1)) if observed else policy['new_job_reserve_mib']


def fits_memory(job, index, jobs, inventory, memory, policy):
    active = [peer for peer in jobs if peer['status'] == 'running' and peer['gpu_index'] == index]
    if len(active) >= policy['max_jobs_per_gpu']:
        return False
    if sum(not peer.get('learner_verified', False) for peer in active) >= policy['max_warming_per_gpu']:
        return False
    reserved = 0
    for peer in active:
        resident = memory.get(peer['pid'], {}).get('memory_mib', 0)
        required = max(peer.get('peak_gpu_mib', 0), resident)
        if not peer.get('learner_verified'):
            required = max(required, estimated_memory(peer, jobs, policy))
        reserved += max(0, required - resident)
    projected = inventory[index]['memory_mib'] + reserved + estimated_memory(job, jobs, policy) + policy['headroom_mib']
    return projected <= policy['memory_target_mib']


def sacred_data(campaign, job):
    paths = [Path(job['sacred_directory'])] if job.get('sacred_directory') else sorted(
        (campaign / 'source/results/sacred').glob('[0-9]*'))
    for path in paths:
        try:
            config = json.loads((path / 'config.json').read_text())
            if config.get('label') != job['id']:
                continue
            job['sacred_directory'] = str(path)
            return json.loads((path / 'run.json').read_text()), json.loads((path / 'info.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return {}, {}


def finish_record(campaign, state, job, code):
    checkpoints = list((Path(job['directory']) / 'models').glob('*/*/learner.th'))
    final = max(checkpoints, key=lambda p: int(p.parent.name)) if checkpoints else None
    run, _ = sacred_data(campaign, job)
    completed = (code in (0, None) and run.get('status') == 'COMPLETED' and
                 final is not None and int(final.parent.name) >= state['timesteps'])
    job.update(status='completed' if completed else 'failed', exit_code=code,
               ended_at=now(), final_checkpoint=str(final) if final else None)
    return completed


def terminate_owned(job, process=None):
    identity = process_identity(job['pid'])
    if identity is None or identity['start_ticks'] != job['process_start_ticks']:
        return
    os.killpg(job['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process is not None:
            process.poll()
        identity = process_identity(job['pid'])
        if identity is None or identity['start_ticks'] != job['process_start_ticks']:
            return
        time.sleep(0.2)
    os.killpg(job['pid'], signal.SIGKILL)
    if process is not None:
        process.wait()


def execute(campaign, adopt=False):
    state_path = campaign / 'status.json'
    with (campaign / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text())
        if state['status'] != ('running' if adopt else 'prepared'):
            raise ValueError('Use --adopt only for a running campaign; existing failed jobs are not restarted')
        for name, expected in json.loads((campaign / 'source_manifest.json').read_text()).items():
            if hashlib.sha256((campaign / 'source' / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Snapshot changed after preparation: ' + name)
        policy = read_policy(campaign)
        inventory = gpu_inventory()
        for index in state['gpu_indices']:
            if inventory[index]['uuid'] != state['gpu_inventory'][str(index)]['uuid']:
                raise RuntimeError('Selected GPU UUID changed')
            if not adopt and not policy['allow_shared_gpus'] and inventory[index]['memory_mib'] > 512:
                raise RuntimeError('Selected GPU {} is occupied'.format(index))
        active = {}
        for job in state['jobs']:
            if job['status'] == 'running':
                validate_adoption(job, state['gpu_inventory'][str(job['gpu_index'])]['uuid'])
                active[job['id']] = (None, None, job)
                job['adopted_at'] = now()
        old_supervisor = state.get('supervisor_pid')
        state.update(status='running', supervisor_pid=os.getpid(), scheduler_version=2)
        state.setdefault('started_at', now())
        if adopt:
            state.setdefault('supervisor_handoffs', []).append(
                dict(time=now(), previous_pid=old_supervisor, pid=os.getpid(), preserved_jobs=list(active)))
        (campaign / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
        save(state_path, state)
        print(now(), 'Supervisor', os.getpid(), 'adopted', list(active), flush=True)
        last_launch = 0.0
        def stop(signum, frame):
            raise KeyboardInterrupt('Supervisor stop signal {}'.format(signum))
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while True:
                policy = read_policy(campaign)
                inventory, memory = gpu_inventory(), process_memory()
                state.update(scheduler_policy=policy, updated_at=now(),
                             gpu_usage={str(index): inventory[index] for index in state['gpu_indices']})
                for key, (process, stream, job) in list(active.items()):
                    code = process.poll() if process is not None else None
                    identity = process_identity(job['pid'])
                    alive = identity is not None and identity['start_ticks'] == job['process_start_ticks']
                    if alive and (process is None or code is None):
                        used = memory.get(job['pid'], {}).get('memory_mib', 0)
                        job['gpu_memory_mib'] = used
                        job['peak_gpu_mib'] = max(job.get('peak_gpu_mib', 0), used)
                        if not job.get('learner_verified'):
                            _, info = sacred_data(campaign, job)
                            if info.get('loss_td'):
                                job['learner_verified'] = True
                                job['first_verified_learner_step'] = info['loss_td_T'][-1]
                        continue
                    if stream is not None:
                        stream.close()
                    active.pop(key)
                    completed = finish_record(campaign, state, job, code)
                    save(state_path, state)
                    print(now(), job['id'], job['status'], 'exit', code, flush=True)
                    if not completed:
                        raise RuntimeError('Job failed or final checkpoint is missing: ' + job['id'])
                pending = ready_jobs(state['jobs'], len(state['jobs']), policy['serialize_maps'])
                if not active and not pending:
                    state.update(status='completed', ended_at=now())
                    save(state_path, state)
                    print(now(), 'All runs completed', flush=True)
                    return
                # One new process at a time, so measured memory can settle before
                # the next launch. Only the original selected GPU UUIDs are used.
                if pending and time.monotonic() - last_launch >= policy['launch_interval_seconds']:
                    job = pending[0]
                    candidates = [index for index in state['gpu_indices']
                                  if fits_memory(job, index, state['jobs'], inventory, memory, policy)]
                    if candidates:
                        index = min(candidates, key=lambda i: inventory[i]['memory_mib'])
                        directory = Path(job['directory'])
                        directory.mkdir(parents=True, exist_ok=True)
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=state['gpu_inventory'][str(index)]['uuid'],
                                   CUDA_DEVICE_ORDER='PCI_BUS_ID', SC2PATH=state['sc2path'],
                                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                        stream = (directory / 'console.log').open('ab')
                        process = subprocess.Popen(job['command'], cwd=state['source_snapshot'], env=env,
                                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        identity = process_identity(process.pid)
                        if identity is None:
                            raise RuntimeError('Training process exited during launch')
                        job.update(status='running', gpu_index=index, pid=process.pid, started_at=now(),
                                   process_start_ticks=identity['start_ticks'])
                        state['training_launched'] = True
                        active[job['id']] = (process, stream, job)
                        last_launch = time.monotonic()
                        print(now(), 'Started', job['id'], 'GPU', index, 'PID', process.pid, flush=True)
                save(state_path, state)
                time.sleep(3)
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                         ended_at=now(), error=repr(exc))
            for process, stream, job in active.values():
                terminate_owned(job, process)
                if stream is not None:
                    stream.close()
                job.update(status='interrupted', ended_at=now())
            save(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Run a previously prepared campaign')
    parser.add_argument('--adopt', action='store_true', help='Adopt verified live training PIDs after stopping only their old supervisor')
    parser.add_argument('--python', default=sys.executable, help='Training Python (must have CUDA PyTorch)')
    parser.add_argument('--gpus', type=int, nargs='+', default=[6, 7])
    parser.add_argument('--timesteps', type=int, default=3000000)
    parser.add_argument('--sc2path', default=os.environ.get('SC2PATH', str(Path.home() / 'StarCraftII')))
    args = parser.parse_args()
    campaign = args.campaign.expanduser().resolve()
    if args.run or args.adopt:
        execute(campaign, adopt=args.adopt)
    else:
        state = prepare(campaign, str(Path(args.python).absolute()), args.gpus, args.timesteps, args.sc2path)
        print('Prepared {} runs at {}'.format(len(state['jobs']), campaign))
        print('Start explicitly with: {} {} --campaign {} --run'.format(sys.executable, Path(__file__).resolve(), campaign))


if __name__ == '__main__':
    main()

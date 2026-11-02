"""Sweep orchestration checks do not launch GPU jobs or create game processes."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('sweep', Path(__file__).resolve().parents[1] / 'scripts/run_smac_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_default_plan_has_30_unique_runs_with_pretrained_paper_budget(tmp_path):
    jobs = sweep.make_jobs(tmp_path, '/python', sweep.MAPS, [1, 2, 3, 4, 5], 3000000)
    assert len({j['id'] for j in jobs}) == len(jobs) == 30
    assert [j['map'] for j in jobs[:5]] == ['3m'] * 5
    for job in jobs:
        assert 'pretrain=True' in job['command']
        assert 't_max=3000000' in job['command']
        assert 'use_cuda=True' in job['command']
        assert 'seed=' + str(job['seed']) in job['command']


def test_map_barrier_and_capacity(tmp_path):
    jobs = sweep.make_jobs(tmp_path, '/python', ['3m', '8m'], [1, 2, 3], 10)
    assert sweep.ready_jobs(jobs, 2) == jobs[:2]
    for job in jobs[:2]:
        job['status'] = 'completed'
    jobs[2]['status'] = 'running'
    assert sweep.ready_jobs(jobs, 2) == []
    jobs[2]['status'] = 'completed'
    assert sweep.ready_jobs(jobs, 1) == jobs[3:4]
    jobs[3]['status'] = 'failed'
    assert sweep.ready_jobs(jobs, 2) == []


def test_parallel_policy_preserves_pending_order_without_map_barrier(tmp_path):
    jobs = sweep.make_jobs(tmp_path, '/python', ['3m', '8m'], [1, 2], 10)
    jobs[0]['status'] = jobs[1]['status'] = 'running'
    assert sweep.ready_jobs(jobs, 2) == []
    assert sweep.ready_jobs(jobs, 2, serialize_maps=False) == jobs[2:]


def test_memory_budget_counts_startup_reservations_and_global_headroom(tmp_path):
    policy = sweep.read_policy(tmp_path)
    policy.update(max_jobs_per_gpu=9)
    jobs = sweep.make_jobs(tmp_path, '/python', ['3m'], [1, 2, 3], 10)
    jobs[0].update(status='running', gpu_index=6, pid=100, peak_gpu_mib=500, learner_verified=True)
    inventory = {6: {'memory_mib': 520}}
    memory = {100: {'memory_mib': 500}}
    assert sweep.fits_memory(jobs[1], 6, jobs, inventory, memory, policy)
    inventory[6]['memory_mib'] = 4500
    assert not sweep.fits_memory(jobs[1], 6, jobs, inventory, memory, policy)
    # The second job has barely started and must reserve its eventual footprint.
    inventory[6]['memory_mib'] = 4100
    jobs[1].update(status='running', gpu_index=6, pid=101, peak_gpu_mib=0)
    memory[101] = {'memory_mib': 0}
    assert not sweep.fits_memory(jobs[2], 6, jobs, inventory, memory, policy)


def test_initial_map_reservation_and_warming_limit(tmp_path):
    policy = sweep.read_policy(tmp_path)
    policy.update(max_jobs_per_gpu=9, max_warming_per_gpu=1)
    jobs = sweep.make_jobs(tmp_path, '/python', ['3m', '8m'], [1, 2], 10)
    assert sweep.estimated_memory(jobs[0], jobs, policy) == 1000
    jobs[0].update(status='running', gpu_index=6, pid=100, peak_gpu_mib=500)
    assert not sweep.fits_memory(jobs[1], 6, jobs, {6: {'memory_mib': 100}}, {}, policy)
    jobs[0]['learner_verified'] = True
    assert sweep.estimated_memory(jobs[1], jobs, policy) == 526
    assert sweep.estimated_memory(jobs[2], jobs, policy) == 1000


def test_adoption_verifies_pid_command_gpu_and_start_time():
    import os
    import subprocess
    import sys
    import pytest
    command = [sys.executable, '-c', 'import time; time.sleep(30)']
    process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='test-gpu'), start_new_session=True)
    try:
        job = dict(id='test', pid=process.pid, command=command)
        sweep.validate_adoption(job, 'test-gpu')
        assert job['process_start_ticks'] > 0 and process.poll() is None
        with pytest.raises(ValueError, match='GPU'):
            sweep.validate_adoption(job, 'another-gpu')
        job['process_start_ticks'] += 1
        with pytest.raises(ValueError, match='reused'):
            sweep.validate_adoption(job, 'test-gpu')
        job['command'] = ['/different-command']
        with pytest.raises(ValueError, match='command'):
            sweep.validate_adoption(job, 'test-gpu')
    finally:
        process.terminate()
        process.wait(timeout=5)

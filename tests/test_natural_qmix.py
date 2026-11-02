"""The natural baseline preserves the defender TD update without training predictors."""
from types import SimpleNamespace
import torch
from test_learner import config, make_batch, make_learner
from run import _evaluate


def test_natural_control_has_same_td_update_and_no_predictor_updates():
    args=config()
    torch.manual_seed(91)
    batch=make_batch(args)
    batch.update({'forced_actions':batch['actions'].clone()})
    torch.manual_seed(92)
    wall=make_learner(args,batch)
    torch.manual_seed(92)
    natural=make_learner(config(train_attack_predictors=False),batch)
    before={label:{k:v.clone() for k,v in model.state_dict().items()} for label,model in
            [('planner',natural.planning_transformer),('qdiff',natural.qdiff_transformer)]}
    def forbidden(*_,**__):
        raise AssertionError('Natural training must not construct attack targets')
    natural.auxiliary.train=forbidden
    wall.train(batch,1,1)
    metrics=natural.train(batch,1,1)
    assert 'loss_td' in metrics and 'loss_planning' not in metrics and 'loss_qdiff' not in metrics
    for a,b in [(wall.mac.agent,natural.mac.agent),(wall.mixer,natural.mixer)]:
        for key,value in a.state_dict().items():
            torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
    for label,model in [('planner',natural.planning_transformer),('qdiff',natural.qdiff_transformer)]:
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,before[label][key],rtol=0,atol=0)
    assert not natural.optimizer_planning.state and not natural.optimizer_qdiff.state


def test_natural_evaluation_does_not_use_untrained_predictors():
    class Runner:
        count=0
        def run(self,test_mode):
            assert test_mode
            self.count+=1
        def run_wolfpack_attacker(self,**_):
            raise AssertionError('Evaluate with an independent trained attacker instead')
    runner=Runner()
    _evaluate(SimpleNamespace(test_nepisode=3,save_replay=False,evaluate_wolfpack=False),runner,123)
    assert runner.count==3


def test_natural_config_is_accepted_by_actual_entry_point():
    from main import _get_config
    args=["--config=natural_qmix"]
    parsed=_get_config(args,"--config","algs")
    assert parsed["num_attack_train"]==0
    assert parsed["train_attack_predictors"] is False
    assert parsed["evaluate_wolfpack"] is False
    assert not args


def test_public_evaluator_rejects_untrained_natural_attacker(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_checkpoint.py"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"train_attack_predictors": False}))
    output = tmp_path / "evaluation"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path),
         "--checkpoint", str(tmp_path / "learner.th"),
         "--sc2path", str(tmp_path / "SC2"), "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "supply a trained --scheduler-checkpoint" in result.stderr
    assert not output.exists()

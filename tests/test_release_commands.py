"""Release protocols must not silently mix attackers, budgets or seed counts."""
from pathlib import Path
import sys
import json
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import release_utils as util
import evaluate_release as evaluation
import train_release as training


def fixture_catalog(root):
    assets = {}

    def asset(path, data=b"fixture weights"):
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        assets[path] = {"bytes": len(data), "sha256": util.digest(p)}
        return path

    defenders = {}
    attackers = {}
    for seed in range(1, 6):
        cfg = "configs/seed%d.json" % seed
        util.save_json(
            root / cfg,
            dict(
                seed=seed,
                env_args={"map_name": "3m"},
                temperature=0.5,
                num_attack_test=8,
                num_followup_agents=1,
            ),
        )
        defenders[str(seed)] = dict(
            checkpoint=asset("models/d%d.th" % seed), config=cfg
        )
        attackers[str(seed)] = dict(checkpoint=asset("models/a%d.th" % seed))
    for name in ("agent.th", "mixer.th", "opt.th"):
        asset("pretrain_model/qmix/3m/" + name)
    util.save_json(
        root / "configs/train.json", dict(num_attack_train=4, num_followup_agents=1)
    )
    entry = dict(
        defenders=defenders,
        attackers=attackers,
        selected_defender_seed=4,
        selected_attacker_seed=2,
        evaluation=dict(num_attack_test=4, num_followup_agents=2, temperature=0.2),
        training_config="configs/train.json",
        pretrained_directory="pretrain_model/qmix/3m",
    )
    return dict(format_version=1, map_order=["3m"], maps={"3m": entry}, assets=assets)


@pytest.mark.parametrize(
    "profile,count", [("selected", 3), ("five-seed", 11), ("cross", 35)]
)
def test_release_task_counts_and_matched_threat_conditions(
    tmp_path, monkeypatch, profile, count
):
    catalog = fixture_catalog(tmp_path)
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    tasks = evaluation.build_tasks(
        catalog, ["3m"], tmp_path, tmp_path / "eval", profile, 128, tmp_path / "sc2"
    )
    assert len(tasks) == count and len({t["id"] for t in tasks}) == count
    for t in tasks:
        cfg = util.read_json(t["config"])
        assert (
            cfg["num_attack_test"] == 4
            and cfg["num_followup_agents"] == 2
            and cfg["temperature"] == 0.2
        )
        assert t["episodes"] == 128 and t["env_seed"] == 2731001
        assert t["checkpoint_sha256"] == util.digest(t["checkpoint"])
        assert t["scheduler_sha256"] == util.digest(t["scheduler_checkpoint"])
    if profile != "cross":
        assert {t["attacker_seed"] for t in tasks} == {2}
    if profile == "selected":
        assert {t["seed"] for t in tasks} == {4}
    assert sum(t["defender"] == "pretrained_qmix" for t in tasks) == (
        5 if profile == "cross" else 1
    )


def test_asset_hash_and_path_guard(tmp_path):
    catalog = fixture_catalog(tmp_path)
    util.verified_asset(tmp_path, "models/d1.th", catalog)
    (tmp_path / "models/d1.th").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        util.verified_asset(tmp_path, "models/d1.th", catalog)
    for path in ("../outside", "/tmp/outside"):
        with pytest.raises(ValueError):
            util.safe_path(tmp_path, path)


def test_training_uses_original_training_config_not_selected_test_settings(
    tmp_path, monkeypatch
):
    catalog = fixture_catalog(tmp_path)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    cmd = training.command_for("3m", 5, catalog, tmp_path, tmp_path / "out", cpu=True)
    assert str(tmp_path / "configs/train.json") in cmd
    assert (
        "seed=5" in cmd
        and "pretrain=True" in cmd
        and "t_max=3000000" in cmd
        and "use_cuda=False" in cmd
    )
    assert not any(a.startswith("num_followup_agents=") for a in cmd)
    assert util.read_json(tmp_path / "configs/train.json")["num_followup_agents"] == 1


def test_pretrained_cross_evaluations_are_not_five_training_seeds(
    tmp_path, monkeypatch
):
    catalog = fixture_catalog(tmp_path)
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    tasks = evaluation.build_tasks(
        catalog, ["3m"], tmp_path, tmp_path / "eval", "cross", 2, tmp_path / "sc2"
    )
    for t in tasks:
        util.save_json(
            t["output"],
            dict(
                episodes=[
                    dict(win=True, changed_steps=0),
                    dict(win=False, changed_steps=4),
                ]
            ),
        )
    result = evaluation.summarize(tasks, tmp_path / "eval")
    raw = next(r for r in result["metrics"] if r["defender"] == "pretrained_qmix")
    assert raw["n_defender_seeds"] == 1 and raw["defender_seed_sd_pp"] is None
    wall = next(
        r
        for r in result["metrics"]
        if r["defender"] == "wall" and r["mode"] == "wolfpack"
    )
    assert wall["n_defender_seeds"] == 5 and wall["mean_win_pct"] == 50


def test_shipped_catalog_configs_and_reproducible_training_conditions():
    """CI validates the published protocol without downloading binary weights."""
    root = Path(__file__).resolve().parents[1]
    catalog = util.read_json(root / "src/config/release_catalog.json")
    assert catalog["format_version"] == 1
    assert set(catalog["map_order"]) == {
        "3m",
        "8m",
        "2s3z",
        "1c3s5z",
        "MMM",
        "3s_vs_3z",
    }
    for name, entry in catalog["maps"].items():
        assert (
            set(entry["defenders"])
            == set(entry["attackers"])
            == {"1", "2", "3", "4", "5"}
        )
        assert str(entry["selected_defender_seed"]) in entry["defenders"]
        assert str(entry["selected_attacker_seed"]) in entry["attackers"]
        config = util.read_json(util.safe_path(root, entry["training_config"]))
        assert config["pretrain"] and config["t_max"] == 3000000
        assert config["env_args"]["map_name"] == name
        for key in (
            "num_attack_train",
            "num_followup_agents",
            "temperature",
            "lr",
            "epsilon_start",
        ):
            assert config[key] == entry["training"][key]
        assert (config.get("planner_lr") or config["lr"]) == entry["training"][
            "planner_lr"
        ]
        assert (config.get("qdiff_lr") or config["lr"]) == entry["training"]["qdiff_lr"]
        for seed, model in entry["defenders"].items():
            cfg = util.read_json(util.safe_path(root, model["config"]))
            assert cfg["seed"] == int(seed) and cfg["env_args"]["map_name"] == name
            assert model["training_steps"] >= 3000000
            assert model["checkpoint"] in catalog["assets"]
            for key in ("pretrain_dir", "local_results_path", "checkpoint_path"):
                assert not Path(cfg[key]).is_absolute()
            assert not Path(cfg["env_args"]["replay_dir"]).is_absolute()
            assert cfg["pretrain_dir"] == entry["pretrained_directory"]
        for model in entry["attackers"].values():
            assert model["checkpoint"] in catalog["assets"]
        for filename in ("agent.th", "mixer.th", "opt.th"):
            assert entry["pretrained_directory"] + "/" + filename in catalog["assets"]
    for path, info in catalog["assets"].items():
        util.safe_path(root, path)
        assert info["bytes"] > 0
        assert len(info["sha256"]) == 64 and set(info["sha256"]) <= set(
            "0123456789abcdef"
        )

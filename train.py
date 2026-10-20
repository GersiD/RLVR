'''
Train a model with GRPO on verifiable coding problems.

Usage: uv run python train.py <run-name> [config]
The config (default rlvr.toml) sets the model, problem set, hyperparameters, eval schedule and checkpointing.
The algorithm itself is in src/rlvr/grpo.py.
'''
import sys
import tomllib
from pathlib import Path

import torch

from envs import mbpp
from rlvr.grpo import train
from rlvr.llm import HFCausalLM

ENVS = {"mbpp": mbpp.load_problems}


def main(run_name: str, config_path: Path) -> None:
    conf = tomllib.loads(config_path.read_text())
    model, data, evaluation, checkpoint = conf["model"], conf["data"], conf["eval"], conf["checkpoint"]

    load_problems = ENVS[data["env"]]
    split = {"train_fraction": data["train_fraction"], "seed": data["split_seed"]}
    policy = HFCausalLM(
        model["path"],
        device=None if model["device"] == "auto" else model["device"],
        dtype=getattr(torch, model["dtype"]),
    )
    train(
        policy,
        load_problems("train", **split),
        Path(conf["output_dir"]) / run_name,
        test_problems=load_problems("test", **split) if evaluation["enabled"] else (),
        eval_every=evaluation["every"],
        eval_batch_size=evaluation["batch_size"],
        eval_max_new_tokens=evaluation["max_new_tokens"],
        save_checkpoints=checkpoint["enabled"],
        save_every=checkpoint["every"],
        save_optimizer=checkpoint["save_optimizer"],
        config_file=config_path,
        **conf["train"],
    )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(sys.argv[1], Path(sys.argv[2] if len(sys.argv) == 3 else "rlvr.toml"))

'''
GRPO (group relative policy optimization) with verifiable rewards.

Each step samples a group of completions per problem, scores them with a reward function, and pushes the policy
toward the completions that beat their group's average.
'''
import json
import random
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from rlvr.llm import LLM
from rlvr.rewards import STOP_SEQUENCES, CodeProblem, RewardFn, run_tests, score


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    '''
    Normalize each reward against the other completions for the same problem.

    Args:
        rewards: Shape (num_problems * group_size,), with each problem's group contiguous.
        group_size: Completions per problem.
        eps: Keeps the division finite when every completion in a group got the same reward.

    Returns:
        Advantages, same shape as rewards. A group whose rewards are all equal gets all zeros: it carries no signal.
    '''
    r = rewards.view(-1, group_size) # (num_problems, group_size)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)).view(-1)


def grpo_step(
    policy: LLM,
    optimizer: torch.optim.Optimizer,
    problems: Sequence[CodeProblem],
    reward_fn: RewardFn = run_tests,
    group_size: int = 8,
    micro_batch: int = 4,
    max_new_tokens: int = 512,
    max_grad_norm: float = 1.0,
    stop: Sequence[str] = STOP_SEQUENCES,
) -> dict[str, float]:
    '''
    Run one GRPO update: sample, score, and take one optimizer step.

    There is one gradient step per batch of samples, so the PPO ratio is always 1 and its clipping is omitted.
    There is no KL penalty against a reference model. Completions end early at any string in stop.

    Returns:
        Metrics: mean reward, fraction of groups with any signal, loss, gradient norm before clipping, wall
        times in seconds of sampling, reward computation, and the gradient update (time_sample_s,
        time_reward_s, time_update_s), mean scored tokens per completion (completion_tokens_mean), and the
        fraction of completions that never emitted EOS and ran to max_new_tokens (hit_max_tokens_frac).
    '''
    # Timers are read without a device sync, so optimizer.step()'s queued GPU work may land in a later phase:
    # per-phase times are approximate, and train()'s elapsed_s is the exact total.
    t0 = time.perf_counter()
    rollout = policy.sample([p.prompt for p in problems], group_size, max_new_tokens, stop=stop)
    t1 = time.perf_counter()
    rewards = torch.tensor(
        score([problems[i // group_size] for i in range(len(rollout.texts))], rollout.texts, reward_fn),
        device=policy.device,
    )
    t2 = time.perf_counter()
    advantages = group_advantages(rewards, group_size)
    mask = rollout.completion_mask.float()
    num_tokens = mask.sum()
    hit_max = ~rollout.finished

    # Accumulate gradients over micro-batches so the full-vocab logits fit in memory.
    t3 = time.perf_counter()
    optimizer.zero_grad()
    total_loss = 0.0
    for i in range(0, len(rollout.texts), micro_batch):
        s = slice(i, i + micro_batch)
        logp = policy.log_probs(rollout.sequences[s], rollout.attention_mask[s], rollout.prompt_len)
        loss = -(advantages[s, None] * logp * mask[s]).sum() / num_tokens
        loss.backward()
        total_loss += loss.item()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    t4 = time.perf_counter()

    return {
        "reward": rewards.mean().item(),
        "groups_with_signal": (rewards.view(-1, group_size).std(1) > 0).float().mean().item(),
        "loss": total_loss,
        "grad_norm": grad_norm.item(),
        "time_sample_s": t1 - t0,
        "time_reward_s": t2 - t1,
        "time_update_s": t4 - t3,
        "completion_tokens_mean": mask.sum(1).mean().item(),
        "hit_max_tokens_frac": hit_max.float().mean().item(),
    }


def save_checkpoint(policy: LLM, optimizer: torch.optim.Optimizer | None, path: str | Path) -> None:
    '''
    Save a checkpoint: the model (via policy.save), plus optimizer state unless optimizer is None.

    Fails if path already exists, so checkpoints are never overwritten. To resume, load the model from path
    (e.g. HFCausalLM(path)) and pass torch.load(path / "optimizer.pt") to the new optimizer's load_state_dict.
    '''
    path = Path(path)
    path.mkdir(parents=True)
    policy.save(path)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), path / "optimizer.pt")


def evaluate(
    policy: LLM,
    problems: Sequence[CodeProblem],
    reward_fn: RewardFn = run_tests,
    batch_size: int = 16,
    max_new_tokens: int = 512,
    stop: Sequence[str] = STOP_SEQUENCES,
) -> float:
    '''Greedy pass@1: the fraction of problems whose greedy completion passes reward_fn.'''
    passed = 0.0
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i + batch_size]
        rollout = policy.sample([p.prompt for p in batch], 1, max_new_tokens, temperature=0.0, stop=stop)
        passed += sum(score(batch, rollout.texts, reward_fn))
    return passed / len(problems)


def train(
    policy: LLM,
    problems: Sequence[CodeProblem],
    run_dir: str | Path,
    steps: int,
    test_problems: Sequence[CodeProblem] = (),
    reward_fn: RewardFn = run_tests,
    batch_size: int = 4,
    group_size: int = 8,
    micro_batch: int = 4,
    max_new_tokens: int = 512,
    lr: float = 1e-6,
    max_grad_norm: float = 1.0,
    stop: Sequence[str] = STOP_SEQUENCES,
    seed: int = 0,
    eval_every: int = 50,
    eval_batch_size: int = 16,
    eval_max_new_tokens: int = 512,
    save_checkpoints: bool = True,
    save_every: int = 100,
    save_optimizer: bool = True,
    config_file: str | Path | None = None,
) -> None:
    '''
    Train policy with GRPO.

    Args:
        policy: Model to train, with float32 weights.
        problems: Training problem pool. Each step samples batch_size of them.
        run_dir: New directory for this run. Fails if it already exists, so runs are never overwritten.
            The run's settings are written to run_dir/run.json, and every step's metrics are appended to
            run_dir/metrics.jsonl.
        steps: Number of optimizer steps.
        test_problems: Held-out problems, scored with evaluate() before training, every eval_every steps, and
            at the end. Skipped if empty.
        stop: Strings that end a completion early, in training and evaluation. Defaults to the code reward's
            STOP_SEQUENCES, so generation stops where run_tests would cut the program anyway.
        seed: Seed for problem sampling.
        eval_every: Steps between test evaluations.
        eval_batch_size, eval_max_new_tokens: Passed to evaluate().
        save_checkpoints: Whether to save checkpoints to run_dir/checkpoint-<step>.
        save_every: Steps between checkpoints. The final step is always saved.
        save_optimizer: Include optimizer state in checkpoints so training can resume from them. Adds about
            twice the model's float32 size to each checkpoint.
        config_file: Config file the run was launched from. Copied into run_dir.
        The remaining arguments are passed to grpo_step; lr goes to AdamW.
    '''
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True)
    config = {
        "base_model": str(policy.model_path),
        "reward_fn": reward_fn.__name__,
        "num_train_problems": len(problems),
        "num_test_problems": len(test_problems),
        "steps": steps,
        "batch_size": batch_size,
        "group_size": group_size,
        "micro_batch": micro_batch,
        "max_new_tokens": max_new_tokens,
        "lr": lr,
        "max_grad_norm": max_grad_norm,
        "stop": list(stop),
        "seed": seed,
        "eval_every": eval_every,
        "eval_batch_size": eval_batch_size,
        "eval_max_new_tokens": eval_max_new_tokens,
        "save_checkpoints": save_checkpoints,
        "save_every": save_every,
        "save_optimizer": save_optimizer,
    }
    (run_dir / "run.json").write_text(json.dumps(config, indent=2))
    if config_file is not None:
        shutil.copy(config_file, run_dir)
    metrics_path = run_dir / "metrics.jsonl"

    def log(step: int, metrics: dict[str, float]) -> None:
        print(f"step {step}: " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()), flush=True)
        with metrics_path.open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")

    def run_eval(step: int) -> None:
        if test_problems:
            start = time.perf_counter()
            pass_rate = evaluate(policy, test_problems, reward_fn, eval_batch_size, eval_max_new_tokens, stop)
            log(step, {"test_pass@1": pass_rate, "eval_time_s": time.perf_counter() - start})

    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    train_start = time.perf_counter()
    run_eval(0)
    for step in range(1, steps + 1):
        batch = rng.sample(problems, batch_size)
        metrics = grpo_step(
            policy, optimizer, batch, reward_fn, group_size, micro_batch, max_new_tokens, max_grad_norm, stop
        )
        log(step, {**metrics, "elapsed_s": time.perf_counter() - train_start})
        if step % eval_every == 0 or step == steps:
            run_eval(step)
        if save_checkpoints and (step % save_every == 0 or step == steps):
            save_checkpoint(policy, optimizer if save_optimizer else None, run_dir / f"checkpoint-{step}")

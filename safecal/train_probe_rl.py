"""Small CPU-only policy-gradient pilot for SafeCal diagnostic experiment selection.

Research scope: this trains a neural policy over which of seven shielded probes
(or stop) to choose. A fixed BayesianID posterior supplies observations and
chooses the *final calibration*. This is not LLM RL, a full RL calibration policy,
or a real-hardware safety result. Training uses simulated terminal net rewards.

Run:
  python train_probe_rl.py --episodes 600 --seed 20261121 --output rl_pilot
  python train_probe_rl.py --evaluate rl_pilot/policy.pt --output rl_pilot_eval

No evaluation split, hidden plant state, oracle, or test metrics are available
in the training loop. Evaluation code loads held-out seed ranges only afterwards.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from bayes_id import BayesianID, N_PARTICLES, SIGMAS
from safecal import PROBES, SafeCalEnv

# Low CPU overhead and reproducibility; no GPU or cloud required.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
N_PROBES = len(PROBES)
N_ACTIONS = N_PROBES + 1
STOP = N_PROBES
COST_SCALE = 10.0
PARAMS = ("motor", "friction", "load", "damping", "sensor_bias")
PARAM_MEANS = np.array([0.86, 0.18, 0.17, 0.23, 0.0])
PARAM_SCALE = np.array([0.3, 0.18, 0.22, 0.05, 0.075])


class ProbeNet(nn.Module):
    def __init__(self, features: int = 0, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


def observation(model: BayesianID, env: SafeCalEnv) -> np.ndarray:
    """Features derived strictly from public probe observations + fixed prior.

    The Bayesian particle bank contains no hidden evaluation plant. This
    interface does not expose `env._plant`, family, or task seed to the policy.
    """
    w = model.weights
    plants = model.plants
    a = np.array([[getattr(p, key) for key in PARAMS] for p in plants], dtype=np.float64)
    avg = w @ a
    sd = np.sqrt(np.maximum(w @ ((a-avg)**2), 0))
    n = np.array([(r.name in env.used) for r in PROBES], dtype=np.float64)
    readings = np.zeros((N_PROBES, 2))
    for record in env.records:
        i = model.probe_to_idx[record['probe']]
        y = record['observed']
        readings[i, :len(y)] = y
    readings[:, 0] /= 0.13
    readings[:, 1] /= 0.35
    means = (avg - PARAM_MEANS) / PARAM_SCALE
    stds = sd / PARAM_SCALE
    entropy = -np.sum(w * np.log(np.maximum(w, 1e-300))) / math.log(N_PARTICLES)
    entropy2 = np.sum(w ** 2)
    # Approximate predictive dispersion, useful for selecting informative probes.
    predictions = model.predictions
    avg_pred = np.sum(w[:, None, None] * predictions, axis=0)
    var_pred = np.sum(w[:, None, None] * (predictions - avg_pred[None]) ** 2, axis=0)
    normvar = np.sqrt(np.sum(var_pred / SIGMAS ** 2, axis=1)) / 25
    v = np.concatenate([means, stds, [entropy, entropy2 * 40],
                        np.clip(normvar, 0, 4), n, readings.flatten(),
                        [len(env.records)/2.0, env.remaining_budget/4.5]])
    assert np.isfinite(v).all(), "Invalid features"
    return v.astype(np.float32)


def action_logits(net: ProbeNet, obs: np.ndarray, available: set[str], can_probe: bool):
    x = torch.from_numpy(obs)
    logits, value = net(x)
    mask = torch.full((N_ACTIONS,), -1e9)
    if can_probe:
        for j, p in enumerate(PROBES):
            if p.name in available:
                mask[j] = 0.0
    mask[STOP] = 0.0
    return logits + mask, value


class Episode(NamedTuple):
    reward: float
    quality: float
    cost: float
    probes: tuple[str, ...]
    logprob: torch.Tensor | None
    values: list[torch.Tensor]
    entropies: list[torch.Tensor]


def play(seed: int, net: ProbeNet, *, split: str = 'train', deterministic: bool = False,
         save_grad: bool = False, rng_seed: int = 2027) -> Episode:
    """Run one episode; RL chooses only diagnostic actions, Bayes chooses calibration."""
    env = SafeCalEnv(seed, combined=split == 'combined', novel=split == 'shifted')
    model = BayesianID(samples=8, seed=seed * 31 + 379)
    logprobs = []
    values = []
    entropies = []
    for _ in range(3):
        obs = observation(model, env)
        logits, value = action_logits(net, obs, set(env.available_actions()),
                                      can_probe=len(env.records) < 2)
        if deterministic:
            choice = int(torch.argmax(logits))
        else:
            distribution = Categorical(logits=logits)
            choice = int(distribution.sample())
            if save_grad:
                logprobs.append(distribution.log_prob(torch.tensor(choice)))
                values.append(value)
                entropies.append(distribution.entropy())
        if choice == STOP:
            break
        probe = PROBES[choice].name
        model.update(env.step(probe))
    cal = model.choose_calibration()
    result = env.finish(cal)
    return Episode(float(result['reward']), float(result['quality']), float(result['test_cost']),
                   tuple(r['probe'] for r in env.records),
                   torch.stack(logprobs).sum() if logprobs else None,
                   values, entropies)


def seed_stream(rng: random.Random) -> tuple[int, str]:
    # Prior: single-fault and combined-fault conditions, drawn in equal ratio.
    # Fixed evaluation and shifted-physics seed ranges are never used in training.
    if rng.randrange(2) == 0:
        return rng.randint(1000, 300_000), 'train'
    return rng.randint(50_000, 500_000), 'combined'


def learn(episodes: int, rng_seed: int, out: Path, *, lr: float = 0.001,
          batch_size: int = 24, resume: bool = False) -> ProbeNet:
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    rng = random.Random(rng_seed)
    # Features dimension generated by a pilot, no hidden plant accessed by model.
    first_env = SafeCalEnv(1000)
    first_model = BayesianID(samples=8)
    dim = len(observation(first_model, first_env))
    net = ProbeNet(features=dim)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    log = []
    counter = 0
    if resume:
        checkpoint = torch.load(out/'policy.pt', map_location='cpu', weights_only=True)
        assert checkpoint['seed'] == rng_seed and checkpoint['batch_size'] == batch_size
        net.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        counter = checkpoint['episodes']
        for _ in range(counter):
            seed_stream(rng)
        log = json.loads((out/'train_log.json').read_text())
        print(f'Resuming from {counter} episodes to {episodes}', flush=True)
    if episodes < counter:
        raise ValueError('Expected target episodes >= previous checkpoint')
    for start in range(counter, episodes, batch_size):
        batch = []
        for _ in range(min(batch_size, episodes-start)):
            seed, split = seed_stream(rng)
            batch.append(play(seed, net, split=split, save_grad=True))
        # A value baseline reduces variance. Use the same terminal return
        # at each decision, with step cost already included in return.
        pg_losses=[]
        vf_losses=[]
        ents=[]
        for ep in batch:
            reward = (ep.reward - 0.80) * COST_SCALE
            for val in ep.values:
                vf_losses.append((val - reward)**2)
            if ep.logprob is not None and ep.values:
                advantage = (torch.tensor(reward) - torch.stack(ep.values).mean()).detach()
                pg_losses.append(-ep.logprob * advantage)
                ents.extend(ep.entropies)
        pg = torch.stack(pg_losses).mean()
        vf = torch.stack(vf_losses).mean()
        entropy = torch.stack(ents).mean()
        loss = pg + 0.5 * vf - 0.005 * entropy
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        optim.step()
        counter += len(batch)
        rec = {'episodes': counter, 'train_mean_reward': statistics.fmean(e.reward for e in batch),
               'train_mean_probes': statistics.fmean(len(e.probes) for e in batch),
               'loss': float(loss.detach()), 'policy_loss':float(pg.detach())}
        log.append(rec)
        if len(log) % 5 == 0 or counter == episodes:
            print(f'episodes {counter:5d}/{episodes} | batch reward {rec["train_mean_reward"]:.5f} | probes {rec["train_mean_probes"]:.2f} | loss {rec["loss"]:.4f}',flush=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': net.state_dict(), 'features': dim, 'seed': rng_seed,
                'episodes': episodes, 'batch_size': batch_size,
                'optimizer': optim.state_dict(), 'torch_rng_state': torch.get_rng_state()}, out/'policy.pt')
    (out/'train_log.json').write_text(json.dumps(log, indent=2))
    return net


def evaluate(net: ProbeNet, count: int = 60):
    """Only evaluate after training: the tests are fixed and never used for tuning."""
    results = {}
    with torch.no_grad():
        for split, seed0 in [('heldout',900000),('combined',1200000),('shifted',1500000)]:
            rows = []
            for i in range(count):
                ep=play(seed0+i, net, split=split, deterministic=True)
                rows.append({'seed':seed0+i, 'reward':ep.reward,'quality':ep.quality,
                             'cost':ep.cost, 'num_probes':len(ep.probes),'probes':ep.probes})
            results[split] = {'mean_reward':statistics.fmean(r['reward'] for r in rows),
                              'mean_cost':statistics.fmean(r['cost'] for r in rows),
                              'mean_probes':statistics.fmean(r['num_probes'] for r in rows),
                              'rows':rows}
            print(f'{split:9} learned reward={results[split]["mean_reward"]:.5f}  cost={results[split]["mean_cost"]:.3f} probes={results[split]["mean_probes"]:.2f}',flush=True)
    return results


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--episodes', type=int, default=600)
    cli.add_argument('--batch-size', type=int, default=24)
    cli.add_argument('--seed', type=int, default=20261121)
    cli.add_argument('--output', default='rl_pilot')
    cli.add_argument('--evaluate', default='')
    cli.add_argument('--eval-tasks', type=int, default=60)
    cli.add_argument('--resume', action='store_true')
    cli.add_argument('--train-only', action='store_true')
    opt=cli.parse_args()
    out=Path(opt.output)
    if opt.evaluate:
        c=torch.load(opt.evaluate,map_location='cpu', weights_only=True)
        model=ProbeNet(features=c['features'])
        model.load_state_dict(c['state_dict'])
    else:
        model=learn(opt.episodes,opt.seed,out,batch_size=opt.batch_size,resume=opt.resume)
    if opt.train_only:
        print('Training checkpoint saved. Held-out evaluation not accessed.')
        return
    result=evaluate(model,count=opt.eval_tasks)
    out.mkdir(parents=True,exist_ok=True)
    (out/'eval.json').write_text(json.dumps(result,indent=2))
    print('Saved',str(out/'eval.json'),flush=True)


if __name__ == '__main__': main()
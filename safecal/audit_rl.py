"""Independent audit: learned diagnostic policy versus strong classical and fixed-probe controls."""
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from bayes_id import BayesianID
from safecal import SafeCalEnv

BASE = Path(__file__).parent


def fixed_one_probe(seed, split, name):
    env = SafeCalEnv(seed, combined=split == 'combined', novel=split == 'shifted')
    model = BayesianID(samples=8, seed=seed * 31 + 379)
    model.update(env.step(name))
    outcome=env.finish(model.choose_calibration())
    return {'reward':float(outcome['reward']), 'quality':float(outcome['quality']),
            'cost':float(outcome['test_cost']), 'probes':[name]}


def paired_ci(left,right,seed=101,n_boot=10000):
    arr=[a-b for a,b in zip(left,right)]
    assert len(arr)>1
    rng=random.Random(seed)
    b=sorted(statistics.fmean(arr[rng.randrange(len(arr))] for i in range(len(arr))) for _ in range(n_boot))
    return {'mean_difference':statistics.fmean(arr),
            'bootstrap_95':[b[int(.025*n_boot)],b[int(.975*n_boot)-1]],
            'n':len(arr)}


def audit():
    trained=json.loads((BASE/'rl_pilot_20261121/eval.json').read_text())
    result={}
    for split,s in [('heldout',900000),('combined',1200000),('shifted',1500000)]:
        old=json.loads((BASE/f'results_bayes2_{split}.json').read_text())
        oracle=old['seed_rows']
        learned=trained[split]['rows']
        assert [z['seed'] for z in learned]==[z['seed'] for z in oracle]
        fixed=[fixed_one_probe(s+i,split,'pulse_neg_low') for i in range(len(learned))]
        b2=[z['policies']['bayes2']['reward'] for z in oracle]
        default=[z['policies']['default']['reward'] for z in oracle]
        rl=[z['reward'] for z in learned]
        assert max(abs(a-b['reward']) for a,b in zip(rl,fixed))<1e-12
        result[split]={
            'learned_mean':statistics.fmean(rl),
            'bayes2_mean':statistics.fmean(b2),
            'fixed_neg_low_mean':statistics.fmean(z['reward'] for z in fixed),
            'default_mean':statistics.fmean(default),
            'learned_minus_bayes2':paired_ci(rl,b2,seed=44),
            'learned_minus_default':paired_ci(rl,default,seed=66),
            'learned_minus_fixed':paired_ci(rl,[z['reward'] for z in fixed],seed=55),
            'probe_histogram':{str(k):v for k,v in Counter(tuple(z['probes']) for z in learned).items()},
            'learned_cost':statistics.fmean(z['cost'] for z in learned),
            'bayes2_cost':statistics.fmean(z['policies']['bayes2']['cost'] for z in oracle),
        }
        print(f"{split:9}: RL {result[split]['learned_mean']:.5f}; Bayesian2 {result[split]['bayes2_mean']:.5f}; RL-Bayes {result[split]['learned_minus_bayes2']['mean_difference']:+.5f} CI {result[split]['learned_minus_bayes2']['bootstrap_95']}; RL-fixed = {result[split]['learned_minus_fixed']['mean_difference']:+.10f}")
    (BASE/'rl_pilot_20261121/analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True))
    return result


if __name__=='__main__':audit()
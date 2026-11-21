"""Recompute paired differences and publishable summaries from frozen JSON traces."""
import json
import random
import statistics
from pathlib import Path

ROOT=Path(__file__).parent
FILES={
    'heldout':('results_heldout_v1.json','results_bayes2_heldout.json'),
    'combined':('results_combined_v1.json','results_bayes2_combined.json'),
    'shifted':('results_shifted_v1.json','results_bayes2_shifted.json'),
}

def run():
    out={'protocol':'SafeCal v0.2, simulated-only, seed-paired, no RL training',
         'training_budget_selection':json.loads((ROOT/'budget_selection.json').read_text()),
         'splits':{}}
    for split,(a,b) in FILES.items():
        full=json.loads((ROOT/a).read_text()); selected=json.loads((ROOT/b).read_text())
        assert [r['seed'] for r in full['seed_rows']]==[r['seed'] for r in selected['seed_rows']]
        policies=['default','random','fixed','greedy','bayes','bayes2']
        by_policy={}
        for p in policies:
            src=selected if p=='bayes2' else full
            rows=[row['policies'][p] for row in src['seed_rows']]
            by_policy[p]={k:round(statistics.fmean(r[k] for r in rows),5)
                          for k in ('reward','quality','cost')}
        rng=random.Random(94123)
        comparisons={}
        for left,right in [('bayes2','default'),('bayes2','greedy'),('bayes2','bayes'),('bayes','greedy')]:
            src_l=selected if left=='bayes2' else full
            src_r=selected if right=='bayes2' else full
            differences=[l['policies'][left]['reward']-r['policies'][right]['reward'] for l,r in zip(src_l['seed_rows'],src_r['seed_rows'])]
            draws=sorted(statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences)
                          for _ in range(4000))
            comparisons[f'{left}_minus_{right}']={'mean':round(statistics.fmean(differences),5),
                                                'bootstrap_95':[round(draws[99],5),round(draws[3899],5)]}
        out['splits'][split]={'tasks':len(full['seed_rows']), 'policies':by_policy,
                            'oracle_quality':full['summary']['oracle'], 'comparisons':comparisons}
    (ROOT/'PUBLIC_RESULTS.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    for split,r in out['splits'].items():
        print(split, 'bayes2',r['policies']['bayes2'], 'vs greedy:',r['comparisons']['bayes2_minus_greedy'])

if __name__=='__main__':run()

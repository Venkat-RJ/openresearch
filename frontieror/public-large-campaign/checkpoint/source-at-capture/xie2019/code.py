import argparse,json,os,random,itertools
os.environ['OMP_NUM_THREADS']='1'
from ortools.sat.python import cp_model

def profiles(d):
 rng=random.Random(d.get('random_seed',42));raw=[]
 for typ,count in [('big',d['num_big_profiles']),('small',d['num_small_profiles'])]:
  cfg=d['qc_profiles'][typ];buckets={du:[list(c) for c in itertools.product(range(cfg['min_qc'],cfg['max_qc']+1),repeat=du)] for du in range(cfg['min_handling_shifts'],cfg['max_handling_shifts']+1)};allp=[p for g in buckets.values() for p in g]
  if len(allp)<=count:chosen=allp
  else:
   chosen=[rng.choice(g) for g in buckets.values()];ids={id(c) for c in chosen};rest=[c for c in allp if id(c) not in ids];chosen+=rng.sample(rest,min(max(0,count-len(chosen)),len(rest)))
  raw+=chosen
 return {v['id']:[p for p in raw if len(p)*d['time_steps_per_shift']<=v['b']-v['a']+1] for v in d['vessels']}

def solve(d,limit):
 prof=profiles(d);S=d['time_steps_per_shift'];H=S*d['num_shifts_per_step'];m=cp_model.CpModel();choices={};berth={};qc={};terms=[]
 for v in d['vessels']:
  opts=[]
  for b in d['berths']:
   for p,qs in enumerate(prof[v['id']]):
    for t in range(max(1,v['a'],b['a_k']),min(H,v['b'],b['b_k'])+1):
     first=S-((t-1)%S);seq=[qs[0]]*first+[q for q in qs[1:] for _ in range(S)];end=t+len(seq)-1
     if end>H or end>b['b_k']:continue
     x=m.new_bool_var(f'x{v["id"]}_{b["id"]}_{p}_{t}');opts.append((x,b['id'],p,t,end));terms.append((d['c1']*abs(b['id']-v['k_bar'])+d['c2']*abs(t-v['t_bar']))*x)
     for h in range(t,end+1):berth.setdefault((b['id'],h),[]).append(x)
     for z,q in enumerate(seq):qc.setdefault(t+z,[]).append(q*x)
  if not opts:raise RuntimeError('No legal placement for vessel')
  m.add_exactly_one(x for x,*_ in opts);choices[v['id']]=opts
 for xs in berth.values():m.add(sum(xs)<=1)
 for xs in qc.values():m.add(sum(xs)<=d['total_qc'])
 m.minimize(sum(terms));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,limit-1);st=s.solve(m)
 if st not in (cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError(s.status_name(st))
 out=[]
 for vid,opts in choices.items():
  x,b,p,t,e=next(o for o in opts if s.value(o[0]));out.append({'id':vid,'berth':b,'profile':p,'start_time':t,'end_time':e})
 return {'objective_value':s.objective_value,'vessels':out}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)

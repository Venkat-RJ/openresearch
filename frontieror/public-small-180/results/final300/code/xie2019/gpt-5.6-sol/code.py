import argparse,json,os,time,random,itertools
wall_started=time.time()
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']): del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True)
a=p.parse_args(); end=time.monotonic()+a.time_limit-1; d=json.load(open(a.instance_path)); logger=configured_logger(a.log_path,sense='minimize'); rng=random.Random(d.get('random_seed',42)); S=d['time_steps_per_shift']; H=S*d['num_shifts_per_step']; vessels=d['vessels']; berths=d['berths']; best=float('inf')
def generate(cfg,count):
 buckets={dur:[{'duration_shifts':dur,'qc_per_shift':list(combo)} for combo in itertools.product(range(cfg['min_qc'],cfg['max_qc']+1),repeat=dur)] for dur in range(cfg['min_handling_shifts'],cfg['max_handling_shifts']+1)}; allp=[v for group in buckets.values() for v in group]
 if len(allp)<=count: return allp
 selected=[rng.choice(group) for group in buckets.values()]; ids={id(v) for v in selected}; rest=[v for v in allp if id(v) not in ids]; selected+=rng.sample(rest,min(max(0,count-len(selected)),len(rest))); return selected
raw=generate(d['qc_profiles']['big'],d['num_big_profiles'])+generate(d['qc_profiles']['small'],d['num_small_profiles']); options={}
for v in vessels:
 ps=[p for p in raw if p['duration_shifts']*S<=v['b']-v['a']+1]; opts=[]
 for t in range(v['a'],v['b']+1):
  partial=S-(t-1)%S; plans=[]
  for pi,pr in enumerate(ps):
   q=[pr['qc_per_shift'][0]]*partial+[qc for qc in pr['qc_per_shift'][1:] for _ in range(S)]; finish=t+len(q)-1
   if finish>H: continue
   if any(len(qq)<=len(q) and all(a<=b for a,b in zip(qq,q)) for ppi,qq in plans): continue
   plans=[(ppi,qq) for ppi,qq in plans if not(len(q)<=len(qq) and all(a<=b for a,b in zip(q,qq)))]; plans.append((pi,q))
  for pi,q in plans:
   for b in berths:
    if t>=b['a_k'] and t+len(q)-1<=b['b_k']: opts.append((d['c1']*abs(b['id']-v['k_bar'])+d['c2']*abs(t-v['t_bar']),b['id'],t,pi,q))
 options[v['id']]=sorted(opts,key=lambda o:(o[0],sum(o[4]),len(o[4])))
def emit(chosen):
 global best
 obj=sum(o[0] for o in chosen.values())
 if obj<best:
  best=obj; out={'objective_value':obj,'vessels':[{'id':i,'berth':o[1],'start_time':o[2],'end_time':o[2]+len(o[4])-1,'profile':o[3]} for i,o in chosen.items()]}; json.dump(out,open(a.solution_path,'w')); logger.log_solution(obj,out)
rr=random.Random(0)
for attempt in range(100):
 if time.monotonic()>=end: break
 order=sorted(vessels,key=lambda v:v['a']) if attempt==0 else rr.sample(vessels,len(vessels)); occupied=set(); usage=[0]*(H+1); chosen={}
 for v in order:
  candidates=[o for o in options[v['id']] if all((o[1],o[2]+u) not in occupied and usage[o[2]+u]+q<=d['total_qc'] for u,q in enumerate(o[4]))]
  if not candidates: break
  o=rr.choice(candidates[:min(3,len(candidates))]) if attempt else candidates[0]; chosen[v['id']]=o
  for u,q in enumerate(o[4]): occupied.add((o[1],o[2]+u)); usage[o[2]+u]+=q
 if len(chosen)==len(vessels): emit(chosen)
m=gp.Model(); m.Params.Threads=1; m.Params.Seed=0; m.Params.MIPGap=1e-4; m.Params.NumericFocus=0
limit=max(1,1800//len(vessels)); opts=[(v['id'],o) for v in vessels for o in options[v['id']][:limit]]; z=[m.addVar(vtype=GRB.BINARY,obj=o[0]) for i,o in opts]
for v in vessels: m.addConstr(gp.quicksum(z[j] for j,(i,o) in enumerate(opts) if i==v['id'])==1)
for b in berths:
 for t in range(1,H+1): m.addConstr(gp.quicksum(z[j] for j,(i,o) in enumerate(opts) if o[1]==b['id'] and o[2]<=t<o[2]+len(o[4]))<=1)
for t in range(1,H+1): m.addConstr(gp.quicksum(o[4][t-o[2]]*z[j] for j,(i,o) in enumerate(opts) if o[2]<=t<o[2]+len(o[4]))<=d['total_qc'])
m.update(); assert m.NumVars<=2000 and m.NumConstrs<=2000
def cb(model,where):
 if where==GRB.Callback.MIPSOL:
  vals=model.cbGetSolution(z); emit({i:o for (i,o),v in zip(opts,vals) if v>.5})
m.Params.TimeLimit=max(.1,end-time.monotonic()); m.optimize(cb)
if m.SolCount: emit({i:o for (i,o),v in zip(opts,z) if v.X>.5})

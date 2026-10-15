import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,math
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));items=[i for i in d['items'] if i['type']=='treatment'];T=d['parameters']['T'];W=d['parameters']['W'];L=d['parameters']['L'];L=len(items) if L=='infinity' else int(L);L=min(L,len(items));model=cp_model.CpModel();period={d['source_item_id']:model.new_constant(0),d['sink_item_id']:model.new_constant(T+1)};es={i['item_id']:1 for i in items};ls={i['item_id']:T for i in items};es[d['source_item_id']]=ls[d['source_item_id']]=0;es[d['sink_item_id']]=ls[d['sink_item_id']]=T+1
for _ in range(len(d['items'])):
 for arc in d['arcs']:
  u,v,l=arc['from'],arc['to'],arc['lag'];es[v]=max(es[v],es[u]+l);ls[u]=min(ls[u],ls[v]-l)
xs={};ys={(b,t):model.new_bool_var(f'y{b}_{t}') for b in range(1,L+1) for t in range(1,T+1)};bucket={k:[] for k in ys}
for it in items:
 i=it['item_id'];period[i]=model.new_int_var(es[i],ls[i],f'p{i}');choices=[]
 for t in range(es[i],ls[i]+1):
  for b in range(1,L+1):
   x=model.new_bool_var(f'x{i}_{b}_{t}');xs[i,b,t]=x;choices.append(x);bucket[b,t].append((it['weight'],x));model.add(period[i]==t).only_enforce_if(x)
 model.add(sum(choices)==1)
for arc in d['arcs']:model.add(period[arc['to']]>=period[arc['from']]+arc['lag'])
for (b,t),y in ys.items():
 model.add(sum(w*x for w,x in bucket[b,t])<=W*y)
 if b<L:model.add(y>=ys[b+1,t])
model.minimize(sum(ys.values()));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;weights={it['item_id']:it['weight'] for it in items}
class CB(cp_model.CpSolverSolutionCallback):
 def on_solution_callback(self):
  assignments=[];bins={}
  for (i,b,t),x in xs.items():
   if self.value(x):assignments.append({'item_id':i,'bin':b,'period':t,'weight':weights[i]});bins.setdefault((b,t),[]).append(i)
  active=[{'bin':b,'period':t,'items':ids,'total_weight':sum(weights[i] for i in ids)} for (b,t),ids in bins.items()];s={'objective_value':len(active),'assignments':assignments,'active_bin_periods':active,'num_active_bin_periods':len(active)};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(len(active),s)
solver.solve(model,CB())

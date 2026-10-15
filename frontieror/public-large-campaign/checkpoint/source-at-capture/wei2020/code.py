import argparse,json,time,os,random,math
os.environ['OMP_NUM_THREADS']='1'
from ortools.sat.python import cp_model

def solve(d,limit):
 vector='items' in d
 w=[x['w'] for x in d['items']] if vector else d['weights'];v=[x['v'] for x in d['items']] if vector else [0]*len(w);W=d['W'] if vector else d['c'];V=d['V'] if vector else 1;n=len(w)
 deadline=time.monotonic()+max(.01,limit-1)
 best=None;rng=random.Random(0)
 for rep in range(40):
  order=sorted(range(n),key=lambda i:-(w[i]/W+v[i]/V)*(1 if rep==0 else rng.uniform(.7,1.3)));bins=[];loads=[]
  for i in order:
   feasible=[k for k,(a,b) in enumerate(loads) if a+w[i]<=W and b+v[i]<=V]
   if feasible:k=max(feasible,key=lambda k:loads[k][0]/W+loads[k][1]/V)
   else:k=len(bins);bins.append([]);loads.append([0,0])
   bins[k].append(i);loads[k][0]+=w[i];loads[k][1]+=v[i]
  if best is None or len(bins)<len(best):best=bins
  if time.monotonic()>deadline:break
 if n*len(best)<60000 and time.monotonic()<deadline:
  model=cp_model.CpModel();B=len(best);x=[[model.new_bool_var(f'x{i}_{k}') for k in range(B)] for i in range(n)];y=[model.new_bool_var(f'y{k}') for k in range(B)]
  for i in range(n):model.add_exactly_one(x[i])
  for k in range(B):
   model.add(sum(w[i]*x[i][k] for i in range(n))<=W*y[k]);model.add(sum(v[i]*x[i][k] for i in range(n))<=V*y[k])
   if k:model.add(y[k]<=y[k-1])
  model.add(sum(y)>=max(math.ceil(sum(w)/W),math.ceil(sum(v)/V)))
  for k,g in enumerate(best):
   model.add_hint(y[k],1);s=set(g)
   for i in range(n):model.add_hint(x[i][k],int(i in s))
  model.minimize(sum(y));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());status=s.solve(model)
  if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):best=[[i for i in range(n) if s.value(x[i][k])] for k in range(B) if s.value(y[k])]
 return {'objective_value':len(best),'bins':[{'items':g,'total_weight':sum(w[i] for i in g),'total_volume':sum(v[i] for i in g)} for g in best] if vector else best}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)

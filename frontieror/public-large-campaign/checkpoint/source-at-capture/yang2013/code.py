import argparse,json,os,time,random
os.environ['OMP_NUM_THREADS']='1'
from ortools.sat.python import cp_model

def solve(d,limit):
 n=d['n'];w=d['weights'];C=d['capacity'];p=d['profit_matrix'];deadline=time.monotonic()+max(.01,limit-1);rng=random.Random(0);best=[];bv=-1
 def value(c):return sum(p[i][i] for i in c)+sum(p[i][j] for k,i in enumerate(c) for j in c[:k])
 for rep in range(100):
  c=[];left=C;available=set(range(n))
  while available:
   fits=[i for i in available if w[i]<=left]
   if not fits:break
   i=max(fits,key=lambda i:(p[i][i]+sum(p[i][j] for j in c))/w[i]*(1 if rep==0 else rng.uniform(.6,1.4)));available.remove(i);c.append(i);left-=w[i]
  z=value(c)
  if z>bv:best,bv=c,z
  if time.monotonic()>deadline:break
 if n<=500 and time.monotonic()<deadline:
  m=cp_model.CpModel();x=[m.new_bool_var(f'x{i}') for i in range(n)];m.add(sum(w[i]*x[i] for i in range(n))<=C);terms=[p[i][i]*x[i] for i in range(n)]
  for i in range(n):
   for j in range(i):
    if p[i][j]:y=m.new_bool_var(f'y{i}_{j}');m.add(y<=x[i]);m.add(y<=x[j]);m.add(y>=x[i]+x[j]-1);terms.append(p[i][j]*y)
  m.maximize(sum(terms));m.add(sum(terms)>=bv)
  for i in range(n):m.add_hint(x[i],int(i in best))
  s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());st=s.solve(m)
  if st in (cp_model.FEASIBLE,cp_model.OPTIMAL):best=[i for i in range(n) if s.value(x[i])];bv=value(best)
 return {'objective_value':bv,'solution':[int(i in best) for i in range(n)]}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)

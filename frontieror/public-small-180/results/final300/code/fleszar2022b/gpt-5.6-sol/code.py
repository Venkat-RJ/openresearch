import argparse,json,time,random,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-4)
 z=json.load(open(a.instance_path));n,m=z['n'],z['m'];w=np.array(z['weights']);v=np.array(z['values']);q=np.array(z['pairwise_values']);np.fill_diagonal(q,0);cap=z['capacity'];logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None;bestval=-10**30;best=None;rng=random.Random(0)
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 def cost(t):return int(sum(v[t>=0])+sum(q[i,j] for i in range(n) for j in range(i+1,n) if t[i]>=0 and t[i]==t[j]))
 def save(t):
  nonlocal bestval,best
  val=cost(t)
  if val>bestval:
   bestval=val;best=t.copy();sol={'objective_value':val,'assignment':{str(i):int(k) for i,k in enumerate(t) if k>=0},'runtime':time.monotonic()-start};json.dump(sol,open(a.solution_path,'w'))
   if logger:logger.log_solution(val,sol)
 def descent(t):
  while time.monotonic()<end:
   sets=np.array([t==k for k in range(m)],dtype=int);load=sets@w;gains=q@sets.T+v[:,None];old=np.array([gains[i,k] if k>=0 else 0 for i,k in enumerate(t)]);delta=gains-old[:,None];delta[load[None,:]+w[:,None]>cap]=-10**15
   for i,k in enumerate(t):
    if k>=0:delta[i,k]=0
   i,k=np.unravel_index(np.argmax(delta),delta.shape);drop=int(np.argmax(-old))
   if -old[drop]>delta[i,k] and old[drop]<0:t[drop]=-1
   elif delta[i,k]>0:t[i]=k
   else:break
   save(t)
  return t
 t=np.full(n,-1,dtype=int);save(t);descent(t)
 if n<=22:
  lic=os.environ.get('GRB_LICENSE_FILE')
  if lic and not os.path.exists(lic):os.environ.pop('GRB_LICENSE_FILE')
  import gurobipy as gp
  model=gp.Model();model.Params.Seed=0;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;model.Params.Threads=1;model.Params.TimeLimit=max(.1,end-time.monotonic());model.Params.OutputFlag=0;x=model.addVars(n,m,vtype=gp.GRB.BINARY);pairs=[(i,j,k) for i in range(n) for j in range(i+1,n) for k in range(m) if q[i,j]!=0];y=model.addVars(pairs,vtype=gp.GRB.CONTINUOUS,lb=0,ub=1)
  for i in range(n):model.addConstr(gp.quicksum(x[i,k] for k in range(m))<=1)
  for k in range(m):model.addConstr(gp.quicksum(int(w[i])*x[i,k] for i in range(n))<=cap)
  for i,j,k in pairs:model.addConstr(y[i,j,k]<=x[i,k]);model.addConstr(y[i,j,k]<=x[j,k]);model.addConstr(y[i,j,k]>=x[i,k]+x[j,k]-1)
  model.setObjective(gp.quicksum(int(v[i])*x[i,k] for i in range(n) for k in range(m))+gp.quicksum(int(q[i,j])*y[i,j,k] for i,j,k in pairs),gp.GRB.MAXIMIZE)
  for i in range(n):
   for k in range(m):x[i,k].Start=int(t[i]==k)
  def cb(md,where):
   if where==gp.GRB.Callback.MIPSOL:
    vals=md.cbGetSolution(x);s=np.full(n,-1,dtype=int)
    for i in range(n):
     for k in range(m):
      if vals[i,k]>.5:s[i]=k
    save(s)
  model.optimize(cb);return
 while time.monotonic()<end:
  t=best.copy();t[rng.sample(range(n),rng.randint(3,12))]=-1;descent(t);save(t)
if __name__=='__main__':main()

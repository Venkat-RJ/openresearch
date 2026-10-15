import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,itertools,time
import highspy
from solution_logger import SolutionLogger

def main():
 start=time.monotonic()
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','log_path'):p.add_argument('--'+k,required=True)
 p.add_argument('--time_limit',type=int,default=60);a=p.parse_args();d=json.load(open(a.instance_path));n=d['n'];T=d['T'];m=d['m'];S=d['n_scenarios'];Q=d['Q'];C=d['C'];L=d['L'];hcost=d['h'];dem=d['demand_scenarios'];rho=d['scenario_probabilities'];logger=SolutionLogger(a.log_path,sense='minimize')
 baseline={};baseline_cost=0.;feasible=d['I0'][0]<=L[0]
 for w in range(S):
  inventory=list(map(float,d['I0']))
  for t in range(1,T+1):
   if inventory[0]:baseline[f'I_0_{t}_{w}']=inventory[0]
   baseline_cost+=rho[w]*hcost[0]*inventory[0]
   for i in range(1,n+1):
    demand=dem[w][i-1][t-1];emergency=max(0.,demand-inventory[i]);inventory[i]=max(0.,inventory[i]-demand)
    feasible=feasible and inventory[i]+demand<=L[i]+1e-8
    if emergency:baseline[f'e_{i}_{t}_{w}']=emergency
    if inventory[i]:baseline[f'I_{i}_{t}_{w}']=inventory[i]
    baseline_cost+=rho[w]*(hcost[i]*inventory[i]+d['sigma'][i-1]*emergency)
 if feasible:
  sol={'objective_value':baseline_cost,'model_variables':baseline,'y':{},'z':{},'x':{}};json.dump(sol,open(a.solution_path,'w'))
  if hasattr(logger,'log_solution'):logger.log_solution(baseline_cost,sol)
  else:logger.log(baseline_cost)
 print('BUILD_START elapsed',time.monotonic()-start,'scenarios',S,flush=True)
 h=highspy.Highs();h.setOptionValue('threads',1);h.setOptionValue('time_limit',float(a.time_limit));h.setOptionValue('output_flag',False);variables={};costs={}
 def var(name,lb=0,ub=highspy.kHighsInf,cost=0,integer=False):
  v=h.addVariable(lb=lb,ub=ub,obj=cost,type=highspy.HighsVarType.kInteger if integer else highspy.HighsVarType.kContinuous);variables[name]=v;costs[name]=cost;return v
 y={t:var('y_'+str(t),ub=1,cost=d['f'],integer=True) for t in range(1,T+1)};z={};x={};edges=list(itertools.combinations(range(n+1),2));q={};I={};P={};E={}
 for t in range(1,T+1):
  for k in range(1,m+1):
   for i in range(n+1):z[i,k,t]=var(f'z_{i}_{k}_{t}',ub=1,integer=True)
   for i,j in edges:x[i,j,k,t]=var(f'x_{i}_{j}_{k}_{t}',ub=2 if i==0 else 1,cost=d['transportation_costs'][i][j],integer=True)
   for i in range(n+1):h.addConstr(sum(x[min(i,j),max(i,j),k,t] for j in range(n+1) if j!=i)==2*z[i,k,t])
   flow={}
   for i in range(n+1):
    for j in range(n+1):
     if i==j:continue
     flow[i,j]=h.addVariable(lb=0,ub=n);h.addConstr(flow[i,j]<=n*x[min(i,j),max(i,j),k,t])
   for i in range(1,n+1):
    h.addConstr(sum(flow[j,i]-flow[i,j] for j in range(n+1) if j!=i)==z[i,k,t]);h.addConstr(z[i,k,t]<=z[0,k,t])
   for i,j in edges:
    if i:h.addConstr(x[i,j,k,t]<=z[i,k,t]);h.addConstr(x[i,j,k,t]<=z[j,k,t])
  for i in range(1,n+1):h.addConstr(sum(z[i,k,t] for k in range(1,m+1))<=1)
  for k in range(1,m):
   h.addConstr(z[0,k,t]>=z[0,k+1,t])
   for j in range(1,n+1):h.addConstr(sum(2.**(j-i)*(z[i,k,t]-z[i,k+1,t]) for i in range(1,j+1))>=0)
 for w in range(S):
  if time.monotonic()-start>=a.time_limit-5.:
   print('BUILD_DEADLINE scenarios_built',w,'of',S,flush=True);return
  for i in range(n+1):I[i,0,w]=float(d['I0'][i])
  for t in range(1,T+1):
   production_limit=min(C,Q,sum(sum(dem[w][i][t-1:]) for i in range(n)));P[t,w]=var(f'p_{t}_{w}',cost=rho[w]*d['u']);h.addConstr(P[t,w]<=production_limit*y[t])
   for i in range(n+1):
    upper=L[i] if i==0 else L[i]-dem[w][i-1][t-1];I[i,t,w]=var(f'I_{i}_{t}_{w}',ub=max(0,upper),cost=rho[w]*hcost[i])
   for i in range(1,n+1):
    E[i,t,w]=var(f'e_{i}_{t}_{w}',cost=rho[w]*d['sigma'][i-1])
    for k in range(1,m+1):q[i,k,t,w]=var(f'q_{i}_{k}_{t}_{w}');h.addConstr(q[i,k,t,w]<=min(L[i],Q,sum(dem[w][i-1][t-1:]))*z[i,k,t])
    h.addConstr(I[i,t-1,w]+sum(q[i,k,t,w] for k in range(1,m+1))+E[i,t,w]-I[i,t,w]==dem[w][i-1][t-1])
    if upper<0:raise ValueError('demand exceeds customer storage')
   h.addConstr(I[0,t-1,w]+P[t,w]-I[0,t,w]==sum(q[i,k,t,w] for i in range(1,n+1) for k in range(1,m+1)))
   for k in range(1,m+1):h.addConstr(sum(q[i,k,t,w] for i in range(1,n+1))<=Q*z[0,k,t])
 remaining=a.time_limit-(time.monotonic()-start)-5.
 print('BUILD_END elapsed',time.monotonic()-start,'solve_budget',remaining,flush=True)
 if remaining<=0:return
 h.setOptionValue('time_limit',remaining);h.run()
 print('SOLVE_END elapsed',time.monotonic()-start,flush=True)
 if h.getSolution().value_valid:
  vals=h.getSolution().col_value;mv={name:float(vals[int(v)]) for name,v in variables.items() if abs(vals[int(v)])>1e-10};obj=sum(costs[name]*value for name,value in mv.items())
  if feasible and obj>=baseline_cost:return
  sol={'objective_value':obj,'model_variables':mv,'y':{str(t):round(vals[int(v)]) for t,v in y.items()},'z':{'_'.join(map(str,key)):round(vals[int(v)]) for key,v in z.items()},'x':{'_'.join(map(str,key)):round(vals[int(v)]) for key,v in x.items()}};json.dump(sol,open(a.solution_path,'w'))
  if hasattr(logger,'log_solution'):logger.log_solution(obj,sol)
  else:logger.log(obj)
if __name__=='__main__':main()

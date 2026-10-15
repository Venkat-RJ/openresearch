import argparse,json,time,os,itertools
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def config(m):
 m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0

def solve(data,deadline,logger):
 n,T=data['n'],data['T'];Q,C=data['Q'],data['C'];cost=data['transportation_costs'];rho=data['scenario_probabilities'];demand=data['demand_scenarios'];S=data['n_scenarios'];I0=data['I0'];L=data['L'];h=data['h'];sigma=data['sigma'];unit=data['u'];fixed=data['f'];G=min(10,S);masks=range(1<<n);tours={0:[0]};routecost={0:0}
 if data['m']!=1:raise RuntimeError('route-subset decomposition requires single-vehicle instance')
 for mask in list(masks)[1:]:
  items=[i+1 for i in range(n) if mask>>i&1];tour=min(itertools.permutations(items),key=lambda p:cost[0][p[0]]+sum(cost[a][b] for a,b in zip(p,p[1:]))+cost[p[-1]][0]);tours[mask]=[0]+list(tour);routecost[mask]=cost[0][tour[0]]+sum(cost[a][b] for a,b in zip(tour,tour[1:]))+cost[tour[-1]][0]
 master=gp.Model();config(master);a=master.addVars(T,list(masks),vtype=GRB.BINARY);y=master.addVars(T,vtype=GRB.BINARY);theta=master.addVars(G)
 for t in range(T):master.addConstr(gp.quicksum(a[t,mask] for mask in masks)==1)
 z={(i,t):gp.quicksum(a[t,mask] for mask in masks if (mask!=0 if i==0 else mask>>(i-1)&1)) for i in range(n+1) for t in range(T)};master.setObjective(gp.quicksum(fixed*y[t]+gp.quicksum(routecost[mask]*a[t,mask] for mask in masks) for t in range(T))+theta.sum())
 lp=gp.Model();config(lp);lp.Params.Method=1;p=lp.addVars(T);q=lp.addVars(n,T);inv=lp.addVars(n+1,T);em=lp.addVars(n,T);plant=[];customer={};prod=[];vehicle=[];delivery={};upper={}
 for t in range(T):
  plant.append(lp.addConstr(inv[0,t]-p[t]+gp.quicksum(q[i,t] for i in range(n))-(inv[0,t-1] if t else 0)==(I0[0] if t==0 else 0)));prod.append(lp.addConstr(p[t]<=0));vehicle.append(lp.addConstr(gp.quicksum(q[i,t] for i in range(n))<=0));inv[0,t].UB=L[0]
  for i in range(n):
   customer[i,t]=lp.addConstr(inv[i+1,t]-q[i,t]-em[i,t]-(inv[i+1,t-1] if t else 0)==0);delivery[i,t]=lp.addConstr(q[i,t]<=0);upper[i,t]=lp.addConstr(inv[i+1,t]<=L[i+1])
 lp.setObjective(gp.quicksum(unit*p[t]+gp.quicksum(h[i]*inv[i,t] for i in range(n+1))+gp.quicksum(sigma[i]*em[i,t] for i in range(n)) for t in range(T)))
 chosen=[(1<<n)-1]*T;setup=[1]*T;best=None;seen=set();duration=0
 for iteration in range(150):
  tick=time.monotonic();values=[0.]*G;constants=[0.]*G;py=[[0.]*T for _ in range(G)];pz=[{(i,t):0. for i in range(n+1) for t in range(T)} for _ in range(G)];witness={};complete=True
  for w in range(S):
   if time.monotonic()>=deadline:complete=False;break
   mt=[min(C,Q,sum(sum(demand[w][i][t:]) for i in range(n))) for t in range(T)];mp={(i,t):min(L[i+1],Q,sum(demand[w][i][t:])) for i in range(n) for t in range(T)}
   for t in range(T):
    prod[t].RHS=mt[t]*setup[t];vehicle[t].RHS=Q*int(chosen[t]!=0)
    for i in range(n):customer[i,t].RHS=(I0[i+1] if t==0 else 0)-demand[w][i][t];upper[i,t].RHS=L[i+1]-demand[w][i][t];delivery[i,t].RHS=mp[i,t]*int(chosen[t]>>i&1)
   lp.Params.TimeLimit=max(.01,deadline-time.monotonic());lp.optimize()
   if lp.Status!=GRB.OPTIMAL:complete=False;break
   g=w%G;prob=rho[w];values[g]+=prob*lp.ObjVal;linear=0.
   for t in range(T):
    cy=prod[t].Pi*mt[t];py[g][t]+=prob*cy;linear+=cy*setup[t];cz=vehicle[t].Pi*Q;pz[g][0,t]+=prob*cz;linear+=cz*int(chosen[t]!=0)
    if p[t].X>1e-10:witness[f'p_{t+1}_{w}']=p[t].X
    for i in range(n+1):
     if inv[i,t].X>1e-10:witness[f'I_{i}_{t+1}_{w}']=inv[i,t].X
    for i in range(n):
     cz=delivery[i,t].Pi*mp[i,t];pz[g][i+1,t]+=prob*cz;linear+=cz*int(chosen[t]>>i&1)
     if q[i,t].X>1e-10:witness[f'q_{i+1}_1_{t+1}_{w}']=q[i,t].X
     if em[i,t].X>1e-10:witness[f'e_{i+1}_{t+1}_{w}']=em[i,t].X
   constants[g]+=prob*(lp.ObjVal-linear)
  if not complete:break
  obj=fixed*sum(setup)+sum(routecost[mask] for mask in chosen)+sum(values)
  if best is None or obj<best['objective_value']-1e-7:
   result={'objective_value':obj,'y':{str(t+1):setup[t] for t in range(T)},'z':{},'x':{},'model_variables':witness}
   for t,mask in enumerate(chosen):
    witness[f'y_{t+1}']=setup[t];tour=tours[mask]
    if mask==0:continue
    for i in tour:result['z'][f'{i}_1_{t+1}']=1;witness[f'z_{i}_1_{t+1}']=1
    for i,j in zip(tour,tour[1:]+tour[:1]):
     i,j=sorted((i,j));key=f'{i}_{j}_1_{t+1}';result['x'][key]=result['x'].get(key,0)+1
   for key,value in result['x'].items():witness['x_'+key]=value
   best=result
   if logger:logger.log_solution(obj,best)
  for g in range(G):master.addConstr(theta[g]>=constants[g]+gp.quicksum(py[g][t]*y[t] for t in range(T))+gp.quicksum(pz[g][i,t]*z[i,t] for i in range(n+1) for t in range(T)))
  duration=max(duration,time.monotonic()-tick);seen.add(tuple(chosen+setup))
  if deadline-time.monotonic()<duration+.3:break
  for t in range(T):
   y[t].Start=setup[t]
   for mask in masks:a[t,mask].Start=int(mask==chosen[t])
  master.Params.TimeLimit=max(.01,min(5.,deadline-time.monotonic()-duration));master.optimize()
  if not master.SolCount:break
  if best['objective_value']-master.ObjBound<=1e-4*max(1,abs(best['objective_value'])):break
  chosen=[max(masks,key=lambda mask:a[t,mask].X) for t in range(T)];setup=[int(y[t].X>.5) for t in range(T)]
  if tuple(chosen+setup) in seen:break
 if best is None:raise RuntimeError('no complete stochastic routing witness')
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

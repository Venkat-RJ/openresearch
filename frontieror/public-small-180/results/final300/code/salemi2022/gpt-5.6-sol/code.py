import time
_wall_start=time.time()
import argparse,json,os
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();deadline=start+a.time_limit-.3;d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
G=d['generators'];I=range(d['n_generators']);T=range(d['T']);S=range(d['n_scenarios']);D=d['demand_scenarios'];R=d['reserve_scenarios'];worst=[max(D[s][t] for s in S) for t in T];reserve=[max(D[s][t]+R[s][t] for s in S) for t in T]
def model():
 m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1;return m
m=model();x=m.addVars(I,T,vtype=GRB.BINARY);y=m.addVars(I,T,vtype=GRB.BINARY);yb=m.addVars(I,T,vtype=GRB.BINARY);pw=m.addVars(I,T,lb=0)
for i in I:
 g=G[i]
 for t in T:
  prev=x[i,t-1] if t else 0;pp=pw[i,t-1] if t else 0
  m.addConstr(y[i,t]-yb[i,t]==x[i,t]-prev);m.addConstr(pw[i,t]>=g['m']*x[i,t]);m.addConstr(pw[i,t]<=g['M']*x[i,t]);m.addConstr(pw[i,t]-pp<=g['RU']*prev+g['SU']*y[i,t]);m.addConstr(pp-pw[i,t]<=g['RD']*x[i,t]+g['SD']*yb[i,t])
 for t in range(g['L']-1,d['T']):m.addConstr(gp.quicksum(y[i,j] for j in range(t-g['L']+1,t+1))<=x[i,t])
 for t in range(g['ell']-1,d['T']):m.addConstr(gp.quicksum(yb[i,j] for j in range(t-g['ell']+1,t+1))<=1-x[i,t])
for t in T:m.addConstr(gp.quicksum(pw[i,t] for i in I)>=worst[t]);m.addConstr(gp.quicksum(G[i]['M']*x[i,t] for i in I)>=reserve[t])
m.setObjective(gp.quicksum(G[i]['c_f']*x[i,t]+(max(G[i]['startup_costs_K']) if t else 0)*y[i,t]+G[i]['c_g']*pw[i,t] for i in I for t in T))
best=float('inf');solution=None
def pack(v):
 xx=[[int(v(x[i,t])>.5) for t in T] for i in I];yy=[[int(v(y[i,t])>.5) for t in T] for i in I];bb=[[int(v(yb[i,t])>.5) for t in T] for i in I];q=[[max([0]+[G[i]['startup_costs_K'][k-1]*(xx[i][t]-sum(xx[i][t-k:t])) for k in range(1,min(t,len(G[i]['startup_costs_K']))+1)]) for t in T] for i in I];prod=[[[max(0,v(pw[i,t])) for t in T] for s in S] for i in I];cap=[[[G[i]['M']*xx[i][t] for t in T] for s in S] for i in I]
 return {'x':xx,'y':yy,'y_bar':bb,'q':q,'p':prod,'p_bar':cap}
def publish(s):
 global best,solution
 obj=sum(G[i]['c_f']*s['x'][i][t]+s['q'][i][t] for i in I for t in T)+sum(G[i]['c_g']*s['p'][i][sc][t] for i in I for sc in S for t in T)/len(S);s['objective_value']=obj
 if obj<best:
  best=obj;solution=s
  if log:log.log_solution(obj,s)
def cb(mm,where):
 if where==GRB.Callback.MIPSOL:publish(pack(mm.cbGetSolution))
m.Params.TimeLimit=max(.01,min(a.time_limit*.35,deadline-time.monotonic()));m.optimize(cb)
if not m.SolCount:raise RuntimeError('No robust feasible commitment')
s=pack(lambda z:z.X);publish(s)
for sc in S:
 if time.monotonic()>=deadline:break
 rec=model();p=rec.addVars(I,T,lb={(i,t):G[i]['m']*s['x'][i][t] for i in I for t in T},ub={(i,t):G[i]['M']*s['x'][i][t] for i in I for t in T})
 for i in I:
  for t in T:
   pp=p[i,t-1] if t else 0;xp=s['x'][i][t-1] if t else 0;rec.addConstr(p[i,t]-pp<=G[i]['RU']*xp+G[i]['SU']*s['y'][i][t]);rec.addConstr(pp-p[i,t]<=G[i]['RD']*s['x'][i][t]+G[i]['SD']*s['y_bar'][i][t]);p[i,t].Start=s['p'][i][sc][t]
 for t in T:rec.addConstr(gp.quicksum(p[i,t] for i in I)>=D[sc][t])
 rec.setObjective(gp.quicksum(G[i]['c_g']*p[i,t] for i in I for t in T));rec.Params.TimeLimit=max(.01,deadline-time.monotonic());rec.optimize()
 if rec.SolCount:
  for i in I:s['p'][i][sc]=[max(0,p[i,t].X) for t in T]
  publish(json.loads(json.dumps(s)))
 rec.dispose()
solution['runtime']=time.monotonic()-start;json.dump(solution,open(a.solution_path,'w'))

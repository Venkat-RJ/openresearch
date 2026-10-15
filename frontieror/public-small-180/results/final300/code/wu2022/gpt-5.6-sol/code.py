import argparse,json,os,time
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
a=p.parse_args(); start=time.monotonic(); d=json.load(open(a.instance_path)); logger=configured_logger(a.log_path,sense='minimize'); dims=d['dimensions']; J=dims['num_items']; T=dims['num_periods']; K=dims['num_plants']; C=dims['num_customers']; caps=d['plant_capabilities']; customers=[[c for c in range(C) if j in d['customer_item_demands'][c]] for j in range(J)]; td=[[sum(d['demand'][j][c][t] for c in customers[j]) for t in range(T)] for j in range(J)]
sd={(j,t,l):sum(td[j][t:l+1]) for j in range(J) for t in range(T) for l in range(t,T)}
cv={(j,k,t,l):d['production_cost'][j][k][t]*sd[j,t,l]+sum(sum(d['holding_cost'][j][k][u] for u in range(t,s))*td[j][s] for s in range(t+1,l+1)) for j in range(J) for k in caps[j] for t in range(T) for l in range(t,T)}
trans={(j,k):sum(d['demand'][j][c][t]*d['transportation_cost'][j][c][k][t] for c in customers[j] for t in range(T)) for j in range(J) for k in caps[j]}
m=gp.Model(); m.Params.Threads=1; m.Params.Seed=0; m.Params.MIPGap=1e-4; m.Params.NumericFocus=0
Z={k:m.addVar(vtype=GRB.BINARY,obj=d['plant_opening_cost'][k]) for k in range(K)}; U={(j,k):m.addVar(vtype=GRB.BINARY,obj=trans[j,k]) for j in range(J) for k in caps[j]}; Y={(j,k,t):m.addVar(vtype=GRB.BINARY,obj=d['setup_cost'][j][k][t]) for j,k in U for t in range(T)}; W={key:m.addVar(obj=c,ub=1) for key,c in cv.items()}
for j in range(J):
 m.addConstr(gp.quicksum(U[j,k] for k in caps[j])==1); m.addConstr(gp.quicksum(W[j,k,0,l] for k in caps[j] for l in range(T))==1)
 for k in caps[j]:
  m.addConstr(U[j,k]<=Z[k])
  for t in range(T):
   m.addConstr(Y[j,k,t]<=U[j,k]); m.addConstr(gp.quicksum(W[j,k,t,l] for l in range(t,T))<=Y[j,k,t])
   if t: m.addConstr(gp.quicksum(W[j,k,u,t-1] for u in range(t))==gp.quicksum(W[j,k,t,l] for l in range(t,T)))
for k in range(K):
 for t in range(T): m.addConstr(gp.quicksum(d['setup_time'][j][k][t]*Y[j,k,t]+d['production_time'][j][k][t]*gp.quicksum(sd[j,t,l]*W[j,k,t,l] for l in range(t,T)) for j in range(J) if k in caps[j])<=d['capacity'][k][t]*Z[k])
m.update(); assert m.NumVars<=2000 and m.NumConstrs<=2000
binary=list(Z.values())+list(U.values())+list(Y.values()); completion=m.copy(); completion.Params.Threads=1; completion.Params.Seed=0; completion.Params.MIPGap=1e-4; completion.Params.NumericFocus=0; completion.Params.OutputFlag=0; cvvars=completion.getVars()
for var in cvvars: var.VType=GRB.CONTINUOUS
fixed=[cvvars[var.index] for var in binary]; priced={key:cvvars[var.index] for key,var in W.items()}; seen=set(); best=float('inf')
def emit(values,wcost):
 global best
 zs={str(k):int(v>.5) for k,v in zip(Z,values[:len(Z)])}; us={f'{j}_{k}':int(v>.5) for (j,k),v in zip(U,values[len(Z):len(Z)+len(U)])}; ys={f'{j}_{k}_{t}':int(v>.5) for (j,k,t),v in zip(Y,values[len(Z)+len(U):])}; z=sum(d['plant_opening_cost'][k]*zs[str(k)] for k in range(K))+sum(trans[j,k]*us[f'{j}_{k}'] for j,k in U)+sum(d['setup_cost'][j][k][t]*ys[f'{j}_{k}_{t}'] for j,k,t in Y)+wcost
 if z>=best-1e-7: return
 best=z; sol={'objective_value':z,'Z':zs,'U':us,'Y':ys}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(z,sol)
def publish(values):
 signature=tuple(int(v>.5) for v in values)
 if signature in seen or time.monotonic()>=start+a.time_limit-1: return
 for var,val in zip(fixed,signature): var.LB=var.UB=val
 completion.Params.TimeLimit=max(.1,start+a.time_limit-1-time.monotonic()); completion.optimize()
 if completion.Status==GRB.OPTIMAL:
  seen.add(signature); emit(signature,sum(cv[key]*var.X for key,var in priced.items()))
def cb(model,where):
 if where==GRB.Callback.MIPSOL:
  publish(model.cbGetSolution(binary))
  if time.monotonic()>=start+a.time_limit-1: model.terminate()
m.Params.TimeLimit=max(.1,.85*(a.time_limit-(time.monotonic()-start)-1)); m.optimize(cb)
if m.SolCount:
 values=[v.X for v in binary]; publish(values)
 for var,val in zip(binary,values): var.LB=var.UB=int(val>.5); var.VType=GRB.CONTINUOUS
 m.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-1); m.optimize()
 if m.Status==GRB.OPTIMAL: emit(values,sum(cv[key]*var.X for key,var in W.items()))
completion.dispose()

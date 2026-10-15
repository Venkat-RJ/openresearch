import os
license=os.environ.get('GRB_LICENSE_FILE')
if license and not os.path.exists(license):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
import argparse,json,time
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-4);z=json.load(open(a.instance_path));n=z['num_customers'];N=list(range(1,n+1));last=n+1;Q=z['vehicle_capacity'];rho=z['inconvenience_cost_function']['rho'];d=z['distance_matrix'];records={c['id']:c for c in z['customers']};records[0]=dict(z['depot'],demand=0,service_time=0);records[last]=dict(z['depot_end'],demand=0,service_time=0);lo={i:records[i]['time_window'][0] for i in records};hi={i:records[i]['time_window'][1] for i in records};service={i:records[i]['service_time'] for i in records};dem={i:records[i]['demand'] for i in records};mid={i:(lo[i]+hi[i])/2 for i in N};arcs=[(i,j) for i in [0]+N for j in N+[last] if i!=j and (i,j)!=(0,last) and lo[i]+service[i]+d[i][j]<=hi[j]]
def params(m,budget):
 m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1;m.Params.TimeLimit=max(.1,budget)
m=gp.Model('compact_vrptw');params(m,max(.1,(end-time.monotonic())*.8));m.Params.LazyConstraints=1;x=m.addVars(arcs,vtype=gp.GRB.BINARY);t=m.addVars(N,lb=lo,ub=hi);load=m.addVars(N,lb={i:dem[i] for i in N},ub=Q);v=m.addVars(N,lb=0)
for i in N:
 m.addConstr(gp.quicksum(x[u,j] for u,j in arcs if u==i)==1);m.addConstr(gp.quicksum(x[u,j] for u,j in arcs if j==i)==1)
m.addConstr(gp.quicksum(x[i,j] for i,j in arcs if i==0)<=z['num_vehicles']);m.addConstr(gp.quicksum(x[i,j] for i,j in arcs if i==0)==gp.quicksum(x[i,j] for i,j in arcs if j==last))
for i,j in arcs:
 ti=lo[0] if i==0 else t[i];tj=hi[last] if j==last else t[j];M=max(0,hi[i]+service[i]+d[i][j]-lo[j]);m.addConstr(ti+service[i]+d[i][j]<=tj+M*(1-x[i,j]))
 if i in N and j in N:m.addConstr(load[j]>=load[i]+dem[j]-Q*(1-x[i,j]))
for i in N:
 for k in [lo[i],(3*lo[i]+hi[i])/4,mid[i],(lo[i]+3*hi[i])/4,hi[i]]:m.addConstr(v[i]>=2*rho*(k-mid[i])*t[i]+rho*(mid[i]**2-k*k))
m.setObjective(gp.quicksum(d[i][j]*x[i,j] for i,j in arcs)+gp.quicksum(v[i] for i in N));logger=SolutionLogger(a.log_path,sense='minimize');best=float('inf');best_routes=None

logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
def save(xvals,tvals):
 global best,best_routes
 succ={i:j for (i,j),val in xvals.items() if val>.5 and i!=0};routes=[]
 for (i,j),val in xvals.items():
  if i==0 and val>.5:
   route=[0,j]
   while route[-1]!=last:route.append(succ[route[-1]])
   routes.append(route)
 routing=sum(d[i][j] for route in routes for i,j in zip(route,route[1:]));inconvenience=sum(rho*(tvals[i]-mid[i])**2 for i in N);value=routing+inconvenience
 if value>=best-1e-8:return
 best=value;best_routes=routes;s={'objective_value':value,'routes':routes,'schedule':{str(i):tvals[i] for i in N},'routing_cost':routing,'inconvenience_cost':inconvenience,'runtime':time.monotonic()-start};json.dump(s,open(a.solution_path,'w'));logger.log_solution(value,s)

def cb(model,where):
 if where!=gp.GRB.Callback.MIPSOL:return
 xv=model.cbGetSolution(x);tv=model.cbGetSolution(t);vv=model.cbGetSolution(v);save(xv,tv)
 for i in N:
  k=tv[i]
  if rho*(k-mid[i])**2>vv[i]+1e-8:model.cbLazy(v[i]>=2*rho*(k-mid[i])*t[i]+rho*(mid[i]**2-k*k))
m.optimize(cb)
if m.SolCount:save({arc:x[arc].X for arc in arcs},{i:t[i].X for i in N})
if best_routes is not None and time.monotonic()<end:
 qp=gp.Model('fixed_route_schedule');params(qp,end-time.monotonic());qp.Params.Method=0;qp.Params.BarConvTol=1e-12;qp.Params.OptimalityTol=1e-9;qp.Params.FeasibilityTol=1e-9;qt=qp.addVars(N,lb=lo,ub=hi)
 for route in best_routes:
  for i,j in zip(route,route[1:]):qp.addConstr((lo[0] if i==0 else qt[i])+service[i]+d[i][j]<=(hi[last] if j==last else qt[j]))
 qp.setObjective(gp.quicksum(rho*(qt[i]-mid[i])*(qt[i]-mid[i]) for i in N));qp.optimize()
 if qp.SolCount:
  xv={arc:0 for arc in arcs}
  for route in best_routes:
   for arc in zip(route,route[1:]):xv[arc]=1
  save(xv,{i:qt[i].X for i in N})

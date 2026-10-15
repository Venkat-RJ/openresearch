import time
_wall_start=time.time()
import argparse,json,os
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic()
d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
n=d['num_customers'];J=list(range(1,n+1));end=n+1;A=[(0,j) for j in J]+[(i,j) for i in J for j in J if i!=j]+[(i,end) for i in J]
TT=d['truck_travel_times'];DD=d['drone_travel_times'];T={(i,j):TT[i][0 if j==end else j] for i,j in A};D={(i,j):DD[i][0 if j==end else j] for i,j in A};rs=[0];remaining=set(J)
while remaining:
 j=min(remaining,key=lambda j:max(TT[rs[-1]][j],DD[rs[-1]][j]));rs.append(j);remaining.remove(j)
rs.append(end);times={0:0}
for i,j in zip(rs,rs[1:]):times[j]=times[i]+max(T[i,j],D[i,j])
M=times[end]
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
xt=m.addVars(A,vtype=GRB.BINARY);xd=m.addVars(A,vtype=GRB.BINARY);yt=m.addVars(J,vtype=GRB.BINARY);yd=m.addVars(J,vtype=GRB.BINARY);yc=m.addVars(J,vtype=GRB.BINARY);t=m.addVars(range(n+2),lb=0,ub=M)
m.addConstr(t[0]==0)
for i in J:
 m.addConstr(yt[i]+yd[i]+yc[i]==1)
 for x,y in ((xt,yt),(xd,yd)):
  m.addConstr(gp.quicksum(x[ii,j] for ii,j in A if ii==i)==y[i]+yc[i]);m.addConstr(gp.quicksum(x[j,ii] for j,ii in A if ii==i)==y[i]+yc[i])
 m.addConstr(xd[0,i]+xd[i,end]<=1)
 for j in J:
  if i<j:m.addConstr(xd[i,j]+xd[j,i]<=yc[i]+yc[j])
for x,W in ((xt,T),(xd,D)):
 m.addConstr(gp.quicksum(x[0,j] for j in J)==1);m.addConstr(gp.quicksum(x[i,end] for i in J)==1)
 for i,j in A:m.addConstr(t[i]+W[i,j]<=t[j]+(M+W[i,j])*(1-x[i,j]))
 m.addConstr(gp.quicksum(W[i,j]*x[i,j] for i,j in A)<=t[end])
m.setObjective(t[end])
initial={'objective_value':M,'truck_route':rs[:-1]+[0],'drone_route':rs[:-1]+[0],'customer_types':{str(i):'combined' for i in J},'arrival_times':{str(i):times[i] for i in range(n+2)}}
if log:log.log_solution(M,initial)
for x in (xt,xd):
 for ij in A:x[ij].Start=int(ij in set(zip(rs,rs[1:])))
for i in J:yt[i].Start=0;yd[i].Start=0;yc[i].Start=1
for i in range(n+2):t[i].Start=times[i]
def pack(v):
 def route(x):
  succ={i:j for i,j in A if v(x[i,j])>.5};r=[0]
  while r[-1]!=end:r.append(succ[r[-1]])
  return r[:-1]+[0]
 return {'objective_value':v(t[end]),'truck_route':route(xt),'drone_route':route(xd),'customer_types':{str(i):'truck' if v(yt[i])>.5 else 'drone' if v(yd[i])>.5 else 'combined' for i in J},'arrival_times':{str(i):v(t[i]) for i in range(n+2)}}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible truck/drone incumbent')
s=pack(lambda z:z.X)
if log:log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

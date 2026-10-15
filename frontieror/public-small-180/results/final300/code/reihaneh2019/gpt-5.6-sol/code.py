import time
_wall_start=time.time()
import argparse,json,os,itertools
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic()
d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
S=list(range(1,d['num_delivery_sites']+1));K=[c['id'] for c in d['customers']];Q=d['vehicle_capacity'];D={c['id']:c['demand'] for c in d['customers']};C=d['routing_cost_matrix'];F=d['assignment_cost_matrix'];routes=[];cost=[]
for size in range(1,len(S)+1):
 for subset in itertools.combinations(S,size):
  rs=min(itertools.permutations(subset),key=lambda r:C[0][r[0]]+sum(C[i][j] for i,j in zip(r,r[1:]))+C[r[-1]][0])
  routes.append([0]+list(rs)+[0]);cost.append(sum(C[i][j] for i,j in zip(routes[-1],routes[-1][1:])))
R=range(len(routes));m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
z=m.addVars(R,vtype=GRB.BINARY);x=m.addVars([(r,i,k) for r in R for i in routes[r][1:-1] for k in K],vtype=GRB.BINARY)
for k in K:m.addConstr(gp.quicksum(v for (r,i,kk),v in x.items() if kk==k)==1)
for i in S:m.addConstr(gp.quicksum(z[r] for r in R if i in routes[r])<=1)
for r in R:
 m.addConstr(gp.quicksum(D[k]*v for (rr,i,k),v in x.items() if rr==r)<=Q*z[r])
m.addConstr(z.sum()<=d['num_vehicles'])
m.setObjective(gp.quicksum(cost[r]*z[r] for r in R)+gp.quicksum(F[k-1][i-1]*v for (r,i,k),v in x.items()))
def pack(v):
 selected=[r for r in R if v(z[r])>.5];ids={r:k+1 for k,r in enumerate(selected)};ass={};tours={}
 for (r,i,k),q in x.items():
  if v(q)>.5:ass[str(k)]=i;tours[str(k)]={'vehicle':ids[r],'site':i}
 rr={str(ids[r]):{'sequence':routes[r],'arcs':[list(t) for t in zip(routes[r],routes[r][1:])]} for r in selected}
 return {'objective_value':sum(cost[r] for r in selected)+sum(F[int(k)-1][i-1] for k,i in ass.items()),'routes':rr,'assignments':ass,'customer_tours':tours}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible allocation and routing')
s=pack(lambda z:z.X)
if log:log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

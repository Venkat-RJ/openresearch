import time
_wall_start=time.time()
import argparse,json,os,math,itertools
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
base=d['depot']['id'];S=[s['id'] for s in d['satellites']];C=[c['id'] for c in d['customers']];N={q['id']:q for q in [d['depot']]+d['satellites']+d['customers']};dem={c['id']:c['demand'] for c in d['customers']};group={c['id']:c['group'] for c in d['customers']};G=[g['group_id'] for g in d['groups']];K=range(d['T1']);Q=d['Q2']
def distance(i,j):return math.hypot(N[i]['x']-N[j]['x'],N[i]['y']-N[j]['y'])
patterns=[]
for z in range(1,len(S)+1):
 for r in itertools.permutations(S,z):patterns.append([base]+list(r)+[base])
R=range(len(patterns));pc=[sum(distance(i,j) for i,j in zip(r,r[1:])) for r in patterns]
A=[(i,j) for i in S+C for j in S+C if i!=j and not(i in S and j in S) and dem.get(i,0)+dem.get(j,0)<=Q]
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
p1=m.addVars(K,R,vtype=GRB.BINARY);w=m.addVars(S,K,lb=0);y=m.addVars(A,vtype=GRB.BINARY);f=m.addVars(A,lb=0);z=m.addVars(G,S,vtype=GRB.BINARY)
for g in G:m.addConstr(gp.quicksum(z[g,s] for s in S)==1)
for k in K:
 m.addConstr(gp.quicksum(p1[k,r] for r in R)<=1);m.addConstr(gp.quicksum(w[s,k] for s in S)<=d['Q1'])
 for s in S:m.addConstr(w[s,k]<=min(d['Q1'],N[s]['capacity'])*gp.quicksum(p1[k,r] for r in R if s in patterns[r]))
 if k:
  m.addConstr(gp.quicksum(p1[k-1,r] for r in R)>=gp.quicksum(p1[k,r] for r in R))
for s in S:
 m.addConstr(gp.quicksum(w[s,k] for k in K)<=N[s]['capacity'])
 m.addConstr(gp.quicksum(w[s,k] for k in K)==gp.quicksum(f[s,j] for j in C if (s,j) in f))
 m.addConstr(gp.quicksum(y[s,j] for j in C if (s,j) in y)==gp.quicksum(y[j,s] for j in C if (j,s) in y))
 m.addConstr(gp.quicksum(y[s,j] for j in C if (s,j) in y)<=N[s]['max_vehicles'])
m.addConstr(gp.quicksum(y[s,j] for s in S for j in C if (s,j) in y)<=d['T2'])
for i in C:
 m.addConstr(gp.quicksum(y[ii,j] for ii,j in A if ii==i)==1);m.addConstr(gp.quicksum(y[j,ii] for j,ii in A if ii==i)==1)
 m.addConstr(gp.quicksum(f[j,ii] for j,ii in A if ii==i)-gp.quicksum(f[ii,j] for ii,j in A if ii==i)==dem[i])
for i,j in A:
 m.addConstr(f[i,j]<=(Q-dem.get(i,0))*y[i,j])
 if j in S:m.addConstr(f[i,j]==0)
 for s in S:
  if i in C and j in C:
   if group[i]!=group[j]:m.addConstr(z[group[j],s]>=z[group[i],s]+y[i,j]-1)
  elif i==s:m.addConstr(z[group[j],s]>=y[i,j])
  elif j in S and j!=s:m.addConstr(z[group[i],s]+y[i,j]<=1)
m.setObjective(gp.quicksum(pc[r]*p1[k,r] for k in K for r in R)+gp.quicksum(N[s]['handling_cost']*w[s,k] for s in S for k in K)+gp.quicksum(distance(i,j)*y[i,j] for i,j in A))
def pack(v):
 first=[];second=[]
 for k in K:
  for r in R:
   if v(p1[k,r])>.5:first.append({'vehicle':k,'route':patterns[r],'deliveries':{str(s):max(0,v(w[s,k])) for s in S if v(w[s,k])>1e-8}})
 succ={i:j for i,j in A if i in C and v(y[i,j])>.5}
 for s in S:
  for j in C:
   if (s,j) in y and v(y[s,j])>.5:
    route=[s,j]
    while route[-1] in C:route.append(succ[route[-1]])
    second.append({'satellite':s,'route':route})
 obj=sum(sum(distance(i,j) for i,j in zip(r['route'],r['route'][1:]))+sum(N[int(s)]['handling_cost']*v for s,v in r['deliveries'].items()) for r in first)+sum(sum(distance(i,j) for i,j in zip(r['route'],r['route'][1:])) for r in second)
 return {'objective_value':obj,'first_echelon_routes':first,'second_echelon_routes':second,'group_assignments':{str(g):next(s for s in S if v(z[g,s])>.5) for g in G}}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible echelon incumbent')
s=pack(lambda q:q.X)
if log:log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

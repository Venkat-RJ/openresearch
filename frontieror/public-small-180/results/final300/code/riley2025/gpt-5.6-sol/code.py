import time
_wall_start=time.time()
import argparse,json,os,itertools
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();deadline=start+a.time_limit-.4;d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
I=range(d['num_demand_sites']);J=range(d['num_production_sites']);M=range(d['num_module_sizes']);K=range(d['units_per_module_size']);S=range(d['num_scenarios']);T=range(d['num_time_periods']);D=[sc['demands'] for sc in d['scenarios']];prob=[sc['probability'] for sc in d['scenarios']];U=[q['capacity'] for q in d['module_sizes']];C=d['production_costs'];P=d['penalty_costs'];capital={(q['module_size_id'],q['unit_id']):q['capital_cost'] for q in d['module_units']}
def model():
 m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1;return m
best=float('inf');solution=None
zero={'allocation':{},'fractions':{}}
def pack(counts,plans):
 values={};outy={};obj=sum(capital[m,k] for m in M for k in range(counts[m]))
 def put(name,ix,v):
  if abs(v)>1e-9:values[name+'['+','.join(map(str,ix))+']']=v
 for m in M:
  for k in K:
   val=int(k<counts[m]);outy['y_'+str(m)+'_'+str(k)]=val;put('y',(m,k),val)
 for s in S:
  prev={}
  for m in M:put('z',(m,s),counts[m])
  for t in T:
   plan=plans.get((s,t),zero);alloc=plan['allocation'];frac=plan['fractions']
   for i in I:
    served=0
    for j in J:
     f=frac.get((i,j),0);served+=f;put('x',(i,j,t,s),f);obj+=prob[s]*C[i][j]*D[s][i][t]*f
    unmet=max(0,1-served);put('q',(i,t,s),unmet);obj+=prob[s]*P[i][t]*D[s][i][t]*unmet
   for m in M:
    total=0
    for j in J:
     v=alloc.get((j,m),0);total+=v;put('v',(j,m,t,s),v);change=v-prev.get((j,m),0)
     if change>0:put('w_fd',(j,m,t,s),change)
     elif change<0:put('w_td',(j,m,t,s),-change)
    put('v_depot',(m,t,s),counts[m]-total)
   prev=alloc
 return {'objective_value':obj,'y':outy,'model_variables':values}
def publish(counts,plans):
 global best,solution
 sol=pack(counts,plans)
 if sol['objective_value']<best:
  best=sol['objective_value'];solution=sol
  if log:log.log_solution(best,sol)
publish([0]*len(M),{})
master=model();y=master.addVars(M,K,vtype=GRB.BINARY);counts_expr={m:gp.quicksum(y[m,k] for k in K) for m in M}
for m in M:
 for k in range(1,len(K)):master.addConstr(y[m,k]<=y[m,k-1])
allpairs=list(itertools.product(S,T));sample=sorted(allpairs,key=lambda st:sum(D[st[0]][i][st[1]] for i in I));sample=[sample[round(k*(len(sample)-1)/7)] for k in range(8)];objective=gp.quicksum(capital[m,k]*y[m,k] for m in M for k in K);candidates=[]
for s,t in sample:
 x=master.addVars(I,J,lb=0,ub=1);v=master.addVars(J,M,vtype=GRB.INTEGER,lb=0)
 for m in M:master.addConstr(gp.quicksum(v[j,m] for j in J)<=counts_expr[m])
 for i in I:master.addConstr(gp.quicksum(x[i,j] for j in J)<=1)
 for j in J:
  load=gp.quicksum(D[s][i][t]*x[i,j] for i in I);master.addConstr(load<=gp.quicksum(U[m]*v[j,m] for m in M));master.addConstr(load<=d['u_max'])
 objective+=len(T)/len(sample)*gp.quicksum(D[s][i][t]*(P[i][t]+gp.quicksum((C[i][j]-P[i][t])*x[i,j] for j in J)) for i in I)
master.setObjective(objective)
def cb(mm,where):
 if where==GRB.Callback.MIPSOL:
  counts=[sum(int(mm.cbGetSolution(y[m,k])>.5) for k in K) for m in M]
  if counts not in candidates:candidates.append(counts)
  publish(counts,{})
master.Params.TimeLimit=max(.01,min(a.time_limit*.12,deadline-time.monotonic()));master.optimize(cb)
if master.SolCount:
 counts=[sum(int(y[m,k].X>.5) for k in K) for m in M];candidates=[counts]+[c for c in reversed(candidates) if c!=counts]
for counts in candidates:
 if time.monotonic()>=deadline:break
 plans={}
 for s,t in allpairs:
  if time.monotonic()>=deadline:break
  rec=model();x=rec.addVars(I,J,lb=0,ub=1);v=rec.addVars(J,M,vtype=GRB.INTEGER,lb=0)
  for m in M:rec.addConstr(gp.quicksum(v[j,m] for j in J)<=counts[m])
  for i in I:rec.addConstr(gp.quicksum(x[i,j] for j in J)<=1)
  for j in J:
   load=gp.quicksum(D[s][i][t]*x[i,j] for i in I);rec.addConstr(load<=gp.quicksum(U[m]*v[j,m] for m in M));rec.addConstr(load<=d['u_max'])
  rec.setObjective(gp.quicksum(D[s][i][t]*(P[i][t]+gp.quicksum((C[i][j]-P[i][t])*x[i,j] for j in J)) for i in I))
  def rc(mm,where):
   if where==GRB.Callback.MIPSOL:
    plans[s,t]={'allocation':{(j,m):int(round(mm.cbGetSolution(v[j,m]))) for j in J for m in M},'fractions':{(i,j):max(0,mm.cbGetSolution(x[i,j])) for i in I for j in J}};publish(counts,plans)
  rec.Params.TimeLimit=max(.01,min(.4,deadline-time.monotonic()));rec.optimize(rc)
  if rec.SolCount:plans[s,t]={'allocation':{(j,m):int(round(v[j,m].X)) for j in J for m in M},'fractions':{(i,j):max(0,x[i,j].X) for i in I for j in J}};publish(counts,plans)
  rec.dispose()
solution['runtime']=time.monotonic()-start;json.dump(solution,open(a.solution_path,'w'))

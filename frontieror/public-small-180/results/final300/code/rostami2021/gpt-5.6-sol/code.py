import time
_wall_start=time.time()
import argparse,json,os,itertools,math
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();deadline=start+a.time_limit-.3;d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
n=d['n'];N=range(n);S=d['scenarios'];dist=d['distances'];P=d['cost_parameters'];variant=d['problem_info']['variant'];fixed=d['fixed_costs'];cap=d.get('hub_capacities');out=[[sum(row) for row in s['demands']] for s in S];incoming=[[sum(s['demands'][j][i] for j in N) for i in N] for s in S]
linear=[[[dist[i][k]*(P['chi']*out[ss][i]+P['delta']*incoming[ss][i]) for k in N] for i in N] for ss in range(len(S))]
best=float('inf');solution=None

def publish(H,alloc):
 global best,solution
 val=0 if variant=='SApHMP' else sum(fixed[k] for k in H)
 for ss,s in enumerate(S):
  q=alloc[ss];val+=s['probability']*(sum(linear[ss][i][q[i]] for i in N)+sum(P['alpha']*s['demands'][i][j]*dist[q[i]][q[j]] for i in N for j in N))
 if val>=best:return
 best=val;solution={'objective_value':val,'hubs':list(H),'allocations':{str(ss):{str(i):q[i] for i in N} for ss,q in enumerate(alloc)},'z_values':[int(i in H) for i in N],'x_values':{str(ss):{str(i):{str(k):int(q[i]==k) for k in N if k!=i} for i in N} for ss,q in enumerate(alloc)}}
 if log:log.log_solution(val,solution)

if variant!='SApHMP' or d['p_hubs']==1:
 for k in N:
  if variant=='CSAHLP' and any(sum(out[ss])>cap[k]+1e-7 for ss in range(len(S))):continue
  publish((k,),[[k]*n for s in S])
candidates=[]
for size in range(1,n+1):
 if variant=='SApHMP' and size!=d['p_hubs']:continue
 for H in itertools.combinations(N,size):
  if variant=='CSAHLP' and any(sum(out[ss])>sum(cap[k] for k in H)+1e-7 for ss in range(len(S))):continue
  lb=(0 if variant=='SApHMP' else sum(fixed[k] for k in H))+sum(s['probability']*sum(min(linear[ss][i][k] for k in H) for i in N) for ss,s in enumerate(S));candidates.append((lb,H))
def feasible_allocations(H):
 allocations=[]
 for ss in range(len(S)):
  alloc=[i if i in H else min(H,key=lambda k:linear[ss][i][k]) for i in N]
  if variant=='CSAHLP':
   remaining={k:cap[k]-out[ss][k] for k in H};spokes=sorted((i for i in N if i not in H),key=lambda i:out[ss][i],reverse=True)
   def assign(pos):
    if time.monotonic()>=deadline:return False
    if pos==len(spokes):return True
    i=spokes[pos]
    for k in sorted(H,key=lambda k:linear[ss][i][k]):
     if out[ss][i]>remaining[k]+1e-7:continue
     remaining[k]-=out[ss][i];alloc[i]=k
     if assign(pos+1):return True
     remaining[k]+=out[ss][i]
    return False
   if min(remaining.values())<-1e-7 or not assign(0):return None
  allocations.append(alloc)
 return allocations
for lb,H in sorted(candidates):
 if time.monotonic()>=deadline or lb>=best-1e-8:break
 allocations=feasible_allocations(H)
 if allocations is None:continue
 publish(H,allocations)
 for ss,s in enumerate(S):
  remaining=deadline-time.monotonic()
  if remaining<=.03:break
  m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1;m.Params.NonConvex=2;m.Params.TimeLimit=max(.02,min(3,remaining/(len(S)-ss)))
  spokes=[i for i in N if i not in H];x=m.addVars(spokes,H,vtype=GRB.BINARY)
  for i in spokes:m.addConstr(gp.quicksum(x[i,k] for k in H)==1)
  def X(i,k):return int(i==k) if i in H else x[i,k]
  if variant=='CSAHLP':
   for k in H:m.addConstr(gp.quicksum(out[ss][i]*X(i,k) for i in N)<=cap[k])
  m.setObjective(gp.quicksum(linear[ss][i][k]*x[i,k] for i in spokes for k in H)+gp.quicksum(P['alpha']*s['demands'][i][j]*dist[k][l]*X(i,k)*X(j,l) for i in N for j in N for k in H for l in H if k!=l))
  for i in spokes:
   nearest=min(H,key=lambda k:linear[ss][i][k])
   for k in H:x[i,k].Start=int(k==nearest)
  def cb(model,where):
   if where==GRB.Callback.MIPSOL:
    allocations[ss]=[i if i in H else max(H,key=lambda k:model.cbGetSolution(x[i,k])) for i in N];publish(H,allocations)
  m.optimize(cb)
  if not m.SolCount:m.dispose();break
  allocations[ss]=[i if i in H else max(H,key=lambda k:x[i,k].X) for i in N];publish(H,allocations);m.dispose()
if solution is None:raise RuntimeError('No feasible stochastic hub allocation')
solution['runtime']=time.monotonic()-start;json.dump(solution,open(a.solution_path,'w'))

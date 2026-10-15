import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random
from ortools.sat.python import cp_model
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));customers=d['customers'];ids=[z['id'] for z in customers];nom={z['id']:z['nominal_demand'] for z in customers};dev={z['id']:z['demand_deviation'] for z in customers};gamma=d['uncertainty_set']['Gamma'];cap=d['vehicle_capacity'];K=d['num_vehicles'];D=d['distance_matrix'];depot=d['depot']['id'];deadline=time.monotonic()+max(.1,float(a.time_limit)-2);n=len(ids)
 if n!=d['num_customers'] or set(ids)!=set(range(1,d['num_customers']+1)) or len(D)!=d['num_customers']+1:raise RuntimeError('Input-contract conflict: declared customer count/IDs/matrix disagree')
 def demand(route):return sum(nom[v] for v in route)+sum(sorted((dev[v] for v in route),reverse=True)[:gamma])
 def cost(route):r=[depot]+route+[depot];return sum(D[u][v] for u,v in zip(r,r[1:]))
 best=None;value=float('inf')
 if n<=15:
  dp={};pred={};patterns=[]
  for mask in range(1,1<<n):
   route=[ids[j] for j in range(n) if mask>>j&1]
   if demand(route)>cap:continue
   for j in range(n):
    if not(mask>>j&1):continue
    prev=mask^(1<<j)
    if not prev:dp[mask,j]=D[depot][ids[j]];pred[mask,j]=-1
    else:
     v,k=min((dp[prev,k]+D[ids[k]][ids[j]],k) for k in range(n) if prev>>k&1);dp[mask,j]=v;pred[mask,j]=k
   v,last=min((dp[mask,j]+D[ids[j]][depot],j) for j in range(n) if mask>>j&1);seq=[];q=mask;k=last
   while k>=0:seq.append(ids[k]);old=k;k=pred[q,k];q^=1<<old
   patterns.append((mask,list(reversed(seq)),v))
  model=cp_model.CpModel();bs=[model.NewBoolVar('r'+str(i)) for i in range(len(patterns))]
  for j in range(n):model.Add(sum(bs[k] for k,(mask,_,_) in enumerate(patterns) if mask>>j&1)==1)
  model.Add(sum(bs)==K);model.Minimize(sum(v*bs[k] for k,(_,_,v) in enumerate(patterns)));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,deadline-time.monotonic());status=s.Solve(model)
  if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):best=[route for k,(_,route,_) in enumerate(patterns) if s.Value(bs[k])];value=sum(map(cost,best))
 else:
  rng=random.Random(42);attempt=0
  while time.monotonic()<deadline:
   routes=[[v] for v in ids];valid=True
   savings=[]
   for i,u in enumerate(ids):
    for v in ids[i+1:]:
     saving=D[u][depot]+D[depot][v]-D[u][v];savings.append((saving*(1 if not attempt else rng.uniform(.8,1.2)),u,v))
   savings.sort(reverse=True);owner={v:i for i,v in enumerate(ids)};active=n
   for _,u,v in savings:
    if active==K:break
    i,j=owner[u],owner[v]
    if i==j or u not in (routes[i][0],routes[i][-1]) or v not in (routes[j][0],routes[j][-1]):continue
    left=routes[i] if routes[i][-1]==u else routes[i][::-1];right=routes[j] if routes[j][0]==v else routes[j][::-1];merged=left+right
    if demand(merged)>cap:continue
    routes[i]=merged;routes[j]=[];active-=1
    for w in merged:owner[w]=i
   if active!=K:attempt+=1;continue
   routes=[r for r in routes if r];improved=True
   while improved and time.monotonic()<deadline:
    improved=False
    for i,r in enumerate(routes):
     old=cost(r)
     for x in range(len(r)):
      for y in range(x+2,len(r)+1):
       trial=r[:x]+r[x:y][::-1]+r[y:];new=cost(trial)
       if new<old:r[:]=trial;old=new;improved=True
    move=None;gain=0
    for i,r in enumerate(routes):
     if len(r)<2:continue
     for x,v in enumerate(r):
      u=depot if x==0 else r[x-1];w=depot if x==len(r)-1 else r[x+1];saving=D[u][v]+D[v][w]-D[u][w]
      for j,s in enumerate(routes):
       if i==j or demand(s+[v])>cap:continue
       for y in range(len(s)+1):
        u2=depot if y==0 else s[y-1];w2=depot if y==len(s) else s[y];delta=saving-(D[u2][v]+D[v][w2]-D[u2][w2])
        if delta>gain:gain=delta;move=(i,x,j,y)
    if move:
     i,x,j,y=move;routes[j].insert(y,routes[i].pop(x));improved=True
   obj=sum(map(cost,routes))
   if obj<value:best=[r[:] for r in routes];value=obj
   attempt+=1
 if best is None:raise RuntimeError('Unable to find robust capacity-feasible fleet')
 out={'objective_value':value,'routes':[{'customers':r,'cost':cost(r),'worst_case_demand':demand(r)} for r in best]};json.dump(out,open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(value)
if __name__=='__main__':main()

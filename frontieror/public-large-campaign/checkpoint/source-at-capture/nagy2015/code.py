import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));n=d['num_customers'];C=d['vehicle_capacity'];D=d['distance_matrix'];jobs={j['id']:j for j in d['customers']};deadline=time.monotonic()+max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;best=float('inf');rng=random.Random(0)
def physical(i):return i if i<=n else i-n
def cost(r):return sum(D[physical(r[i])][physical(r[i+1])] for i in range(len(r)-1))
def feasible(r):
 load=sum(jobs[i]['delivery_demand'] for i in r if 1<=i<=n)
 if load>C:return False
 for i in r[1:-1]:
  load+=(-jobs[i]['delivery_demand'] if i<=n else jobs[i-n]['pickup_demand'])
  if load>C:return False
 return True
def save(routes):
 global best
 obj=sum(cost(r) for r in routes)
 if obj<best:
  best=obj;detail=[]
  for r in routes:
   row=[]
   for i in r:
    if not i:row.append({'node':0,'type':'depot','original_customer':0,'quantity':0})
    elif i<=n:row.append({'node':i,'type':'linehaul','original_customer':i,'quantity':jobs[i]['delivery_demand']})
    else:row.append({'node':i,'type':'backhaul','original_customer':i-n,'quantity':jobs[i-n]['pickup_demand']})
   detail.append(row)
  s={'objective_value':obj,'routes':routes,'routes_detailed':detail};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)
base=[[0,i,n+i,0] for i in range(1,n+1)];save(base)
for it in range(100000):
 if time.monotonic()>deadline:break
 routes=[r.copy() for r in base]
 while len(routes)>1 and time.monotonic()<deadline:
  options=[]
  for i in range(len(routes)):
   for j in range(i+1,len(routes)):
    for u,v in [(routes[i],routes[j]),(routes[j],routes[i])]:
     r=u[:-1]+v[1:]
     if feasible(r):options.append(((cost(u)+cost(v)-cost(r))*(1 if not it else rng.uniform(.7,1.3)),i,j,r))
  if not options:break
  saving,i,j,r=max(options)
  if saving<0:break
  routes[i]=r;routes.pop(j)
 save(routes)

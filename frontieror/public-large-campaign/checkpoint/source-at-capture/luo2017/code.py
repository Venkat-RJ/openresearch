import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random,math
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));n=d['num_customers'];C=d['vehicle_capacity'];D=d['distance_matrix'];jobs={j['id']:j for j in d['customers']};depot=d['depot'];aa=d['cost_function']['a'];bb=d['cost_function']['b'];deadline=time.monotonic()+max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;best=float('inf');rng=random.Random(0)
def cost(r,q):
 load=sum(q.values());obj=0
 for i,j in zip(r[:-1],r[1:]):obj+=D[i][j]*(aa*load+bb);load-=q.get(j,0)
 return obj
def feasible(r):
 t=depot['time_window'][0]+depot['service_time']
 for i,j in zip(r[:-1],r[1:]):
  t+=D[i][j]
  if j:
   t=max(t,jobs[j]['time_window'][0])
   if t>jobs[j]['time_window'][1]+1e-7:return False
   t+=jobs[j]['service_time']
 return t<=min(d['scheduling_horizon'],depot['time_window'][1])+1e-7
def save(routes):
 global best
 obj=sum(cost(r,q) for r,q in routes)
 if obj<best:
  best=obj;result=[]
  for k,(r,q) in enumerate(routes):
   load=sum(q.values());edges=[]
   for i,j in zip(r[:-1],r[1:]):edges.append({'from':i,'to':j if j else n+1,'load_weight':load});load-=q.get(j,0)
   result.append({'vehicle':k,'route':r[:-1]+[n+1],'deliveries':{str(i):v for i,v in q.items()},'edges':edges})
  s={'objective_value':obj,'routes':result};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)
base=[]
for j in jobs.values():
 rem=j['demand']
 while rem>0:q=min(C,rem);base.append(([depot['id'],j['id'],depot['id']],{j['id']:q}));rem-=q
save(base)
while time.monotonic()<deadline:
 routes=[(r.copy(),q.copy()) for r,q in base]
 while len(routes)>1 and time.monotonic()<deadline:
  choices=[]
  for i in range(len(routes)):
   for j in range(i+1,len(routes)):
    for u,v in [(i,j),(j,i)]:
     r1,q1=routes[u];r2,q2=routes[v]
     if sum(q1.values())+sum(q2.values())>C or set(q1)&set(q2):continue
     r=r1[:-1]+r2[1:];q={**q1,**q2}
     if feasible(r):choices.append(((cost(r1,q1)+cost(r2,q2)-cost(r,q))*rng.uniform(.8,1.2),i,j,r,q))
  if not choices:break
  saving,i,j,r,q=max(choices)
  if saving<=0:break
  routes[i]=(r,q);routes.pop(j)
 save(routes)

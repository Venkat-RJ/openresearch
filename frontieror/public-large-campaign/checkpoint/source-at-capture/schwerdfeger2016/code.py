import os
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math,random
from ortools.sat.python import cp_model

def solve(d,limit):
 p=d['processing_times']; m=d['num_machines']; n=len(p); mu=sum(p)/m
 groups=[[] for _ in range(m)]; loads=[0]*m
 for j in sorted(range(n),key=lambda j:-p[j]):
  k=min(range(m),key=lambda k:loads[k]); groups[k].append(j); loads[k]+=p[j]
 deadline=time.monotonic()+max(0,limit-1)
 for it in range(300):
  if time.monotonic()>deadline:break
  improved=False
  for a in sorted(range(m),key=lambda k:-loads[k]):
   for b in sorted(range(m),key=lambda k:loads[k]):
    diff=loads[a]-loads[b]
    if diff<=1:continue
    best=(0,None,None)
    for j in groups[a]:
     gain=2*p[j]*(diff-p[j])
     if gain>best[0]:best=(gain,j,None)
     for k in groups[b]:
      q=p[j]-p[k]; gain=2*q*(diff-q)
      if gain>best[0]:best=(gain,j,k)
    if best[0]>0:
     _,j,k=best; groups[a].remove(j);groups[b].append(j);loads[a]-=p[j];loads[b]+=p[j]
     if k is not None:groups[b].remove(k);groups[a].append(k);loads[b]-=p[k];loads[a]+=p[k]
     improved=True
  if not improved:break
 if n*m<15000 and sum(p)<10000000 and time.monotonic()<deadline:
  model=cp_model.CpModel(); x=[[model.new_bool_var(f'x{i}_{j}') for j in range(n)] for i in range(m)]
  for j in range(n):model.add_exactly_one(x[i][j] for i in range(m))
  C=[];sq=[];total=sum(p)
  for i in range(m):
   c=model.new_int_var(0,total,f'c{i}');model.add(c==sum(p[j]*x[i][j] for j in range(n)));C.append(c)
   s=model.new_int_var(0,total*total,f's{i}');model.add_multiplication_equality(s,[c,c]);sq.append(s)
   chosen=set(groups[i])
   for j in range(n):model.add_hint(x[i][j],int(j in chosen))
  model.minimize(sum(sq)); model.add(sum(sq)<=sum(c*c for c in loads))
  s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic())
  status=s.solve(model)
  if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):groups=[[j for j in range(n) if s.value(x[i][j])] for i in range(m)];loads=[sum(p[j] for j in g) for g in groups]
 return {'objective_value':math.sqrt(sum((c-mu)**2 for c in loads))/mu,'assignment':{str(i):g for i,g in enumerate(groups)},'machine_completion_times':{str(i):c for i,c in enumerate(loads)}}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--instance_path',required=True);a.add_argument('--solution_path',required=True);a.add_argument('--time_limit',type=float,default=60);a.add_argument('--log_path');args=a.parse_args()
 out=solve(json.load(open(args.instance_path)),args.time_limit)
 with open(args.solution_path,'w') as f:json.dump(out,f)

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random
import numpy as np
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));D=np.array(d['distance_matrix']);scores=d['scores'];n=d['num_nodes'];dep=d['depot_vertex'];limit=d['distance_limitation_d0'];deadline=time.monotonic()+max(.1,a.time_limit-2);rng=random.Random(0);logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None;best=-1

def length(r):return sum(D[r[i],r[i+1]] for i in range(len(r)-1))
def save(r):
 global best
 if len(r)<4:return
 obj=sum(scores[i] for i in r[:-1])
 if obj>best:
  best=obj;s={'objective_value':obj,'visited_vertices':r[:-1],'tour_edges':[[r[i],r[i+1]] for i in range(len(r)-1)]};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)
for it in range(100000):
 if time.monotonic()>deadline:break
 r=[dep,dep];remaining=set(range(n))-{dep};total=0
 while remaining:
  opts=[]
  for j in remaining:
   pos=min(range(1,len(r)),key=lambda p:D[r[p-1],j]+D[j,r[p]]-D[r[p-1],r[p]])
   extra=D[r[pos-1],j]+D[j,r[pos]]-D[r[pos-1],r[pos]]
   if total+extra<=limit:opts.append((scores[j]/max(.1,extra)*(1 if not it else rng.uniform(.6,1.4)),j,pos,extra))
  if not opts:break
  _,j,pos,extra=max(opts);r.insert(pos,j);total+=extra;remaining.remove(j)
  if time.monotonic()>deadline:break
 for _ in range(3):
  for i in range(1,len(r)-2):
   for j in range(i+1,len(r)-1):
    if D[r[i-1],r[j]]+D[r[i],r[j+1]]<D[r[i-1],r[i]]+D[r[j],r[j+1]]:r[i:j+1]=reversed(r[i:j+1])
 save(r)

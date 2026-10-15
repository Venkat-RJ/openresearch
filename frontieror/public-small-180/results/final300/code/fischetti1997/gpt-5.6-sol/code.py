import argparse,json,time,random,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser(); p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int);p.add_argument('--log_path');a=p.parse_args()
 start=time.monotonic(); deadline=start+max(0.1,a.time_limit-4)
 data=json.load(open(a.instance_path));n=data['problem_size']['n']; groups=[c['node_ids'] for c in data['clusters']];m=len(groups)
 d=np.zeros((n,n),dtype=np.int64); iu=np.triu_indices(n,1);d[iu]=data['distance_matrix_upper_triangular']['values'];d+=d.T.copy()
 membership={v:k for k,g in enumerate(groups) for v in g}; rng=random.Random(0); logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 best=None;bestval=float('inf')
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 def cost(t): return int(sum(d[t[k-1],t[k]] for k in range(m)))
 def save(t):
  nonlocal best,bestval
  value=cost(t)
  if value<bestval:
   best=t[:];bestval=value;sol={'objective_value':value,'tour':best,'visited_nodes':best,'runtime':time.monotonic()-start};json.dump(sol,open(a.solution_path,'w'))
   if logger:logger.log_solution(value,sol)
 def improve(t):
  changed=True
  while changed and time.monotonic()<deadline:
   changed=False
   for k,v in enumerate(t):
    g=groups[membership[v]];u=min(g,key=lambda x:d[t[k-1],x]+d[x,t[(k+1)%m]])
    if u!=v:t[k]=u;changed=True;save(t)
   for i in range(m-2):
    if time.monotonic()>=deadline:break
    for j in range(i+2,m):
     if i==0 and j==m-1:continue
     aa,bb,cc,ee=t[i],t[i+1],t[j],t[(j+1)%m]
     if d[aa,cc]+d[bb,ee]<d[aa,bb]+d[cc,ee]:t[i+1:j+1]=t[i+1:j+1][::-1];changed=True;save(t)
   save(t)
  return t
 rep=[g[0] for g in groups]
 for iteration in range(100000):
  if time.monotonic()>=deadline:break
  if iteration==0:
   t=[rep[0]];unused=set(range(1,m))
   while unused:
    k,v=min(((k,v) for k in unused for v in groups[k]),key=lambda kv:d[t[-1],kv[1]]);t.append(v);unused.remove(k)
  elif iteration%3==0:
   order=list(range(m));rng.shuffle(order);t=[rng.choice(groups[k]) for k in order]
  else:
   t=best[:];i,j=sorted(rng.sample(range(m),2));t[i:j+1]=t[i:j+1][::-1]
   for _ in range(3):k=rng.randrange(m);t[k]=rng.choice(groups[membership[t[k]]])
  save(t);improve(t)
if __name__=='__main__':main()

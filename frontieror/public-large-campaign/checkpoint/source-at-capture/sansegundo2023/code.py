import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse,json,time,sys
sys.setrecursionlimit(10000)
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args(); deadline=time.monotonic()+max(.1,float(a.time_limit)-2)
 d=json.load(open(a.instance_path));g=d.get('graph',d); raw=g['adjacency_list']; labels=sorted(map(int,raw)); idx={v:i for i,v in enumerate(labels)}; n=len(labels)
 adj=[sum(1<<idx[u] for u in raw[str(v)] if u in idx) for v in labels]; best=[]
 def record(c):
  nonlocal best
  if len(c)>len(best):best=c[:]
 def greedy(P):
  c=[]
  while P:
   vs=[];q=P
   while q:
    b=q&-q;q-=b;v=b.bit_length()-1;vs.append(v)
   v=max(vs,key=lambda v:(adj[v]&P).bit_count());c.append(v);P &=adj[v]
  record(c)
 greedy((1<<n)-1)
 def color(P):
  order=[];bounds=[];col=0
  while P:
   col+=1;Q=P
   while Q:
    b=Q&-Q;v=b.bit_length()-1;P ^=b;Q ^=b;Q &=~adj[v];order.append(v);bounds.append(col)
  return order,bounds
 ticks=0
 def expand(C,P):
  nonlocal ticks
  ticks+=1
  if ticks%128==0 and time.monotonic()>deadline:raise TimeoutError
  order,bounds=color(P)
  for j in range(len(order)-1,-1,-1):
   if len(C)+bounds[j]<=len(best):return
   v=order[j];Q=P&adj[v]
   if Q:expand(C+[v],Q)
   else:record(C+[v])
   P &=~(1<<v)
 try:expand([], (1<<n)-1)
 except TimeoutError:pass
 out={'objective_value':len(best),'clique_vertices':[labels[v] for v in best]};json.dump(out,open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  l=SolutionLogger(a.log_path,sense='maximize');l.log(len(best))
if __name__=='__main__':main()

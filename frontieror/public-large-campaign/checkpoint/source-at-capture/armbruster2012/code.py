import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[key]='1'
import time
START=time.monotonic()
import argparse,json,random,heapq
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();end=START+max(.5,a.time_limit-4);d=json.load(open(a.instance_path));n=d['num_nodes'];weights=d['node_weights'];edges=d['edges'];costs=d['edge_weights'];total=sum(weights);F=d['bisection_capacity_F'];low=total-F;target=total//2;log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;best=float('inf');bestpart=None;rng=random.Random(271828)
 def balanced(order):
  part=[0]*n;s=0;position=0
  for position,u in enumerate(order):
   if s+weights[u]>target:break
   part[u]=1;s+=weights[u]
  if low<=s<=F:return part
  for u in order[position:]:
   if not part[u] and weights[u]<=target-s:part[u]=1;s+=weights[u]
   if s>=low:return part
  # Exact subset-sum construction is independent of graph connectivity.
  cap=F;mask=(1<<(cap+1))-1;reachable=1;previous=[-1]*(cap+1);owner=[-1]*(cap+1)
  for u in order:
   w=weights[u]
   if w==0:continue
   fresh=((reachable<<w)&mask)&~reachable
   while fresh:
    bit=fresh&-fresh;t=bit.bit_length()-1;previous[t]=t-w;owner[t]=u;fresh^=bit
   reachable|=(reachable<<w)&mask
   candidates=(reachable>>low)&((1<<(F-low+1))-1)
   if candidates:
    value=low+(candidates&-candidates).bit_length()-1;part=[0]*n
    while value:part[owner[value]]=1;value=previous[value]
    return part
  raise RuntimeError('No capacity-feasible weighted partition exists')
 def emit(part):
  nonlocal best,bestpart
  load=sum(w for w,x in zip(weights,part) if x)
  if not low<=load<=F:return
  obj=sum(w for (u,v),w in zip(edges,costs) if part[u]!=part[v])
  if obj>=best:return
  best=obj;bestpart=part[:]
  if part[0]:normalized=[1-x for x in part]
  else:normalized=part
  sol={'objective_value':obj,'partition_S':[u for u in range(n) if not normalized[u]],'partition_complement':[u for u in range(n) if normalized[u]]};json.dump(sol,open(a.solution_path,'w'))
  if log:
   if hasattr(log,'log_solution'):log.log_solution(obj,sol)
   else:log.log(obj)
 emit(balanced(list(range(n))))
 adj=[{} for _ in range(n)]
 for (u,v),w in zip(edges,costs):
  if u==v:continue
  adj[u][v]=adj[u].get(v,0)+w;adj[v][u]=adj[v].get(u,0)+w
 def refine(initial,deadline,maxpasses=6):
  part=initial[:];emit(part)
  for _ in range(maxpasses):
   if time.monotonic()>deadline:break
   gains=[sum(w if part[u]!=part[v] else -w for v,w in adj[u].items()) for u in range(n)];heaps={};locked=[False]*n
   for u in range(n):heapq.heappush(heaps.setdefault((weights[u],part[u]),[]),(-gains[u],u))
   classes=sorted({w for w in weights if (w,0) in heaps and (w,1) in heaps});sequence=[];cumulative=0;peak=0;peaklen=0
   def candidates(w,side):
    h=heaps.get((w,side),[]);out=[]
    while h and len(out)<4:
     ng,u=heapq.heappop(h)
     if locked[u] or part[u]!=side or -ng!=gains[u]:continue
     out.append((ng,u))
    for e in out:heapq.heappush(h,e)
    return [u for _,u in out]
   def flip(u):
    old=part[u];part[u]=1-old;gains[u]=-gains[u];locked[u]=True
    for v,w in adj[u].items():
     gains[v]+=2*w if part[v]==old else -2*w
     if not locked[v]:heapq.heappush(heaps[(weights[v],part[v])],(-gains[v],v))
   for step in range(n//2):
    if step%64==0 and time.monotonic()>deadline:break
    chosen=None;gain=-float('inf')
    for w in classes:
     left=candidates(w,0);right=candidates(w,1)
     for u in left:
      for v in right:
       g=gains[u]+gains[v]-2*adj[u].get(v,0)
       if g>gain:gain=g;chosen=(u,v)
    if chosen is None:break
    u,v=chosen;flip(u);flip(v);sequence.append(chosen);cumulative+=gain
    if cumulative>peak:peak=cumulative;peaklen=len(sequence)
    if step>128 and cumulative<peak-max(100,abs(best)*.03):break
   for u,v in reversed(sequence[peaklen:]):part[u]=1-part[u];part[v]=1-part[v]
   if peaklen:emit(part)
   else:break
  return part
 if time.monotonic()<end-1:
  # Heavy-edge contraction preserves the full weighted graph objective.
  current=adj;aggregate=weights[:];membership=list(range(n));coarse_deadline=min(end-2,START+a.time_limit*.32)
  while len(current)>160 and time.monotonic()<coarse_deadline:
   count=len(current);order=list(range(count));rng.shuffle(order);mapping=[-1]*count;newweights=[]
   for u in order:
    if mapping[u]>=0:continue
    partners=[v for v in current[u] if mapping[v]<0 and v!=u]
    v=max(partners,key=lambda v:current[u][v]/max(1,(aggregate[u]+aggregate[v])**.25)) if partners else None
    g=len(newweights);mapping[u]=g;weight=aggregate[u]
    if v is not None:mapping[v]=g;weight+=aggregate[v]
    newweights.append(weight)
   if len(newweights)>=count*.95:break
   nxt=[{} for _ in newweights]
   for u,row in enumerate(current):
    gu=mapping[u]
    for v,w in row.items():
     if v<=u:continue
     gv=mapping[v]
     if gu!=gv:nxt[gu][gv]=nxt[gu].get(gv,0)+w;nxt[gv][gu]=nxt[gv].get(gu,0)+w
   membership=[mapping[u] for u in membership];current=nxt;aggregate=newweights
  if len(current)<=600 and time.monotonic()<end-1:
   import numpy as np
   size=len(current);matrix=np.zeros((size,size))
   for u,row in enumerate(current):
    for v,w in row.items():matrix[u,v]=w
   deg=matrix.sum(axis=1);inverse=1/np.sqrt(np.maximum(deg,1));lap=np.eye(size)-inverse[:,None]*matrix*inverse[None,:];values,vectors=np.linalg.eigh(lap)
   for k in range(1,min(4,size)):
    if time.monotonic()>end-2:break
    scores=vectors[:,k]*inverse;order=sorted(range(n),key=lambda u:float(scores[membership[u]]));part=balanced(order);refine(part,min(end,START+a.time_limit*(.60 if k==1 else .82)),4)
 if n<=300 and time.monotonic()<end-1:
  from ortools.sat.python import cp_model
  m=cp_model.CpModel();x=[m.new_bool_var('x%d'%u) for u in range(n)];m.add(x[0]==0);m.add(sum(weights[u]*x[u] for u in range(n))>=low);m.add(sum(weights[u]*x[u] for u in range(n))<=F);cuts=[]
  for k,(u,v) in enumerate(edges):z=m.new_bool_var('e%d'%k);m.add_abs_equality(z,x[u]-x[v]);cuts.append(z)
  m.minimize(sum(w*z for w,z in zip(costs,cuts)));hint=bestpart if not bestpart[0] else [1-z for z in bestpart]
  for u in range(n):m.add_hint(x[u],hint[u])
  for (u,v),z in zip(edges,cuts):m.add_hint(z,int(hint[u]!=hint[v]))
  class Callback(cp_model.CpSolverSolutionCallback):
   def on_solution_callback(self):emit([self.value(z) for z in x])
  solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.01,end-time.monotonic());solver.solve(m,Callback())
 else:
  while time.monotonic()<end-1:
   part=bestpart[:];byweight={}
   for u,w in enumerate(weights):byweight.setdefault((w,part[u]),[]).append(u)
   common=[w for w in set(weights) if byweight.get((w,0)) and byweight.get((w,1))]
   for _ in range(max(4,n//400)):
    available=[w for w in common if byweight[(w,0)] and byweight[(w,1)]]
    if not available:break
    w=rng.choice(available);left=byweight[(w,0)];right=byweight[(w,1)];u=left.pop(rng.randrange(len(left)));v=right.pop(rng.randrange(len(right)));part[u]=1;part[v]=0
   refine(part,end,3)
if __name__=='__main__':main()

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[key]='1'
import time
START=time.monotonic()
import argparse,json,random,math
from solution_logger import SolutionLogger

def main():
 parser=argparse.ArgumentParser()
 for key in ('instance_path','solution_path','log_path'):parser.add_argument('--'+key,required=True)
 parser.add_argument('--time_limit',type=int,default=60);a=parser.parse_args();end=START+max(.5,a.time_limit-3);data=json.load(open(a.instance_path));n=data['num_items'];capacity=data['capacity'];original_p=data['items']['profits'];original_w=data['items']['weights'];edges=data['conflict_graph']['edges'];logger=SolutionLogger(a.log_path,sense='maximize');rng=random.Random(137);best=-1;bestids=[];bestmask=0
 def emit_original(ids):
  nonlocal best,bestids,bestmask
  value=sum(original_p[u] for u in ids)
  if value<=best:return
  best=value;bestids=ids[:];sol={'objective_value':value,'selected_items':sorted(ids)};json.dump(sol,open(a.solution_path,'w'))
  if hasattr(logger,'log_solution'):logger.log_solution(value,sol)
  else:logger.log(value)
 degree=[0]*n
 for u,v in edges:degree[u]+=1;degree[v]+=1
 ordering=sorted(range(n),key=lambda u:(-original_p[u],degree[u],original_w[u]));inverse=[0]*n
 for u,v in enumerate(ordering):inverse[v]=u
 profits=[original_p[u] for u in ordering];weights=[original_w[u] for u in ordering];full=(1<<n)-1;conflicts=[0]*n
 for u,v in edges:
  u=inverse[u];v=inverse[v];conflicts[u]|=1<<v;conflicts[v]|=1<<u
 compatible=[full&~(conflicts[u]|(1<<u)) for u in range(n)]
 def vertices(mask):
  while mask:
   bit=mask&-mask;yield bit.bit_length()-1;mask^=bit
 def emit(ids):
  nonlocal bestmask
  value=sum(profits[u] for u in ids)
  if value>best:bestmask=sum(1<<u for u in ids);emit_original([ordering[u] for u in ids])
 def allowed_after(ids):
  mask=full
  for u in ids:mask&=compatible[u]
  return mask
 def fill(keep,alpha,beta,randomized=False):
  ids=keep[:];remaining=capacity-sum(weights[u] for u in ids);mask=allowed_after(ids)
  while mask and time.monotonic()<end:
   fitting=[u for u in vertices(mask) if weights[u]<=remaining]
   if not fitting:break
   if alpha:
    scores=[(profits[u]/max(1,weights[u])**beta/(1+(conflicts[u]&mask).bit_count())**alpha,u) for u in fitting]
   else:scores=[(profits[u]/max(1,weights[u])**beta,u) for u in fitting]
   if randomized:
    scores.sort(reverse=True);u=scores[rng.randrange(min(4,len(scores)))][1]
   else:u=max(scores)[1]
   ids.append(u);remaining-=weights[u];mask&=compatible[u]
  emit(ids);return ids
 fill([],0,1)
 if time.monotonic()>=end:return
 for alpha,beta in [(0,0),(.5,0),(1,0),(.5,.3),(1,.5),(0,.5)]:
  if time.monotonic()>=end:break
  fill([],alpha,beta)
 def coloring(mask):
  order=[];bounds=[];done=0;uncolored=mask
  while uncolored:
   available=uncolored;group=[]
   while available:
    bit=available&-available;u=bit.bit_length()-1;group.append(u);uncolored^=bit;available^=bit;available&=conflicts[u]
   group.sort(key=lambda u:profits[u]);maximum=0
   for u in group:maximum=max(maximum,profits[u]);order.append(u);bounds.append(done+maximum)
   done+=maximum
  return order,bounds
 def search(mask,keep,remaining,deadline,node_limit=30000):
  nodes=0;initial=sum(profits[u] for u in keep)
  def visit(candidates,ids,profit,room):
   nonlocal nodes
   nodes+=1
   if nodes>node_limit or (nodes&127)==0 and time.monotonic()>deadline:return
   if profit>best:emit(ids)
   if not candidates:return
   candidates=sum(1<<u for u in vertices(candidates) if weights[u]<=room)
   order,bounds=coloring(candidates)
   for k in range(len(order)-1,-1,-1):
    if profit+bounds[k]<=best:return
    u=order[k];bit=1<<u
    visit(candidates&compatible[u],ids+[u],profit+profits[u],room-weights[u]);candidates^=bit
    if nodes>node_limit or time.monotonic()>deadline:return
  visit(mask,keep,initial,remaining)
 if n<=150 and time.monotonic()<end-1:
  from ortools.sat.python import cp_model
  model=cp_model.CpModel();x=[model.new_bool_var(str(u)) for u in range(n)];model.add(sum(weights[u]*x[u] for u in range(n))<=capacity)
  for ou,ov in edges:model.add(x[inverse[ou]]+x[inverse[ov]]<=1)
  model.maximize(sum(profits[u]*x[u] for u in range(n)))
  for u in range(n):model.add_hint(x[u],int(bool(bestmask>>u&1)))
  class Callback(cp_model.CpSolverSolutionCallback):
   def on_solution_callback(self):emit([u for u in range(n) if self.value(x[u])])
  solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.01,end-time.monotonic());solver.solve(model,Callback());return
 pool=[];iteration=0;heuristic_end=min(end-1,START+a.time_limit*.48)
 while time.monotonic()<heuristic_end:
  base=list(vertices(bestmask)) if iteration%7 else []
  if base:
   remove=min(len(base),rng.randrange(1,min(7,len(base))+1));removed=set(rng.sample(base,remove));base=[u for u in base if u not in removed]
  alpha=rng.choice([0,.3,.6,1]);beta=rng.choice([0,.2,.5,.8]);candidate=fill(base,alpha,beta,iteration%3!=0)
  if iteration%12==0 and candidate:
   keep=candidate[:];remove=min(len(keep),4);removed=set(rng.sample(keep,remove));keep=[u for u in keep if u not in removed];mask=allowed_after(keep)
   if mask.bit_count()<=100:search(mask,keep,capacity-sum(weights[u] for u in keep),min(heuristic_end,time.monotonic()+.15),8000)
  iteration+=1
 if time.monotonic()<end-1:search(full,[],capacity,min(end,START+a.time_limit*.78),1000000)
 while time.monotonic()<end-1:
  base=list(vertices(bestmask));remove=min(len(base),rng.randrange(2,min(7,len(base))+1)) if len(base)>=2 else len(base);removed=set(rng.sample(base,remove));keep=[u for u in base if u not in removed];mask=allowed_after(keep)
  if mask.bit_count()<=140:search(mask,keep,capacity-sum(weights[u] for u in keep),min(end,time.monotonic()+.3),30000)
  else:fill(keep,rng.choice([0,.5,1]),rng.choice([0,.2,.5]),True)
if __name__=='__main__':main()

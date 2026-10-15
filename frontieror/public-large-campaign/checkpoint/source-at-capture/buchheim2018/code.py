import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[key]='1'
import time
START=time.monotonic()
import argparse,json,heapq,random
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser()
 for key in ('instance_path','solution_path','log_path'):p.add_argument('--'+key,required=True)
 p.add_argument('--time_limit',type=int,required=True);a=p.parse_args();end=START+max(.5,a.time_limit-4);d=json.load(open(a.instance_path));arcs=d['arcs'];N=d['num_nodes'];A=len(arcs);Q=d['quadratic_costs'];L=d['linear_costs'];s=d['source_node'];t=d['target_node'];logger=SolutionLogger(a.log_path);rng=random.Random(218);best=float('inf');best_path=None;elite=[]
 outgoing=[[] for _ in range(N)];incoming=[[] for _ in range(N)]
 for i,e in enumerate(arcs):outgoing[e['from_node']].append(i);incoming[e['to_node']].append(i)
 C=[[Q[i][j]+Q[j][i] for j in range(A)] for i in range(A)];linear=[L[i]+Q[i][i] for i in range(A)]
 def objective(path):return sum(linear[i] for i in path)+sum(C[i][j] for pos,i in enumerate(path) for j in path[:pos])
 def accept(path):
  nonlocal best,best_path
  u=s;seen={s}
  for i in path:
   e=arcs[i]
   if e['from_node']!=u or e['to_node'] in seen:return
   u=e['to_node'];seen.add(u)
  if u!=t:return
  value=objective(path)
  if value<best:
   best=value;best_path=path[:];solution={'objective_value':value,'solution_arcs':[arcs[i] for i in path]};json.dump(solution,open(a.solution_path,'w'))
   if hasattr(logger,'log_solution'):logger.log_solution(value,solution)
   else:logger.log(value)
  if len(elite)<12 or value<elite[-1][0]:
   key=tuple(path)
   if all(tuple(z)!=key for _,z in elite):elite.append((value,path[:]));elite.sort(key=lambda z:z[0]);del elite[12:]
 def shortest(weights):
  distances=[float('inf')]*N;distances[s]=0;previous=[None]*N;heap=[(0,s)]
  while heap:
   value,u=heapq.heappop(heap)
   if value!=distances[u]:continue
   if u==t:
    path=[]
    while u!=s:
     i=previous[u];path.append(i);u=arcs[i]['from_node']
    return path[::-1]
   for i in outgoing[u]:
    v=arcs[i]['to_node'];new=value+max(0,weights[i])
    if new<distances[v]:distances[v]=new;previous[v]=i;heapq.heappush(heap,(new,v))
  return None
 initial=shortest(linear)
 if initial is None:raise RuntimeError('No source-target path')
 accept(initial)
 degree=[len(incoming[u]) for u in range(N)];queue=[u for u in range(N) if not degree[u]];order=[]
 for u in queue:
  order.append(u)
  for i in outgoing[u]:
   v=arcs[i]['to_node'];degree[v]-=1
   if not degree[v]:queue.append(v)
 dag=len(order)==N;position={u:i for i,u in enumerate(order)};nonnegative=all(c>=0 for row in C for c in row) and all(c>=0 for c in linear)
 def neighborhood(path,left,right,deadline):
  outside=path[:left]+path[right:];start=s if not left else arcs[path[left-1]]['to_node'];target=arcs[path[right-1]]['to_node'];forbidden={s}
  u=s
  for index,i in enumerate(path):
   u=arcs[i]['to_node']
   if index<left-1 or index>=right:forbidden.add(u)
  forbidden.discard(start);forbidden.discard(target)
  weights=[linear[i]+sum(C[i][j] for j in outside) for i in range(A)];lower=[float('inf')]*N;lower[target]=0
  if dag:
   for u in reversed(order):
    if u==target or u in forbidden:continue
    for i in outgoing[u]:
     v=arcs[i]['to_node']
     if v not in forbidden:lower[u]=min(lower[u],weights[i]+lower[v])
  limit=objective(path)-objective(outside);winner=path[left:right];visits=0
  def dfs(u,sequence,cost,visited):
   nonlocal limit,winner,visits
   visits+=1
   if visits%128==0 and time.monotonic()>deadline:return
   if u==target:
    if cost<limit:limit=cost;winner=sequence[:]
    return
   choices=[]
   for i in outgoing[u]:
    v=arcs[i]['to_node']
    if v in visited or v in forbidden:continue
    if dag and position[v]>position[target]:continue
    new=cost+weights[i]+sum(C[i][j] for j in sequence)
    bound=new+(lower[v] if dag else 0)
    if nonnegative and bound>=limit:continue
    choices.append((bound,i,new,v))
   choices.sort()
   for _,i,new,v in choices:
    if time.monotonic()>deadline:return
    dfs(v,sequence+[i],new,visited|{v})
  dfs(start,[],0,{start})
  return path[:left]+winner+path[right:]
 def polish(path,deadline,passes=3):
  for repeat in range(passes):
   improved=False
   for width in (4,7,10,13):
    intervals=list(range(max(0,len(path)-width+1)));rng.shuffle(intervals)
    for left in intervals:
     if time.monotonic()>deadline:return path
     candidate=neighborhood(path,left,left+width,min(deadline,time.monotonic()+.07));old=objective(path)
     if objective(candidate)<old:path=candidate;accept(path);improved=True
   if not improved:break
  return path
 primal_end=min(end-2,START+a.time_limit*(.22 if A<=200 else .72))
 polish(best_path[:],min(primal_end,time.monotonic()+2))
 if dag:
  for width in (32,128,384):
   if time.monotonic()>primal_end:break
   labels=[[] for _ in range(N)];labels[s]=[(0,())]
   for u in order:
    if time.monotonic()>primal_end:break
    current=heapq.nsmallest(width,labels[u]);labels[u]=[]
    if u==t:
     for _,path in current[:8]:accept(list(path))
     continue
    for value,path in current:
     for i in outgoing[u]:
      new=value+linear[i]+sum(C[i][j] for j in path);v=arcs[i]['to_node']
      if nonnegative and new>=best:continue
      labels[v].append((new,path+(i,)))
   polish(best_path[:],min(primal_end,time.monotonic()+2))
 iteration=0
 while time.monotonic()<primal_end:
  iteration+=1
  if iteration%3==0 and elite:
   path=rng.choice(elite)[1][:];left=rng.randrange(len(path));right=min(len(path),left+rng.randint(8,18));candidate=neighborhood(path,left,right,min(primal_end,time.monotonic()+.35))
  else:
   reference=rng.choice(elite)[1] if elite else best_path;alpha=rng.choice((0,.15,.35,.6,1));noise=rng.uniform(0,15);weights=[linear[i]+alpha*sum(C[i][j] for j in reference)+rng.random()*noise for i in range(A)];candidate=shortest(weights)
  if candidate is not None:accept(candidate);polish(candidate,min(primal_end,time.monotonic()+.6),2)
 if time.monotonic()>end-1:return
 from ortools.sat.python import cp_model
 def refine(pool,deadline):
  if time.monotonic()>deadline-1:return
  pool=sorted(pool);model=cp_model.CpModel();x={i:model.new_bool_var('x%d'%i) for i in pool};ranks=None if dag else [model.new_int_var(0,N-1,'r%d'%u) for u in range(N)];chosen=set(best_path)
  if ranks:model.add(ranks[s]==0)
  for u in range(N):
   out=[x[i] for i in outgoing[u] if i in x];inc=[x[i] for i in incoming[u] if i in x];model.add(sum(out)-sum(inc)==(1 if u==s else -1 if u==t else 0));model.add(sum(out)<=1);model.add(sum(inc)<=1)
  terms=[linear[i]*x[i] for i in pool]
  for index,i in enumerate(pool):
   if time.monotonic()>deadline-1:return
   model.add_hint(x[i],int(i in chosen))
   if ranks:model.add(ranks[arcs[i]['to_node']]>ranks[arcs[i]['from_node']]).only_enforce_if(x[i])
   for j in pool[:index]:
    c=C[i][j]
    if c:
     z=model.new_bool_var('z%d_%d'%(i,j));model.add(z<=x[i]);model.add(z<=x[j]);model.add(z>=x[i]+x[j]-1);terms.append(c*z);model.add_hint(z,int(i in chosen and j in chosen))
  model.minimize(sum(terms));model.add(sum(terms)<=int(best))
  class Callback(cp_model.CpSolverSolutionCallback):
   def on_solution_callback(self):
    selected={arcs[i]['from_node']:i for i in pool if self.value(x[i])};path=[];u=s
    while u!=t:
     if u not in selected:return
     i=selected[u];path.append(i);u=arcs[i]['to_node']
    accept(path)
  solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.05,deadline-time.monotonic()-.5);solver.solve(model,Callback())
 if A<=200:refine(range(A),end)
 else:
  pool=set(i for _,path in elite[:6] for i in path)
  nodes={s,t}|{arcs[i]['to_node'] for i in best_path};pool.update(i for i,e in enumerate(arcs) if e['from_node'] in nodes or e['to_node'] in nodes)
  refine(pool,min(end,time.monotonic()+max(2,(end-time.monotonic())*.65)))
  if time.monotonic()<end-2:refine(range(A),end)
if __name__=='__main__':main()

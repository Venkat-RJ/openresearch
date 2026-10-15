import argparse,json,os,time,itertools,random
wall_started=time.time()
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True)
a=p.parse_args(); end=time.monotonic()+a.time_limit-1; d=json.load(open(a.instance_path)); tsp=d['problem_type']=='TSP'; logger=configured_logger(a.log_path,sense='minimize' if tsp else 'maximize'); best=float('inf') if tsp else -float('inf')
if tsp:
 n=d['parameters']['n_cities']; D=d['data']['distance_matrix']
 def emit(tour):
  global best
  z=sum(D[tour[i]][tour[(i+1)%n]] for i in range(n))
  if z>=best-1e-12: return
  best=z; sol={'objective_value':z,'tour':list(tour),'solution':{f'x_{i}_{t}':int(tour[t]==i) for i in range(n) for t in range(n)}}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(z,sol)
 tour=[0]
 while len(tour)<n: tour.append(min(set(range(n))-set(tour),key=lambda j:D[tour[-1]][j]))
 emit(tour)
 if n<=11:
  for q,tail in enumerate(itertools.permutations(range(1,n))):
   if q%128==0 and time.monotonic()>=end: break
   emit((0,)+tail)
 else:
  rng=random.Random(0)
  while time.monotonic()<end:
   improved=True
   while improved and time.monotonic()<end:
    improved=False
    for i in range(n-1):
     for j in range(i+2,n):
      if D[tour[i]][tour[j]]+D[tour[i+1]][tour[(j+1)%n]]<D[tour[i]][tour[i+1]]+D[tour[j]][tour[(j+1)%n]]-1e-12: tour[i+1:j+1]=reversed(tour[i+1:j+1]); improved=True
   emit(tour); tour=rng.sample(tour,n)
else:
 n=d['parameters']['n_nodes']; edges=d['data']['edges']; weights=d['data'].get('edge_weights',[1]*len(edges)); adj=[[] for _ in range(n)]; rng=random.Random(0)
 for (i,j),v in zip(edges,weights): adj[i].append((j,v)); adj[j].append((i,v))
 def emit(x):
  global best
  z=sum(v for (i,j),v in zip(edges,weights) if x[i]!=x[j])
  if z<=best: return
  best=z; sol={'objective_value':z,'solution':{f'x_{i}':x[i] for i in range(n)}}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(z,sol)
 emit([0]*n)
 while time.monotonic()<end:
  x=[rng.randrange(2) for _ in range(n)]
  while time.monotonic()<end:
   gain=[sum(v if x[i]==x[j] else -v for j,v in adj[i]) for i in range(n)]; i=max(range(n),key=lambda i:gain[i])
   if gain[i]<=0: break
   x[i]=1-x[i]
  emit(x)

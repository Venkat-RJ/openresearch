import argparse,json,os,time,heapq,math,random
wall_started=time.time()
started=time.monotonic()
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']): del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True)
a=p.parse_args(); end=started+a.time_limit-3; d=json.load(open(a.instance_path)); n=d['num_nodes']; arcs=[tuple(v) for v in d['arcs']]; A=len(arcs); comm=d['commodities']; K=len(comm); keys=[f'{i}_{j}' for i,j in arcs]; fixed=[d['fixed_costs'][s] for s in keys]; cost=[[d['variable_costs'][s][k]*comm[k]['demand'] for s in keys] for k in range(K)]; adj=[[] for _ in range(n)]; logger=configured_logger(a.log_path,sense='minimize'); best=math.inf
for e,(i,j) in enumerate(arcs): adj[i].append((j,e))
def shortest(k,opened,extra=False):
 origin=comm[k]['origin']; dest=comm[k]['destination']; dist=[math.inf]*n; parent=[None]*n; dist[origin]=0; queue=[(0,origin)]
 while queue:
  z,i=heapq.heappop(queue)
  if z!=dist[i]: continue
  for j,e in adj[i]:
   if not extra and e not in opened: continue
   zz=z+cost[k][e]+(fixed[e] if extra and e not in opened else 0)
   if zz<dist[j]: dist[j]=zz; parent[j]=(i,e); heapq.heappush(queue,(zz,j))
 if not math.isfinite(dist[dest]): return dist,None
 path=[]; i=dest
 while i!=origin: i,e=parent[i]; path.append(e)
 return dist,path
def emit(paths):
 global best,bestpaths
 opened=set(e for path in paths for e in path); z=sum(fixed[e] for e in opened)+sum(cost[k][e] for k,path in enumerate(paths) for e in path)
 if z>=best-1e-7: return
 best=z; bestpaths=[tuple(path) if path and arcs[path[0]][0]==comm[k]['origin'] else tuple(reversed(path)) for k,path in enumerate(paths)]; sol={'objective_value':z,'open_arcs':{keys[e]:1 for e in opened},'routings':{str(k):{keys[e]:1. for e in path} for k,path in enumerate(paths)}}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(z,sol)
paths=[shortest(k,set(range(A)))[1] for k in range(K)]; emit(paths); rng=random.Random(0)
for sweep in range(5):
 for k in rng.sample(range(K),K):
  if time.monotonic()>=end: break
  opened=set(e for l,path in enumerate(paths) if l!=k for e in path); path=shortest(k,opened,True)[1]; paths[k]=path; emit(paths)
 if time.monotonic()>=end: break
def spath(source,dest,weights,ban_nodes,ban_edges):
 dist=[math.inf]*n; parents=[None]*n; dist[source]=0.; queue=[(0.,source)]
 while queue:
  z,i=heapq.heappop(queue)
  if z!=dist[i]: continue
  if i==dest: break
  for j,e in adj[i]:
   if j in ban_nodes or e in ban_edges: continue
   zz=z+weights[e]
   if zz<dist[j]: dist[j]=zz; parents[j]=(i,e); heapq.heappush(queue,(zz,j))
 if not math.isfinite(dist[dest]): return None
 path=[]; i=dest
 while i!=source: i,e=parents[i]; path.append(e)
 return tuple(reversed(path))
def yen(k,weights,count):
 origin=comm[k]['origin']; dest=comm[k]['destination']; first=spath(origin,dest,weights,set(),set()); accepted=[first]; heap=[]; queued={first}
 for q in range(count-1):
  if time.monotonic()>=poolend: break
  prev=accepted[-1]; nodes=[origin]+[arcs[e][1] for e in prev]
  for pos in range(len(prev)):
   prefix=prev[:pos]; bans={path[pos] for path in accepted if len(path)>pos and path[:pos]==prefix}; tail=spath(nodes[pos],dest,weights,set(nodes[:pos]),bans)
   if tail is not None:
    path=prefix+tail
    if path not in queued: queued.add(path); heapq.heappush(heap,(sum(weights[e] for e in path),path))
  if not heap: break
  accepted.append(heapq.heappop(heap)[1])
 return accepted
poolend=min(end,started+.2*a.time_limit); limit=max(1,min(40,(1800-A-K)//K)); pool=[]
for k in range(K):
 ps={bestpaths[k]}
 for factor in [0.,1/K,.15,1.]:
  weights=[cost[k][e]+factor*fixed[e] for e in range(A)]
  ps.update(yen(k,weights,max(2,limit//4)))
 ps=sorted(ps,key=lambda path:sum(cost[k][e]+fixed[e]/K for e in path)); ps=[bestpaths[k]]+[p for p in ps if p!=bestpaths[k]]; pool.append(ps[:limit])
if time.monotonic()<end:
 model=gp.Model(); model.Params.Threads=1; model.Params.Seed=0; model.Params.MIPGap=1e-4; model.Params.NumericFocus=0; y=model.addVars(A,vtype=GRB.BINARY,obj={e:fixed[e] for e in range(A)}); x={(k,q):model.addVar(vtype=GRB.BINARY,obj=sum(cost[k][e] for e in path)) for k,ps in enumerate(pool) for q,path in enumerate(ps)}
 for k,ps in enumerate(pool):
  model.addConstr(gp.quicksum(x[k,q] for q in range(len(ps)))==1)
  for q,path in enumerate(ps): model.addConstr(len(path)*x[k,q]<=gp.quicksum(y[e] for e in path)); x[k,q].Start=int(q==0)
 for e in range(A): model.addConstr(gp.quicksum(x[k,q] for k,ps in enumerate(pool) for q,path in enumerate(ps) if e in path)<=K*y[e]); y[e].Start=int(any(e in path for path in bestpaths))
 model.update(); assert model.NumVars<=2000 and model.NumConstrs<=2000
 def extract(values):
  vals=dict(zip(x,values)); emit([pool[k][max(range(len(pool[k])),key=lambda q:vals[k,q])] for k in range(K)])
 def cb(m,where):
  if where==GRB.Callback.MIPSOL: extract(m.cbGetSolution(list(x.values())))
 model.Params.TimeLimit=max(.1,min(.3*a.time_limit,end-time.monotonic())); model.optimize(cb)
 if model.SolCount: extract([v.X for v in x.values()])
 model.dispose()
if time.monotonic()<end:
 model=gp.Model(); model.Params.Threads=1; model.Params.Seed=0; model.Params.MIPGap=1e-4; model.Params.NumericFocus=0; model.Params.OutputFlag=0; y=model.addVars(A,ub=1.,obj={e:fixed[e] for e in range(A)})
 cover={k:model.addConstr(gp.LinExpr()==1.) for k in range(K)}; use={e:model.addConstr(-K*y[e]<=0) for e in range(A)}; important=sorted(range(A),key=lambda e:fixed[e],reverse=True)[:max(0,min(A,(1900-K-A)//K))]; link={(k,e):model.addConstr(-y[e]<=0) for k in range(K) for e in important}; x={}; added=[set() for _ in range(K)]
 def column(k,path):
  path=tuple(path)
  if path in added[k]: return False
  col=gp.Column([1.],[cover[k]])
  for e in path:
   col.addTerms(1.,use[e])
   if (k,e) in link: col.addTerms(1.,link[k,e])
  x[k,path]=model.addVar(obj=sum(cost[k][e] for e in path),column=col); added[k].add(path); return True
 model.update()
 for k,ps in enumerate(pool):
  for path in ps: column(k,path)
  column(k,bestpaths[k])
 priceend=min(end,started+.45*a.time_limit)
 while time.monotonic()<priceend:
  model.update(); assert model.NumVars<=2000 and model.NumConstrs<=2000; model.Params.TimeLimit=max(.05,priceend-time.monotonic()); model.optimize()
  if model.Status!=GRB.OPTIMAL: break
  pending=[]
  for k in range(K):
   weights=[max(0.,cost[k][e]-use[e].Pi-(link[k,e].Pi if (k,e) in link else 0.)) for e in range(A)]; path=spath(comm[k]['origin'],comm[k]['destination'],weights,set(),set())
   if path not in added[k] and sum(weights[e] for e in path)-cover[k].Pi < -1e-6: pending.append((k,path))
  if not pending: break
  room=2000-model.NumVars
  if room<=0: break
  for k,path in pending[:room]: column(k,path)
 for e in range(A): y[e].VType=GRB.BINARY; y[e].Start=int(any(e in path for path in bestpaths))
 def recover(values):
  opened={e for e,v in enumerate(values) if v>.5}; paths=[shortest(k,opened)[1] for k in range(K)]
  if all(path is not None for path in paths): emit(paths)
 def cbprice(m,where):
  if where==GRB.Callback.MIPSOL: recover(m.cbGetSolution(list(y.values())))
 model.update(); assert model.NumVars<=2000 and model.NumConstrs<=2000; model.Params.TimeLimit=max(.1,end-time.monotonic()); model.optimize(cbprice)
 if model.SolCount: recover([y[e].X for e in range(A)])

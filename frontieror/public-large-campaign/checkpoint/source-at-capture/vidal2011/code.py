import argparse,json,os,time,math,itertools,random
from functools import lru_cache
os.environ['OMP_NUM_THREADS']='1'
from ortools.sat.python import cp_model
from ortools.constraint_solver import pywrapcp,routing_enums_pb2

def smallsolve(d,limit):
 n=d['n'];deps=d['depots'];cust=d['customers'];T=d['t'];fleet=d['m'];cap=d['vehicle_capacity'];dur=d['route_duration_limit'];deadline=time.monotonic()+max(.1,limit-1);m=cp_model.CpModel();y={};cost=[]
 def dist(a,b):return math.hypot(a['x']-b['x'],a['y']-b['y'])
 D=[[dist(a,b) for b in cust] for a in cust];DD=[[dist(a,b) for b in cust] for a in deps]
 for i,c in enumerate(cust):
  for b in range(len(deps)):
   for p,pat in enumerate(c['pattern_list']):y[i,b,p]=m.new_bool_var(f'y{i}_{b}_{p}')
  m.add_exactly_one(y[i,b,p] for b in range(len(deps)) for p in range(len(c['pattern_list'])))
 if n<=10:
  patterns={};columns={}
  for b in range(len(deps)):
   best={}
   def dfs(path,mask,load,length,service):
    if time.monotonic()>deadline:return
    if path:
     end=length+DD[b][path[-1]]
     if end+service<=dur+1e-9 and (mask not in best or end<best[mask][0]):best[mask]=(end,path.copy())
    for j in range(n):
     if mask>>j&1 or load+cust[j]['demand']>cap:continue
     ll=length+(DD[b][j] if not path else D[path[-1]][j]);ss=service+cust[j]['service_duration']
     if ll+DD[b][j]+ss<=dur+1e-9:dfs(path+[j],mask|1<<j,load+cust[j]['demand'],ll,ss)
   dfs([],0,0,0.,0.)
   patterns[b]=best
   for t in range(1,T+1):
    for mask,(length,path) in best.items():
     x=m.new_bool_var(f'r{b}_{t}_{mask}');columns[b,t,mask]=x;cost.append(round(length*1000000)*x)
    m.add(sum(x for (bb,tt,mask),x in columns.items() if bb==b and tt==t)<=fleet)
    for i,c in enumerate(cust):m.add(sum(x for (bb,tt,mask),x in columns.items() if bb==b and tt==t and mask>>i&1)==sum(y[i,b,p] for p,pat in enumerate(c['pattern_list']) if t in pat))
  m.minimize(sum(cost));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());st=s.solve(m)
  if st not in (cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible periodic route set')
  routes=[];counts={};obj=0
  for (b,t,mask),x in columns.items():
   if s.value(x):counts[b,t]=counts.get((b,t),0)+1;length,path=patterns[b][mask];routes.append({'depot':deps[b]['id'],'period':t,'vehicle':counts[b,t],'customers':[cust[i]['id'] for i in path]});obj+=length
 else:
  for b in range(len(deps)):
   for t in range(1,T+1):m.add(sum(c['demand']*y[i,b,p] for i,c in enumerate(cust) for p,pat in enumerate(c['pattern_list']) if t in pat)<=fleet*cap)
  m.minimize(sum(round(DD[b][i]*1000)*y[i,b,p] for i,b,p in y));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,(deadline-time.monotonic())*.2);st=s.solve(m)
  if st not in (cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible depot-pattern assignment')
  routes=[];obj=0
  for b in range(len(deps)):
   for t in range(1,T+1):
    ids=[i for i,c in enumerate(cust) if any(s.value(y[i,b,p]) and t in pat for p,pat in enumerate(c['pattern_list']))]
    if not ids:continue
    locations=[deps[b]]+[cust[i] for i in ids];mgr=pywrapcp.RoutingIndexManager(len(locations),fleet,0);r=pywrapcp.RoutingModel(mgr);cb=r.RegisterTransitCallback(lambda a,z:round(dist(locations[mgr.IndexToNode(a)],locations[mgr.IndexToNode(z)])*1000000));r.SetArcCostEvaluatorOfAllVehicles(cb);dem=r.RegisterUnaryTransitCallback(lambda a:0 if mgr.IndexToNode(a)==0 else locations[mgr.IndexToNode(a)]['demand']);r.AddDimensionWithVehicleCapacity(dem,0,[cap]*fleet,True,'cap');tc=r.RegisterTransitCallback(lambda a,z:round((dist(locations[mgr.IndexToNode(a)],locations[mgr.IndexToNode(z)])+locations[mgr.IndexToNode(a)].get('service_duration',0))*1000000));r.AddDimension(tc,0,round(dur*1000000),True,'time');par=pywrapcp.DefaultRoutingSearchParameters();par.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;par.time_limit.FromMilliseconds(max(10,int((deadline-time.monotonic())/max(1,len(deps)*T)*1000)));sol=r.SolveWithParameters(par)
    if not sol:raise RuntimeError('CVRP routing subproblem infeasible')
    for v in range(fleet):
     cur=r.Start(v);path=[];length=0
     while not r.IsEnd(cur):nxt=sol.Value(r.NextVar(cur));a=mgr.IndexToNode(cur);z=mgr.IndexToNode(nxt);length+=dist(locations[a],locations[z]);cur=nxt
     cur=r.Start(v)
     while not r.IsEnd(cur):a=mgr.IndexToNode(cur);cur=sol.Value(r.NextVar(cur));path+=([locations[a]['id']] if a else [])
     if path:routes.append({'depot':deps[b]['id'],'period':t,'vehicle':v+1,'customers':path});obj+=length
 assignments={}
 for (i,b,p),x in y.items():
  if s.value(x):assignments[str(cust[i]['id'])]={'depot':deps[b]['id'],'pattern':cust[i]['pattern_list'][p]}
 return {'objective_value':obj,'routes':routes,'assignments':assignments}
def solve(d,limit,solution_path=None):
 if d['n']<=10:return smallsolve(d,limit)
 deadline=time.monotonic()+max(.1,limit-1);cust=d['customers'];deps=d['depots'];n=len(cust);K=len(deps);T=d['t'];cap=d['vehicle_capacity'];fleet=d['m'];dur=d['route_duration_limit'];rng=random.Random(0)
 def distance(a,b):return math.hypot(a['x']-b['x'],a['y']-b['y'])
 D=[[distance(a,b) for b in cust] for a in cust];DD=[[distance(a,b) for b in cust] for a in deps];options=[[(b,p) for b in range(K) for p in range(len(c['pattern_list']))] for c in cust];model=cp_model.CpModel();y={}
 for i,c in enumerate(cust):
  for b,p in options[i]:y[i,b,p]=model.new_bool_var(f'y{i}_{b}_{p}')
  model.add_exactly_one(y[i,b,p] for b,p in options[i])
 for b in range(K):
  for t in range(1,T+1):model.add(sum(c['demand']*y[i,b,p] for i,c in enumerate(cust) for p,pat in enumerate(c['pattern_list']) if t in pat)<=fleet*cap)
 model.minimize(sum(round(DD[b][i]*1000)*y[i,b,p] for i,b,p in y));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.01,(deadline-time.monotonic())*.1);status=solver.solve(model)
 if status not in (cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible initial joint assignment')
 assignment=[next((b,p) for b,p in options[i] if solver.value(y[i,b,p])) for i in range(n)];cache={}
 def routeset(b,globalmask):
  if not globalmask:return 0.,[]
  key=(b,globalmask)
  if key in cache:return cache[key]
  ids=[i for i in range(n) if globalmask>>i&1];k=len(ids)
  if k>13 or time.monotonic()>deadline:return float('inf'),[]
  N=1<<k;weight=[0]*N;service=[0]*N;dp={};tours={}
  for mask in range(1,N):
   low=mask&-mask;idx=low.bit_length()-1;prev=mask-low;weight[mask]=weight[prev]+cust[ids[idx]]['demand'];service[mask]=service[prev]+cust[ids[idx]]['service_duration']
   if weight[mask]>cap:continue
   r=mask;besttour=None
   while r:
    bit=r&-r;r-=bit;j=bit.bit_length()-1;previous=mask-bit
    if not previous:value=DD[b][ids[j]];path=(ids[j],)
    else:
     candidates=[(dp[previous,h][0]+D[ids[h]][ids[j]],dp[previous,h][1]+(ids[j],)) for h in range(k) if previous>>h&1 and (previous,h) in dp]
     if not candidates:continue
     value,path=min(candidates,key=lambda z:z[0])
    dp[mask,j]=(value,path);total=value+DD[b][ids[j]]
    if total+service[mask]<=dur+1e-8 and (besttour is None or total<besttour[0]):besttour=(total,path)
   if besttour:tours[mask]=besttour
   if mask%512==0 and time.monotonic()>deadline:return float('inf'),[]
  @lru_cache(None)
  def partition(mask,vehicles):
   if not mask:return 0.,()
   if not vehicles:return float('inf'),()
   if mask in tours:return tours[mask][0],(mask,)
   anchor=mask&-mask;best=(float('inf'),());sub=mask
   while sub:
    if sub&anchor and sub in tours:
     rest,parts=partition(mask^sub,vehicles-1);value=tours[sub][0]+rest
     if value<best[0]:best=(value,(sub,)+parts)
    sub=(sub-1)&mask
   return best
  value,parts=partition(N-1,fleet);result=(value,[tours[sub][1] for sub in parts]);cache[key]=result;return result
 def groups(a):
  masks={(b,t):0 for b in range(K) for t in range(1,T+1)}
  for i,(b,p) in enumerate(a):
   for t in cust[i]['pattern_list'][p]:masks[b,t]|=1<<i
  return masks
 def emit(a,masks,value):
  routes=[]
  for (b,t),mask in masks.items():
   _,paths=routeset(b,mask)
   for v,path in enumerate(paths):routes.append({'depot':deps[b]['id'],'period':t,'vehicle':v+1,'customers':[cust[i]['id'] for i in path]})
  return {'objective_value':value,'routes':routes,'assignments':{str(cust[i]['id']):{'depot':deps[b]['id'],'pattern':cust[i]['pattern_list'][p]} for i,(b,p) in enumerate(a)}}
 masks=groups(assignment);value=sum(routeset(b,mask)[0] for (b,t),mask in masks.items())
 if not math.isfinite(value):return smallsolve(d,max(.1,deadline-time.monotonic()))
 bestvalue=value;bestassignment=assignment.copy();best=emit(assignment,masks,value)
 def persist(out):
  if solution_path:
   with open(solution_path,'w') as f:json.dump(out,f)
 persist(best);stalled=0
 while time.monotonic()<deadline:
  order=list(range(n));rng.shuffle(order);improved=False
  for i in order:
   if time.monotonic()>deadline:break
   oldb,oldp=assignment[i];oldperiods=cust[i]['pattern_list'][oldp];moves=options[i].copy();rng.shuffle(moves);bestmove=None;delta_best=-1e-7
   for b,p in moves:
    if (b,p)==(oldb,oldp):continue
    changes={}
    for t in oldperiods:changes[oldb,t]=masks[oldb,t]&~(1<<i)
    for t in cust[i]['pattern_list'][p]:changes[b,t]=changes.get((b,t),masks[b,t])|1<<i
    delta=sum(routeset(bb,mask)[0]-routeset(bb,masks[bb,t])[0] for (bb,t),mask in changes.items())
    if delta<delta_best:delta_best=delta;bestmove=(b,p,changes)
    if time.monotonic()>deadline:break
   if bestmove:
    b,p,changes=bestmove;assignment[i]=(b,p);masks.update(changes);value+=delta_best;improved=True
    if value<bestvalue-1e-7:bestvalue=value;bestassignment=assignment.copy();best=emit(assignment,masks,value);persist(best)
  if improved:stalled=0;continue
  stalled+=1
  if stalled>30:break
  assignment=bestassignment.copy()
  for i in rng.sample(range(n),min(n,1+stalled%3)):assignment[i]=rng.choice(options[i])
  masks=groups(assignment);value=sum(routeset(b,mask)[0] for (b,t),mask in masks.items())
  if not math.isfinite(value):assignment=bestassignment.copy();masks=groups(assignment);value=bestvalue
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 out=solve(json.load(open(a.instance_path)),a.time_limit,a.solution_path)
 with open(a.solution_path,'w') as f:json.dump(out,f)

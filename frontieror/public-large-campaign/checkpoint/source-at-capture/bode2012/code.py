import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
 os.environ[key]='1'
import time
START=time.monotonic()
import argparse,json,heapq,random,itertools
from collections import defaultdict,Counter
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser()
 for key in ('instance_path','solution_path','log_path'):p.add_argument('--'+key,required=True)
 p.add_argument('--time_limit',type=int,default=60);a=p.parse_args();end=START+max(.5,a.time_limit-4);d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path);edges={e['edge_id']:e for e in d['edges']};required=[e for e in d['edges'] if e['is_required']];n=len(required);K=d['fleet']['num_vehicles'];Q=d['fleet']['vehicle_capacity'];depot=d['depot'];adj=defaultdict(list);dist={};paths={};rng=random.Random(71);best=float('inf');best_sequences=None
 for e in edges.values():
  u,v=e['endpoints'];adj[u].append((v,e['cost'],e['edge_id']));adj[v].append((u,e['cost'],e['edge_id']))
 for root in adj:
  ds={root:0};previous={};heap=[(0,root)]
  while heap:
   value,u=heapq.heappop(heap)
   if value!=ds[u]:continue
   for v,c,e in adj[u]:
    new=value+c
    if new<ds.get(v,float('inf')):ds[v]=new;previous[v]=(u,e);heapq.heappush(heap,(new,v))
  dist[root]=ds
  for v in ds:
   route=[];u=v
   while u!=root:u,e=previous[u];route.append(e)
   paths[root,v]=route[::-1]
 starts=[];ends=[]
 for e in required:
  u,v=e['endpoints'];starts.extend([u,v]);ends.extend([v,u])
 def nearest(group):
  left=group[:];position=depot;sequence=[]
  while left:
   _,state=min((dist[position][starts[2*i+o]],2*i+o) for i in left for o in (0,1));left.remove(state//2);sequence.append(state);position=ends[state]
  return sequence
 def route_cost(sequence):
  position=depot;value=0
  for state in sequence:value+=dist[position][starts[state]]+required[state//2]['service_cost'];position=ends[state]
  return value+dist[position][depot]
 def emit(sequences):
  nonlocal best,best_sequences
  value=sum(route_cost(seq) for seq in sequences)
  if value>=best:return
  if sorted(state//2 for seq in sequences for state in seq)!=list(range(n)):return
  if any(sum(required[state//2]['demand'] for state in seq)>Q for seq in sequences):return
  routes=[]
  for k,seq in enumerate(sequences):
   if not seq:continue
   position=depot;dead=Counter()
   for state in seq:dead.update(paths[position,starts[state]]);position=ends[state]
   dead.update(paths[position,depot]);routes.append({'vehicle':k,'serviced_edges':[required[state//2]['edge_id'] for state in seq],'deadheaded_edges':[{'edge_id':e,'times':times} for e,times in dead.items()]})
  best=value;best_sequences=[seq[:] for seq in sequences];solution={'objective_value':value,'routes':routes};json.dump(solution,open(a.solution_path,'w'))
  if hasattr(logger,'log_solution'):logger.log_solution(value,solution)
  else:logger.log(value)
 groups=[[] for _ in range(K)];loads=[0]*K;feasible=True
 for i in sorted(range(n),key=lambda i:-required[i]['demand']):
  choices=[k for k in range(K) if loads[k]+required[i]['demand']<=Q]
  if not choices:feasible=False;break
  k=min(choices,key=lambda k:(Q-loads[k]-required[i]['demand'],k));groups[k].append(i);loads[k]+=required[i]['demand']
 if not feasible:
  from ortools.sat.python import cp_model
  m=cp_model.CpModel();x=[[m.new_bool_var('pack%d_%d'%(i,k)) for k in range(K)] for i in range(n)]
  for row in x:m.add_exactly_one(row)
  for k in range(K):m.add(sum(required[i]['demand']*x[i][k] for i in range(n))<=Q)
  solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.05,min(5.,end-time.monotonic()-1));status=solver.solve(m)
  if status not in (cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible capacitated service partition')
  groups=[[i for i in range(n) if solver.value(x[i][k])] for k in range(K)]
 sequences=[nearest(group) for group in groups];emit(sequences)
 if time.monotonic()>end-1:return
 import numpy as np
 exact_cache={};pair_cache={}
 def subset_routes(items,deadline):
  count=len(items);size=1<<count;states=2*count;dp=np.full((size,states),np.inf);pred=np.full((size,states),-1,dtype=np.int16);demand=np.zeros(size,dtype=np.int32);cost=np.full(size,np.inf);last=np.full(size,-1,dtype=np.int16);cost[0]=0.;local=[2*i+o for i in items for o in (0,1)];travel=np.array([[dist[ends[before]][starts[after]]+required[after//2]['service_cost'] for after in local] for before in local]);home=np.array([dist[ends[state]][depot] for state in local])
  for mask in range(1,size):
   if mask%256==0 and time.monotonic()>deadline:break
   bit=mask&-mask;j=bit.bit_length()-1;demand[mask]=demand[mask^bit]+required[items[j]]['demand']
   if demand[mask]>Q:continue
   left=mask
   while left:
    bit=left&-left;j=bit.bit_length()-1;prev=mask^bit
    for state in (2*j,2*j+1):
     if not prev:dp[mask,state]=dist[depot][starts[local[state]]]+required[items[j]]['service_cost']
     else:
      values=dp[prev]+travel[:,state];before=int(np.argmin(values));dp[mask,state]=values[before];pred[mask,state]=before
    left^=bit
   values=dp[mask]+home;last[mask]=int(np.argmin(values));cost[mask]=values[last[mask]]
  def recover(mask):
   if not mask:return []
   state=int(last[mask]);sequence=[]
   while mask:
    sequence.append(local[state]);previous=int(pred[mask,state]);mask^=1<<(state//2);state=previous
   return sequence[::-1]
  return cost,recover
 def optimize_route(group,deadline):
  key=tuple(sorted(group))
  if key in exact_cache:return exact_cache[key][:]
  if not group:return []
  if len(group)>11:return nearest(group)
  cost,recover=subset_routes(list(key),deadline);mask=(1<<len(key))-1
  result=recover(mask) if np.isfinite(cost[mask]) else nearest(group);exact_cache[key]=result[:];return result
 route_deadline=min(end-1,START+a.time_limit*.10)
 sequences=[optimize_route(group,route_deadline) if time.monotonic()<route_deadline else nearest(group) for group in groups];emit(sequences)
 pair_deadline=min(end-2,START+a.time_limit*.42)
 for round_index in range(4):
  changed=False;pairs=list(itertools.combinations(range(K),2));rng.shuffle(pairs)
  for k,l in pairs:
   if time.monotonic()>pair_deadline:break
   items=sorted({state//2 for state in sequences[k]+sequences[l]})
   if not items or len(items)>16:continue
   key=tuple(items)
   if key not in pair_cache:
    cost,recover=subset_routes(items,pair_deadline);full=(1<<len(items))-1;values=cost+cost[::-1];mask=int(np.argmin(values))
    if not np.isfinite(values[mask]):continue
    pair_cache[key]=(float(values[mask]),recover(mask),recover(full^mask))
   value,left,right=pair_cache[key];old=route_cost(sequences[k])+route_cost(sequences[l])
   if value<old:
    sequences[k]=left[:];sequences[l]=right[:];emit(sequences);changed=True
  if not changed or time.monotonic()>pair_deadline:break
 if n<=45 and K<=8 and time.monotonic()<end-2:
  from ortools.sat.python import cp_model
  model=cp_model.CpModel();options=2*n;skips=[];all_arcs=[];cost_terms=[];leaders=[]
  hint_sequences=sorted(best_sequences,key=lambda seq:min((state//2 for state in seq),default=n))
  for k in range(K):
   sequence=hint_sequences[k]
   if sequence:
    minimum=min(state//2 for state in sequence)
    if next(state for state in sequence if state//2==minimum)%2:sequence=[state^1 for state in sequence[::-1]]
   chosen=set(sequence);route_nodes=[state+1 for state in sequence];hint_arcs=set(zip([0]+route_nodes,route_nodes+[0])) if sequence else {(0,0)};arcs=[];records=[];skip=[]
   empty=model.new_bool_var('empty%d'%k);arcs.append((0,0,empty));model.add_hint(empty,int(not sequence));records.append((0,0,empty))
   for state in range(options):
    loop=model.new_bool_var('skip%d_%d'%(k,state));skip.append(loop);arcs.append((state+1,state+1,loop));model.add_hint(loop,int(state not in chosen))
   skips.append(skip)
   active=sum(1-loop for loop in skip);model.add(active<=n*(1-empty));model.add(active>=1-empty)
   for state in range(options):
    node=state+1
    for tail,head,cost in [(0,node,dist[depot][starts[state]]),(node,0,dist[ends[state]][depot])]:
     var=model.new_bool_var('arc%d_%d_%d'%(k,tail,head));arcs.append((tail,head,var));records.append((tail,head,var));cost_terms.append(cost*var);model.add_hint(var,int((tail,head) in hint_arcs))
    for other in range(options):
     if other//2==state//2:continue
     var=model.new_bool_var('arc%d_%d_%d'%(k,node,other+1));arcs.append((node,other+1,var));records.append((node,other+1,var));cost_terms.append(dist[ends[state]][starts[other]]*var);model.add_hint(var,int((node,other+1) in hint_arcs))
   model.add_circuit(arcs);all_arcs.append(records);model.add(sum(required[state//2]['demand']*(1-skip[state]) for state in range(options))<=Q)
   markers=[]
   for i in range(n):
    selected=2-skip[2*i]-skip[2*i+1];marker=model.new_int_var(i,n,'mark%d_%d'%(k,i));model.add(marker==n-(n-i)*selected);markers.append(marker);model.add(1-skip[2*i+1]<=sum(2-skip[2*j]-skip[2*j+1] for j in range(i)))
   leader=model.new_int_var(0,n,'leader%d'%k);model.add_min_equality(leader,markers);leaders.append(leader)
   if k:model.add(leaders[k-1]<=leader)
  for i in range(n):model.add(sum(2-skips[k][2*i]-skips[k][2*i+1] for k in range(K))==1)
  model.minimize(sum(cost_terms)+sum(e['service_cost'] for e in required))
  class Callback(cp_model.CpSolverSolutionCallback):
   def on_solution_callback(self):
    result=[]
    for records in all_arcs:
     successor={tail:head for tail,head,var in records if self.value(var)};sequence=[];node=successor.get(0,0)
     while node:sequence.append(node-1);node=successor[node]
     result.append(sequence)
    emit(result)
  solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.05,end-time.monotonic()-1);solver.solve(model,Callback())
 else:
  while time.monotonic()<end-1:
   groups=[[state//2 for state in seq] for seq in best_sequences];k,l=rng.sample(range(K),2) if K>1 else (0,0)
   if k==l or not groups[k]:break
   i=rng.choice(groups[k]);j=rng.choice(groups[l]) if groups[l] else None
   if j is None:groups[k].remove(i);groups[l].append(i)
   else:groups[k].remove(i);groups[l].remove(j);groups[k].append(j);groups[l].append(i)
   if any(sum(required[i]['demand'] for i in group)>Q for group in groups):continue
   emit([optimize_route(group,min(end-1,time.monotonic()+.15)) for group in groups])
if __name__=='__main__':main()

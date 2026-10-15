import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[key]='1'
import time
START=time.monotonic()
import argparse,json,heapq
from collections import defaultdict,deque
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser()
 for key in ('instance_path','solution_path','log_path'):p.add_argument('--'+key,required=True)
 p.add_argument('--time_limit',type=int,default=60);a=p.parse_args();end=START+max(.5,a.time_limit-5);d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path);parameters=d['global_parameters'];lam=parameters['lambda'];F=parameters['frequency_upper_bound_F'];max_length=parameters['max_line_length_edges'];nodes=[v['id'] for v in d['network']['nodes']];edges={e['id']:e for e in d['network']['edges']};modes=d['modes'];od=d['od_matrix'];lines=[];line_seen=set();mode_graphs=[];best=float('inf');small=len(nodes)<=50
 def add_line(mode_index,ns,es):
  if not es or len(es)>max_length or len(set(ns))!=len(ns):return
  mode=modes[mode_index];terminals=set(mode['terminals'])
  if ns[0] not in terminals or ns[-1] not in terminals:return
  if ns[0]>ns[-1]:ns=ns[::-1];es=es[::-1]
  key=(mode_index,tuple(es))
  if key in line_seen:return
  line_seen.add(key);lines.append((mode_index,ns,es))
 for mode in modes:
  graph=defaultdict(list)
  for eid in mode['edge_indices']:
   u,v=edges[eid]['endpoints'];graph[u].append((v,eid));graph[v].append((u,eid))
  mode_graphs.append(graph)
 def nearest_terminal(graph,start,terminals,forbidden):
  queue=deque([start]);previous={start:None};finish=None
  while queue:
   u=queue.popleft()
   if u in terminals:finish=u;break
   for v,e in graph[u]:
    if v not in forbidden and v not in previous:previous[v]=(u,e);queue.append(v)
  if finish is None:return None
  ns=[finish];es=[];u=finish
  while previous[u] is not None:u,e=previous[u];ns.append(u);es.append(e)
  return ns[::-1],es[::-1]
 if small:
  for mi,mode in enumerate(modes):
   terminals=set(mode['terminals']);graph=mode_graphs[mi]
   def dfs(ns,es):
    if es and ns[-1] in terminals:add_line(mi,ns,es)
    if len(es)>=max_length or len(lines)>=10000 or time.monotonic()>end-2:return
    for v,e in graph[ns[-1]]:
     if v not in ns:dfs(ns+[v],es+[e])
   for root in sorted(terminals):dfs([root],[])
 else:
  for mi,mode in enumerate(modes):
   terminals=set(mode['terminals']);graph=mode_graphs[mi]
   for eid in mode['edge_indices']:
    if time.monotonic()>START+a.time_limit*.18:break
    u,v=edges[eid]['endpoints']
    if u in terminals and v in terminals:add_line(mi,[u,v],[eid]);continue
    for u,v in [(u,v),(v,u)]:
     first=nearest_terminal(graph,u,terminals,{v})
     if first is None:continue
     prefix_ns,prefix_es=first;prefix_ns=prefix_ns[::-1];prefix_es=prefix_es[::-1];second=nearest_terminal(graph,v,terminals-set(prefix_ns),set(prefix_ns))
     if second is not None:
      suffix_ns,suffix_es=second;add_line(mi,prefix_ns+suffix_ns,prefix_es+[eid]+suffix_es);break
 adj=defaultdict(list);paths=[];path_seen=set();covered={e for _,_,es in lines for e in es};edge_max_vehicle=defaultdict(float)
 for mi,_,es in lines:
  for e in es:edge_max_vehicle[e]=max(edge_max_vehicle[e],modes[mi]['vehicle_capacity'])
 for eid in covered:
  e=edges[eid];u,v=e['endpoints'];tt=e['traveling_time_seconds'];adj[u].append((v,2*eid,tt));adj[v].append((u,2*eid+1,tt))
 def add_path(q,arcs):
  key=(q,tuple(arcs))
  if key not in path_seen:path_seen.add(key);paths.append((q,arcs,sum(edges[arc//2]['traveling_time_seconds'] for arc in arcs)))
 by_origin=defaultdict(list)
 for q,item in enumerate(od):by_origin[item['origin']].append(q)
 def passenger_trees(traffic=None,factor=0):
  for root,indices in by_origin.items():
   dist={root:0.};prev={};heap=[(0.,root)]
   while heap:
    value,u=heapq.heappop(heap)
    if value!=dist[u]:continue
    for v,arc,tt in adj[u]:
     eid=arc//2;penalty=0 if traffic is None else factor*traffic.get(eid,0)/max(1,edge_max_vehicle[eid]*edges[eid]['edge_capacity']);new=value+tt*(1+penalty)
     if new<dist.get(v,float('inf')):dist[v]=new;prev[v]=(u,arc);heapq.heappush(heap,(new,v))
   for q in indices:
    dest=od[q]['destination']
    if dest not in dist:continue
    arcs=[];u=dest
    while u!=root:u,arc=prev[u];arcs.append(arc)
    add_path(q,arcs[::-1])
 passenger_trees()
 if small:
  for q,item in enumerate(od):
   heap=[(0.,[item['origin']],[])];count=0;pops=0
   while heap and count<12 and pops<30000 and time.monotonic()<end-2:
    cost,ns,arcs=heapq.heappop(heap);pops+=1;u=ns[-1]
    if u==item['destination']:add_path(q,arcs);count+=1;continue
    for v,arc,tt in adj[u]:
     if v not in ns:heapq.heappush(heap,(cost+tt,ns+[v],arcs+[arc]))
 import numpy as np
 import highspy
 INF=highspy.kHighsInf
 def solve_pool(deadline,integer=True):
  nonlocal best
  M=len(lines);H=len(paths);N=2*M+H;h=highspy.Highs();h.setOptionValue('output_flag',False);h.setOptionValue('threads',1);h.setOptionValue('parallel','off');lower=np.zeros(N);upper=np.full(N,INF);upper[:M]=1;upper[M:2*M]=F;cost=np.zeros(N);line_by_edge=defaultdict(list);path_by_arc=defaultdict(list);path_by_od=defaultdict(list)
  for l,(mi,ns,es) in enumerate(lines):
   cost[l]=lam*modes[mi]['fixed_cost_per_line'];cost[M+l]=lam*sum(edges[e]['operating_cost'] for e in es)
   for e in es:line_by_edge[e].append(l)
  for j,(q,arcs,tt) in enumerate(paths):
   cost[2*M+j]=(1-lam)*tt;path_by_od[q].append(j)
   for arc in arcs:path_by_arc[arc].append(j)
  h.addVars(N,lower,upper);h.changeColsCost(N,np.arange(N,dtype=np.int32),cost);starts=[0];ci=[];cv=[];rl=[];ru=[]
  def row(entries,l=-INF,u=INF):
   for j,v in entries:ci.append(j);cv.append(v)
   starts.append(len(ci));rl.append(l);ru.append(u)
  for l in range(M):row([(M+l,1.),(l,-F)],u=0.)
  for eid,e in edges.items():
   row([(M+l,1.) for l in line_by_edge[eid]],u=e['edge_capacity'])
   for arc in (2*eid,2*eid+1):row([(2*M+j,1.) for j in path_by_arc[arc]]+[(M+l,-modes[lines[l][0]]['vehicle_capacity']) for l in line_by_edge[eid]],u=0.)
  for q,item in enumerate(od):row([(2*M+j,1.) for j in path_by_od[q]],item['demand'],item['demand'])
  h.addRows(len(rl),np.asarray(rl),np.asarray(ru),len(ci),np.asarray(starts,dtype=np.int32),np.asarray(ci,dtype=np.int32),np.asarray(cv))
  def emit(v):
   nonlocal best
   active=[l for l in range(M) if v[M+l]>1e-8];flows=[j for j in range(H) if v[2*M+j]>1e-8];frequency=defaultdict(float);capacity=defaultdict(float);arc_flow=defaultdict(float);served=[0.]*len(od)
   for l in active:
    if v[M+l]>F+1e-6:return False
    for eid in lines[l][2]:frequency[eid]+=v[M+l];capacity[eid]+=modes[lines[l][0]]['vehicle_capacity']*v[M+l]
   for j in flows:
    q,arcs,tt=paths[j];served[q]+=v[2*M+j]
    for arc in arcs:arc_flow[arc]+=v[2*M+j]
   if any(frequency[eid]>edges[eid]['edge_capacity']+1e-6 for eid in frequency):return False
   if any(abs(served[q]-item['demand'])>1e-5 for q,item in enumerate(od)):return False
   if any(flow>capacity[arc//2]+1e-5 for arc,flow in arc_flow.items()):return False
   obj=sum(cost[l]+cost[M+l]*v[M+l] for l in active)+sum(cost[2*M+j]*v[2*M+j] for j in flows)
   if obj<best:
    best=obj;solution={'objective_value':float(obj),'active_lines':[{'line_index':l,'mode':modes[lines[l][0]]['name'],'nodes':lines[l][1],'edges':lines[l][2],'frequency':float(v[M+l])} for l in active],'active_passenger_paths':[{'origin':od[paths[j][0]]['origin'],'destination':od[paths[j][0]]['destination'],'arcs':paths[j][1],'flow':float(v[2*M+j])} for j in flows]};json.dump(solution,open(a.solution_path,'w'))
    if hasattr(logger,'log_solution'):logger.log_solution(float(obj),solution)
    else:logger.log(float(obj))
   return True
  if time.monotonic()>deadline-.1:return False
  h.setOptionValue('time_limit',max(.05,min(8.,deadline-time.monotonic()-.1)));h.run();s=h.getSolution();feasible=s.value_valid and emit(s.col_value)
  if not integer or not feasible or time.monotonic()>deadline-.5:return feasible
  seed=np.asarray(s.col_value);seed[:M]=[int(seed[M+l]>1e-8) for l in range(M)];h.changeColsIntegrality(M,np.arange(M,dtype=np.int32),np.ones(M,dtype=np.uint8));h.setSolution(N,np.arange(N,dtype=np.int32),seed);h.setOptionValue('time_limit',h.getRunTime()+max(.05,deadline-time.monotonic()-.2));h.setOptionValue('mip_rel_gap',.01 if not small else 1e-5);h.run();s=h.getSolution()
  if s.value_valid:emit(s.col_value)
  return feasible
 if small:solve_pool(end-1);return
 traffic=defaultdict(float)
 for q,arcs,tt in paths:
  for arc in arcs:traffic[arc//2]+=od[q]['demand']
 first=solve_pool(min(end-2,START+a.time_limit*.38),False)
 # Longer homogeneous terminal paths amortize fixed line costs.
 for mi,mode in enumerate(modes):
  terminals=set(mode['terminals']);graph=mode_graphs[mi]
  for root in sorted(terminals):
   if len(lines)>=6000 or time.monotonic()>START+a.time_limit*.52:break
   dist={root:0.};prev={};heap=[(0.,root)]
   while heap:
    value,u=heapq.heappop(heap)
    if value!=dist[u]:continue
    for v,eid in graph[u]:
     new=value+edges[eid]['traveling_time_seconds']
     if new<dist.get(v,float('inf')):dist[v]=new;prev[v]=(u,eid);heapq.heappush(heap,(new,v))
   candidates=[]
   for terminal in terminals:
    if terminal<=root or terminal not in dist:continue
    ns=[terminal];es=[];u=terminal
    while u!=root and len(es)<=max_length:u,eid=prev[u];ns.append(u);es.append(eid)
    if u==root and 2<=len(es)<=max_length:
     score=sum(traffic.get(e,0) for e in es)/len(es)**.25;candidates.append((score,ns[::-1],es[::-1]))
   for score,ns,es in sorted(candidates,reverse=True)[:5]:add_line(mi,ns,es)
 covered={e for _,_,es in lines for e in es};adj=defaultdict(list)
 for mi,_,es in lines:
  for eid in es:edge_max_vehicle[eid]=max(edge_max_vehicle[eid],modes[mi]['vehicle_capacity'])
 for eid in covered:
  e=edges[eid];u,v=e['endpoints'];tt=e['traveling_time_seconds'];adj[u].append((v,2*eid,tt));adj[v].append((u,2*eid+1,tt))
 passenger_trees(traffic,3.)
 if time.monotonic()<end-3:passenger_trees(traffic,12.)
 solve_pool(end-1)
 if best==float('inf'):raise RuntimeError('No feasible complete transit line and passenger-flow certificate in bounded pool')
if __name__=='__main__':main()

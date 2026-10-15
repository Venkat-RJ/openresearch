import argparse,json,time,itertools
from ortools.sat.python import cp_model

def solve(d,t):
 deadline=time.monotonic()+max(.1,t-5);vertices=d['vertices'];n=len(vertices);index={v:i for i,v in enumerate(vertices)};edges=sorted({tuple(sorted((index[u],index[v]))) for u,v in d['edges']});budget=d['interdiction_budget_k'];adj=[0]*n
 for u,v in edges:adj[u]|=1<<v;adj[v]|=1<<u
 def maximum(a):
  best=[]
  def expand(chosen,p):
   nonlocal best
   order=[];bounds=[];color=0;remaining=p
   while remaining:
    color+=1;available=remaining
    while available:
     bit=available&-available;v=bit.bit_length()-1;remaining^=bit;available^=bit;available&=~a[v];order.append(v);bounds.append(color)
   for h in range(len(order)-1,-1,-1):
    if len(chosen)+bounds[h]<=len(best):return
    v=order[h];q=p&a[v]
    if q:expand(chosen+[v],q)
    elif len(chosen)+1>len(best):best=chosen+[v]
    p&=~(1<<v)
  expand([], (1<<n)-1)
  return len(best)
 base=maximum(adj);bestvalue=base;bestremoved=set()
 for target in range(base-1,0,-1):
  if time.monotonic()>=deadline:break
  cliques=[];aborted=False
  def enumerate_cliques(chosen,p):
   nonlocal aborted
   if len(chosen)==target+1:cliques.append(tuple(chosen));return
   if p.bit_count()<target+1-len(chosen):return
   if len(cliques)>200000 or time.monotonic()>=deadline:aborted=True;return
   while p and not aborted:
    bit=p&-p;p^=bit;v=bit.bit_length()-1;enumerate_cliques(chosen+[v],p&adj[v])
  enumerate_cliques([], (1<<n)-1)
  if aborted:break
  incidence={};clique_edges=[]
  for h,clique in enumerate(cliques):
   row=list(itertools.combinations(clique,2));clique_edges.append(row)
   for e in row:incidence[e]=incidence.get(e,0)|(1<<h)
  uncovered=(1<<len(cliques))-1;removed=set()
  while uncovered and len(removed)<budget and time.monotonic()<deadline:
   edge=max(incidence,key=lambda e:(incidence[e]&uncovered).bit_count());removed.add(edge);uncovered&=~incidence[edge]
  if uncovered:
   if time.monotonic()>=deadline:break
   model=cp_model.CpModel();z={e:model.new_bool_var('e'+str(i)) for i,e in enumerate(incidence)};model.add(sum(z.values())<=budget)
   for row in clique_edges:model.add(sum(z[e] for e in row)>=1)
   for e in z:model.add_hint(z[e],int(e in removed))
   solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.1,deadline-time.monotonic());status=solver.solve(model)
   if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):break
   removed={e for e in z if solver.value(z[e])}
  a=adj[:]
  for u,v in removed:a[u]&=~(1<<v);a[v]&=~(1<<u)
  value=maximum(a)
  if value<bestvalue:bestvalue=value;bestremoved=removed
 return {'objective_value':bestvalue,'interdicted_edges':[[vertices[u],vertices[v]] for u,v in sorted(bestremoved)]}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

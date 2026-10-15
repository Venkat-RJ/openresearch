import argparse,json,os,time
os.environ['OMP_NUM_THREADS']='1'
from ortools.sat.python import cp_model

def solve(d,limit,solution_path=None):
 deadline=time.monotonic()+max(.01,limit-1)
 clusters=d['partition']['clusters'];P=len(clusters);n=d['graph']['num_vertices'];edges=d['graph']['edges'];adj=[set() for _ in range(n)];bits=[0]*n
 for u,v in edges:adj[u].add(v);adj[v].add(u);bits[u]|=1<<v;bits[v]|=1<<u
 selected={};colors={};owner={v:p for p,g in enumerate(clusters) for v in g}
 for p,g in enumerate(clusters):
  opts=[]
  for v in g:
   used={colors[u] for u in adj[v] if u in colors};c=next(k for k in range(P) if k not in used);opts.append((c,len(adj[v]),v))
  c,_,v=min(opts);selected[p]=v;colors[v]=c
 def output(sel,col):return {'objective_value':len(set(col.values())),'selected_vertices':{str(p):v for p,v in sel.items()},'coloring':{str(v):c for v,c in col.items()}}
 best=output(selected,colors)
 def persist(out):
  if solution_path:
   with open(solution_path,'w') as f:json.dump(out,f)
 persist(best)
 weights=[0]*n;remaining=set(range(n));mcs=[]
 while remaining:
  v=max(remaining,key=lambda v:weights[v]);remaining.remove(v);mcs.append(v)
  for u in adj[v]&remaining:weights[u]+=1
 peo=list(reversed(mcs));later=(1<<n)-1;cliques=[];chordal=True
 for v in peo:
  later&=~(1<<v);nb=bits[v]&later;r=nb
  while r:
   b=r&-r;r-=b;u=b.bit_length()-1
   if (nb&~b)&~bits[u]:chordal=False;break
  if not chordal:break
  cliques.append(nb|1<<v)
 B=max(colors.values())+1;model=cp_model.CpModel();chosen={v:model.new_bool_var(f'x{v}') for v in owner};z=model.new_int_var(0,B if chordal else B-1,'color_bound')
 for p,g in enumerate(clusters):
  model.add_exactly_one(chosen[v] for v in g)
  for v in g:model.add_hint(chosen[v],int(v==selected[p]))
 if chordal:
  maximal=[]
  for mask in sorted(set(cliques),key=int.bit_count,reverse=True):
   if not any(mask&other==mask for other in maximal):maximal.append(mask)
  for mask in maximal:
   vertices=[];r=mask
   while r:b=r&-r;r-=b;vertices.append(b.bit_length()-1)
   model.add(sum(chosen[v] for v in vertices)<=z)
  model.add_hint(z,B)
  def color_selected(sel):
   chosen_set=set(sel.values());col={}
   for v in reversed(peo):
    if v not in chosen_set:continue
    used={col[u] for u in adj[v] if u in col};col[v]=next(c for c in range(P) if c not in used)
   return col
 else:
  color=[model.new_int_var(0,B-1,f'c{p}') for p in range(P)]
  for p,g in enumerate(clusters):model.add(color[p]<=z);model.add_hint(color[p],colors[selected[p]])
  model.add(color[0]==0);model.add_hint(z,B-1)
  for ix,(u,v) in enumerate(edges):
   a,b=owner[u],owner[v]
   if a!=b:model.add(color[a]!=color[b]).only_enforce_if([chosen[u],chosen[v]])
   if ix%1024==0 and time.monotonic()>=deadline:return best
 model.minimize(z)
 class Incumbent(cp_model.CpSolverSolutionCallback):
  def on_solution_callback(self):
   nonlocal best
   sel={p:next(v for v in g if self.value(chosen[v])) for p,g in enumerate(clusters)};col=color_selected(sel) if chordal else {v:self.value(color[p]) for p,v in sel.items()};out=output(sel,col)
   if out['objective_value']<=best['objective_value']:best=out;persist(best)
 s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());st=s.solve(model,Incumbent())
 if st in (cp_model.FEASIBLE,cp_model.OPTIMAL):
  sel={p:next(v for v in g if s.value(chosen[v])) for p,g in enumerate(clusters)};col=color_selected(sel) if chordal else {v:s.value(color[p]) for p,v in sel.items()};out=output(sel,col)
  if out['objective_value']<=best['objective_value']:best=out
 return best
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 out=solve(json.load(open(a.instance_path)),a.time_limit,a.solution_path)
 with open(a.solution_path,'w') as f:json.dump(out,f)

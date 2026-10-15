import argparse,json,time,random,itertools
from ortools.sat.python import cp_model

def solve(d,t):
 g=d['graph'];vs=g['vertices'];n=len(vs);p=d['problem_settings'];s=p['parameter_s'];E={tuple(sorted(e)) for e in g['edges']};adj={v:set() for v in vs}
 for u,v in E:adj[u].add(v);adj[v].add(u)
 def valid(group):
  S=set(group)
  if any(len(S-({v}|adj[v]))>s-1 for v in group):return False
  if p['connectivity_required'] and len(group)>1:
   seen={group[0]};todo=[group[0]]
   while todo:
    v=todo.pop();new=(adj[v]&S)-seen;seen|=new;todo.extend(new)
   if seen!=S:return False
  return True
 deadline=time.monotonic()+max(.1,t-1);groups={tuple([v]) for v in vs};rng=random.Random(3)
 if n<=20:
  for size in range(2,n+1):
   for group in itertools.combinations(vs,size):
    if time.monotonic()>deadline-t*.45:break
    if valid(group):groups.add(group)
   if time.monotonic()>deadline-t*.45:break
 else:
  for run in range(1000):
   if time.monotonic()>deadline-t*.45:break
   order=vs.copy();rng.shuffle(order)
   for v in order[:min(20,n)]:
    group=[v]
    for u in order:
     if u!=v and valid(group+[u]):group.append(u)
    groups.add(tuple(sorted(group)))
 groups=sorted(groups,key=lambda g:(-len(g),g));m=cp_model.CpModel();x=[m.new_bool_var(str(i)) for i in range(len(groups))]
 for v in vs:
  expr=sum(x[i] for i,g in enumerate(groups) if v in g)
  if p['decomposition_type']=='partitioning':m.add(expr==1)
  else:m.add(expr>=1)
 m.minimize(sum(x));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.1,deadline-time.monotonic());r=solver.solve(m)
 if r not in (cp_model.OPTIMAL,cp_model.FEASIBLE):chosen=[[v] for v in vs]
 else:chosen=[list(g) for i,g in enumerate(groups) if solver.value(x[i])]
 return {'objective_value':len(chosen),'solution':chosen}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

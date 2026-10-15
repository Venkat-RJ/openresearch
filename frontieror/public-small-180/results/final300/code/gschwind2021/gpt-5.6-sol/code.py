import argparse,json,time,random
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-3);z=json.load(open(a.instance_path));g=z['graph'];vertices=g['vertices'];adj={v:set() for v in vertices}
for u,v in g['edges']:adj[u].add(v);adj[v].add(u)
settings=z['problem_settings'];s=settings['parameter_s'];connected=settings['connectivity_required'];rng=random.Random(0);best=len(vertices)+1;best_groups=None;logger=SolutionLogger(a.log_path,sense='minimize')
logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
def valid(group):
 if any(len(adj[v]&group)<len(group)-s for v in group):return False
 if connected and group:
  seen={next(iter(group))};front=seen.copy()
  while front:
   v=front.pop();new=(adj[v]&group)-seen;seen|=new;front|=new
  return seen==group
 return True
def save(groups):
 global best,best_groups
 groups=[q for q in groups if q]
 if len(groups)>=best:return
 best=len(groups);best_groups=[q.copy() for q in groups];solution={'objective_value':best,'solution':[sorted(q) for q in groups],'runtime':time.monotonic()-start};json.dump(solution,open(a.solution_path,'w'));logger.log_solution(best,solution)
def construct(order):
 groups=[]
 for v in order:
  options=[(len(q),i) for i,q in enumerate(groups) if valid(q|{v})]
  if options:groups[max(options)[1]].add(v)
  else:groups.append({v})
 return groups
def improve(groups):
 save(groups);changed=True
 while changed and time.monotonic()<end:
  changed=False;groups=[q for q in groups if q]
  for i in range(len(groups)):
   for j in range(i+1,len(groups)):
    if groups[i] and groups[j] and valid(groups[i]|groups[j]):groups[i]|=groups[j];groups[j]=set();changed=True;save(groups)
  groups=[q for q in groups if q]
  for i in sorted(range(len(groups)),key=lambda i:len(groups[i])):
   others=[q.copy() for j,q in enumerate(groups) if j!=i];ok=True
   for v in sorted(groups[i],key=lambda v:len(adj[v]),reverse=True):
    targets=[j for j,q in enumerate(others) if valid(q|{v})]
    if not targets:ok=False;break
    others[max(targets,key=lambda j:len(others[j]))].add(v)
   if ok:groups=others;changed=True;save(groups);break
 save(groups)
 return groups
improve(construct(sorted(vertices,key=lambda v:len(adj[v]),reverse=True)))
while time.monotonic()<end:
 order=vertices.copy();rng.shuffle(order);improve(construct(order))

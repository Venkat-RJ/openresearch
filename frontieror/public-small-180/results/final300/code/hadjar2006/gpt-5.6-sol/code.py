import os
license=os.environ.get('GRB_LICENSE_FILE')
if license and not os.path.exists(license):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
import argparse,json,time,random
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-4);z=json.load(open(a.instance_path));tasks=[t['id'] for t in z['tasks']];depots=[d['id'] for d in z['depots']];cap={d['id']:d['capacity'] for d in z['depots']};cost={(q['task_i'],q['task_j']):q['cost'] for q in z['compatible_pairs']};out={(q['depot_id'],q['task_id']):q['cost'] for q in z['depot_to_task_costs']};back={(q['task_id'],q['depot_id']):q['cost'] for q in z['task_to_depot_costs']};successors={i:[j for j in tasks if (i,j) in cost] for i in tasks};rng=random.Random(0);pool={}
def add(route):
 for k in depots:
  if (k,route[0]) in out and (route[-1],k) in back:
   value=out[k,route[0]]+back[route[-1],k]+sum(cost[i,j] for i,j in zip(route,route[1:]));pool[k,tuple(route)]=value
for i in tasks:add([i])
for iteration in range(100):
 if time.monotonic()>=start+max(.1,a.time_limit*.25) or len(pool)>=1000:break
 order=tasks.copy()
 if iteration:rng.shuffle(order)
 lists={i:sorted(successors[i],key=lambda j:cost[i,j]+(0 if not iteration else rng.random()*100)) for i in tasks};matched={}
 def augment(i,seen):
  for j in lists[i]:
   if j in seen:continue
   seen.add(j)
   if j not in matched or augment(matched[j],seen):matched[j]=i;return True
  return False
 for i in order:augment(i,set())
 succ={i:j for j,i in matched.items()}
 for first in tasks:
  if first in matched:continue
  route=[first]
  while route[-1] in succ:route.append(succ[route[-1]])
  add(route)
entries=list(pool.items());model=gp.Model('route_master');model.Params.Seed=0;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;model.Params.Threads=1;model.Params.TimeLimit=max(.1,end-time.monotonic());model.Params.OutputFlag=0
cover={i:model.addConstr(gp.LinExpr()==1) for i in tasks};limits={k:model.addConstr(gp.LinExpr()<=cap[k]) for k in depots};x={};seen=set()
def insert(entry):
 j=len(x);(k,route),value=entry;col=gp.Column([1]*(len(route)+1),[cover[i] for i in route]+[limits[k]]);x[j]=model.addVar(obj=value,column=col);seen.add((k,route))
for entry in entries:insert(entry)
ordered=[q['id'] for q in sorted(z['tasks'],key=lambda q:q['start_time'])]
for iteration in range(60):
 if time.monotonic()>=start+a.time_limit*.4:break
 model.Params.TimeLimit=max(.1,start+a.time_limit*.4-time.monotonic());model.optimize()
 if model.Status!=gp.GRB.OPTIMAL:break
 dual={i:cover[i].Pi for i in tasks};fresh=[]
 for k in depots:
  distances={};paths={}
  for j in ordered:
   if (k,j) in out:distances[j]=out[k,j]-dual[j];paths[j]=(j,)
   for i in ordered:
    if i in distances and (i,j) in cost:
     candidate=distances[i]+cost[i,j]-dual[j]
     if candidate<distances.get(j,float('inf')):distances[j]=candidate;paths[j]=paths[i]+(j,)
   if j in distances and (j,k) in back:
    reduced=distances[j]+back[j,k]-limits[k].Pi;route=paths[j]
    if reduced<-1e-6 and (k,route) not in seen:
     value=out[k,route[0]]+back[route[-1],k]+sum(cost[i,u] for i,u in zip(route,route[1:]));fresh.append((reduced,((k,route),value)))
 if not fresh:break
 if len(x)+len(fresh)>1900:
  keep=[j for j in x if x[j].X>1e-7 or len(entries[j][0][1])==1];remaining=sorted([j for j in x if j not in keep],key=lambda j:x[j].RC);keep+=remaining[:max(0,1300-len(keep))]
  model.remove([x[j] for j in x if j not in keep]);model.update();entries=[entries[j] for j in keep];x={i:x[j] for i,j in enumerate(keep)};seen={entry[0] for entry in entries}
 for reduced,entry in sorted(fresh):
  if len(x)>=1900:break
  if entry[0] in seen:continue
  entries.append(entry);insert(entry)
model.update()
for var in x.values():var.VType=gp.GRB.BINARY
model.Params.OutputFlag=1;model.Params.TimeLimit=max(.1,end-time.monotonic())
logger=SolutionLogger(a.log_path,sense='minimize');best=float('inf')
logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
def save(values):
 global best
 chosen=[j for j,v in enumerate(values) if v>.5];value=sum(entries[j][1] for j in chosen)
 if value>=best:return
 best=value;routes=[{'depot':entries[j][0][0],'tasks':list(entries[j][0][1]),'cost':entries[j][1]} for j in chosen];s={'objective_value':value,'routes':routes,'runtime':time.monotonic()-start};json.dump(s,open(a.solution_path,'w'));logger.log_solution(value,s)
def cb(model,where):
 if where==gp.GRB.Callback.MIPSOL:save(model.cbGetSolution(x).values())
model.optimize(cb)
if model.SolCount:save([x[j].X for j in range(len(entries))])

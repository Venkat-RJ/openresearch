import argparse,json,os,time,random,itertools,math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic();deadline=start+max(.1,a.time_limit-2)
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));n=d['parameters']['n_customers'];K=d['vehicle']['K'];H=d['vehicle']['H'];W=d['vehicle']['W'];Q=d['vehicle']['Q'];dist=d['distance_matrix'];customers={c['id']:c for c in d['customers']};rng=random.Random(0);cache={};pool={};logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 best=float('inf');best_routes=None
 def pack(items):
  if sum(h*w for h,w,q,p in items)>H*W or sum(q for h,w,q,p in items)>Q:return False
  placed=[]
  for h,w,q,pos in sorted(items,key=lambda t:-t[3]):
   found=None
   for y in sorted({0}|{py+ph for px,py,pw,ph,pd in placed}):
    if y+h>H:continue
    for x in sorted({0}|{px+pw for px,py,pw,ph,pd in placed}):
     if x+w>W:continue
     if any(x<px+pw and x+w>px and y<py+ph and y+h>py for px,py,pw,ph,pd in placed):continue
     if any(pd>pos and px<x+w and px+pw>x and py+ph>y+h and py<H for px,py,pw,ph,pd in placed):continue
     found=(x,y,w,h,pos);break
    if found:break
   if found is None:return False
   placed.append(found)
  return True
 def feasible(route):
  key=min(tuple(route),tuple(reversed(route)))
  if key in cache:return cache[key]
  for ordering in (route,list(reversed(route))):
   choices=[[(r['height'],r['width'],r['weight'],pos) for r in item['realizations']] for pos,c in enumerate(ordering,1) for item in customers[c]['items']]
   valid=True
   for scenario in itertools.product(*choices):
    if time.monotonic()>=deadline:return False
    if not pack(list(scenario)):valid=False;break
   if valid:cache[key]=True;return True
  cache[key]=False;return False
 def cost(route):return sum(dist[i][j] for i,j in zip([0]+list(route),list(route)+[0]))
 def improve(route):
  route=list(route);old=cost(route)
  while time.monotonic()<search_end:
   changed=False
   for i in range(len(route)-1):
    for j in range(i+2,len(route)+1):
     candidate=route[:i]+list(reversed(route[i:j]))+route[j:];value=cost(candidate)
     if value<old-1e-8 and feasible(candidate):route=candidate;old=value;changed=True;break
    if changed:break
   if not changed:break
  return route
 def save(routes):
  nonlocal best,best_routes
  obj=sum(cost(r) for r in routes)
  if obj<best:
   best=obj;best_routes=[list(r) for r in routes];s={'objective_value':obj,'routes':best_routes};json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(obj,s)
 attempt=0;search_end=start+max(.1,(a.time_limit-2)*.65)
 while time.monotonic()<search_end:
  if best_routes and attempt%3:
   removed=set(rng.sample(range(1,n+1),min(n,3+attempt%5)));routes=[[c for c in r if c not in removed] for r in best_routes];order=list(removed)
  else:order=list(range(1,n+1));routes=[[] for _ in range(K)]
  rng.shuffle(order);success=True
  for c in order:
   choices=[]
   for k,r in enumerate(routes):
    for pos in range(len(r)+1):
     candidate=r[:pos]+[c]+r[pos:]
     if feasible(candidate):
      value=cost(candidate);choices.append((value-cost(r)+rng.random()*(10 if attempt else 0),k,candidate))
      if len(candidate)>=2:pool[tuple(candidate)]=value
   if not choices:success=False;break
   _,k,r=min(choices);routes[k]=r
  if success and all(len(r)>=2 for r in routes):
   save(routes);routes=[improve(r) for r in routes];save(routes)
   for r in routes:pool[tuple(r)]=cost(r)
  attempt+=1
  if len(pool)>10000:
   keep={tuple(r) for r in best_routes} if best_routes else set();ranked=sorted(pool,key=lambda r:pool[r]/len(r));keep.update(ranked[:6000]);pool={r:pool[r] for r in keep}
 if best_routes is None:raise RuntimeError('No robust packing route set found')
 mandatory={tuple(r) for r in best_routes};ranked=sorted((r for r in pool if r not in mandatory),key=lambda r:pool[r]/len(r));columns=list(mandatory)+ranked[:1900-len(mandatory)];model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;x=model.addVars(len(columns),vtype=GRB.BINARY)
 for c in range(1,n+1):model.addConstr(gp.quicksum(x[h] for h,r in enumerate(columns) if c in r)==1)
 model.addConstr(x.sum()==K);model.setObjective(gp.quicksum(pool[r]*x[h] for h,r in enumerate(columns)));warm={tuple(r) for r in best_routes}
 for h,r in enumerate(columns):x[h].Start=int(r in warm)
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:save([r for h,r in enumerate(columns) if md.cbGetSolution(x[h])>.5])
 model.Params.TimeLimit=max(.1,deadline-time.monotonic());model.optimize(cb)
 if model.SolCount:save([r for h,r in enumerate(columns) if x[h].X>.5])
 final={'objective_value':best,'routes':best_routes};json.dump(final,open(a.solution_path,'w'))
 if logger:logger.log_solution(best,final)
if __name__=='__main__':main()

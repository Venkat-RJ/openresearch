import argparse,json,time,math,random
wall_started=time.time()
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger

def main():
 a=argparse.ArgumentParser(); a.add_argument('--instance_path'); a.add_argument('--solution_path'); a.add_argument('--time_limit',type=int,default=60); a.add_argument('--log_path'); a=a.parse_args(); start=time.monotonic(); end=start+max(.1,a.time_limit-1); d=json.load(open(a.instance_path)); customers=d['customers']; byid={c['id']:c for c in customers}; depots=d['depots']; D=d['d']; P=d['t']; M=d['m']; Q=d['vehicle_capacity']; T=d['route_duration_limit']; coords=[(v['x'],v['y']) for v in depots]+[(v['x'],v['y']) for v in customers]; dist=[[math.dist(u,v) for v in coords] for u in coords]; rng=random.Random(0); logger=configured_logger(a.log_path,sense='minimize') if a.log_path else None; best=float('inf')
 def cost(o,r): return sum(dist[u][v] for u,v in zip([o]+[D+i for i in r],[D+i for i in r]+[o]))
 def feasible(o,r): return sum(byid[i]['demand'] for i in r)<=Q and cost(o,r)+sum(byid[i]['service_duration'] for i in r)<=T+1e-8
 iteration=0
 while time.monotonic()<end:
  routes={(o,p):[] for o in range(D) for p in range(1,P+1)}; assignments={}; order=sorted(customers,key=lambda c:-(c['demand']*c['frequency'])) if iteration==0 else rng.sample(customers,len(customers)); valid=True
  for c in order:
   cid=c['id']; options=[]
   for o in range(D):
    for pat in c['pattern_list']:
     total=0; placements=[]
     for p in pat:
      opts=[]
      for ri,r in enumerate(routes[o,p]):
       for k in range(len(r)+1):
        new=r[:k]+[cid]+r[k:]
        if feasible(o,new): opts.append((cost(o,new)-cost(o,r),ri,new))
      if len(routes[o,p])<M and feasible(o,[cid]): opts.append((cost(o,[cid]),len(routes[o,p]),[cid]))
      if not opts: break
      delta,ri,new=min(opts); total+=delta; placements.append((p,ri,new))
     else: options.append((total*(1+rng.random()*.08*(iteration%3)),o,pat,placements))
   if not options: valid=False; break
   _,o,pat,placements=min(options); assignments[str(cid)]={'depot':o,'pattern':pat}
   for p,ri,new in placements:
    if ri==len(routes[o,p]): routes[o,p].append(new)
    else: routes[o,p][ri]=new
  if valid:
   for (o,p),rs in routes.items():
    for ri,r in enumerate(rs):
     changed=True
     while changed:
      changed=False; old=cost(o,r)
      for i in range(len(r)):
       for j in range(i+2,len(r)+1):
        new=r[:i]+list(reversed(r[i:j]))+r[j:]
        if cost(o,new)<old-1e-8: r=new; old=cost(o,r); changed=True
      if time.monotonic()>=end: break
     rs[ri]=r
   z=sum(cost(o,r) for (o,p),rs in routes.items() for r in rs)
   if z<best:
    best=z; s={'objective_value':z,'routes':[{'depot':o,'period':p,'vehicle':k+1,'customers':r} for (o,p),rs in routes.items() for k,r in enumerate(rs)],'assignments':assignments}; json.dump(s,open(a.solution_path,'w')); logger and logger.log_solution(z,s)
  iteration+=1
if __name__=='__main__': main()

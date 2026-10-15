import argparse,json,time,random
import numpy as np

def solve(d,t):
 K=d['num_sites'];T=d['num_periods'];p=d['parameters'];eps=p['epsilon'];delta=p['delta'];sites=d['sites'];customers=d['customers'];week=np.array([p['weekly_attraction_cycle'][h%7] for h in range(T)]);pop=np.array([c['population'] for c in customers]);selfa=np.array([c['num_businesses']/p['avg_businesses_customers'] for c in customers])[:,None]*week;alpha=np.zeros((len(customers),K,T));dist=np.array([[d['site_to_site_distances'][f'{i+1}_{j+1}'] for j in range(K)] for i in range(K)])
 for i,c in enumerate(customers):
  for k,s in enumerate(sites):alpha[i,k]=week*s['num_businesses']/p['avg_businesses_sites']/max(1e-10,d['customer_to_site_distances'][f'{c["id"]}_{s["id"]}']/1000)**p['q']
 cap=np.array([s['capacity'] for s in sites]);rev=np.array([s['revenue_per_customer'] for s in sites]);fixed=np.array([s['fixed_cost'] for s in sites]);mx=np.array([s['max_events'] for s in sites]);deadline=time.monotonic()+max(.1,t-1)
 def evaluate(route,event):
  if T>1 and np.any(dist[route[:-1],route[1:]]>p['L']+1e-8):return -float('inf'),None
  if any(event[h:h+delta+2].sum()>delta for h in range(T-delta-1)):return -float('inf'),None
  if any(sum(event[h] for h in range(T) if route[h]==k)>mx[k] for k in range(K)):return -float('inf'),None
  attraction=alpha[:,route,np.arange(T)]*event
  denom=selfa.copy()
  for h in range(T):denom[:,h]+=attraction[:,max(0,h-eps):min(T,h+eps+1)].sum(axis=1)
  attendance=np.minimum(cap[route],np.sum(pop[:,None]*attraction/denom,axis=0))
  profit=float(np.dot(rev[route],attendance)-np.dot(fixed[route],event)-p['c']*np.sum(dist[route[:-1],route[1:]]));return profit,attendance
 rng=random.Random(17);best=(-float('inf'),None,None,None)
 for k in range(K):
  route=np.full(T,k);ev=np.zeros(T,dtype=int);v,att=evaluate(route,ev)
  if v>best[0]:best=(v,route.copy(),ev.copy(),att)
 iterations=0;route=best[1].copy();ev=best[2].copy();cur=best[0]
 while time.monotonic()<deadline:
  rr=route.copy();ee=ev.copy();h=rng.randrange(T)
  if rng.random()<.65:ee[h]^=1
  else:rr[h]=rng.randrange(K)
  v,att=evaluate(rr,ee)
  if v>cur+1e-8:route=rr;ev=ee;cur=v
  if v>best[0]:best=(v,rr.copy(),ee.copy(),att)
  iterations+=1
  if iterations%3000==0:
   route=np.full(T,rng.randrange(K));ev=np.zeros(T,dtype=int);cur=0
 v,route,ev,att=best;attendance={f'{k+1}_{h+1}':float(att[h]) if route[h]==k else 0 for k in range(K) for h in range(T)}
 return {'objective_value':v,'tour':{str(h+1):int(route[h]+1) for h in range(T)},'events':{str(h+1):int(ev[h]) for h in range(T)},'event_sites':{str(h+1):int(route[h]+1) for h in range(T) if ev[h]},'attendance':attendance}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

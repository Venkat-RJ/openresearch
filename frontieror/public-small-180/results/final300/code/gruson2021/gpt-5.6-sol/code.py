import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import argparse,json,time,random
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-4);z=json.load(open(a.instance_path));T=z['num_periods'];W=z['num_warehouses'];R=z['num_retailers'];S=z['num_scenarios'];F=1+W+R;r2w=np.array(z['retailer_to_warehouse']);dem=np.array(z['demands'],dtype=float);prob=np.array(z['scenario_probabilities']);mean=np.tensordot(prob,dem,axes=(0,0));setup=np.array([z['setup_costs_plant']]+z['setup_costs_warehouses']+z['setup_costs_retailers']);h0=np.broadcast_to(np.array(z['holding_cost_plant']), (T,));h1=np.array(z['holding_cost_warehouse']);h1=np.full((W,T),float(h1)) if h1.ndim==0 else np.repeat(h1[:,None],T,axis=1) if h1.ndim==1 else h1;h2=np.array(z['holding_costs_retailers']);H0=np.r_[0,np.cumsum(h0)[:-1]];H1=np.c_[np.zeros(W),np.cumsum(h1,axis=1)[:,:-1]];times=np.arange(T);logger=SolutionLogger(a.log_path,sense='minimize');best=float('inf');best_y=None;rng=random.Random(0)
logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
def objective(y):
 pcost=H0+np.minimum.accumulate(np.where(y[0],-H0,np.inf));wcost=H1+np.minimum.accumulate(np.where(y[1:1+W],pcost[None,:]-H1,np.inf),axis=1);retcost=wcost[r2w];cost=h2[:,None]*times+np.minimum.accumulate(np.where(y[1+W:],retcost-h2[:,None]*times,np.inf),axis=1)
 return float(np.sum(setup*y)+np.sum(mean*cost))
def save(y):
 global best,best_y
 value=objective(y)
 if value>=best-1e-8:return
 best=value;best_y=y.copy();s={'objective_value':value,'setup_variables':{f'y_{i}_{t}':int(y[i,t]) for i in range(F) for t in range(T)},'production_plant':{},'delivery_warehouse':{},'delivery_retailer':{},'inventory_plant':{},'inventory_warehouse':{},'inventory_retailer':{},'runtime':time.monotonic()-start}
 xp=np.zeros((S,T));xw=np.zeros((S,W,T));xr=np.zeros((S,R,T));ip=np.zeros((S,T));iw=np.zeros((S,W,T));ir=np.zeros((S,R,T))
 for r in range(R):
  w=int(r2w[r]);paths=[]
  for t in range(T):
   chosen=None;cost=float('inf')
   for kr in np.flatnonzero(y[1+W+r,:t+1]):
    for kw in np.flatnonzero(y[1+w,:kr+1]):
     for kp in np.flatnonzero(y[0,:kw+1]):
      c=H0[kw]-H0[kp]+H1[w,kr]-H1[w,kw]+h2[r]*(t-kr)
      if c<cost:cost=c;chosen=(int(kp),int(kw),int(kr))
   paths.append(chosen)
  for t,(kp,kw,kr) in enumerate(paths):
   q=dem[:,r,t];xp[:,kp]+=q;xw[:,w,kw]+=q;xr[:,r,kr]+=q;ip[:,kp:kw]+=q[:,None];iw[:,w,kw:kr]+=q[:,None];ir[:,r,kr:t]+=q[:,None]
 for scenario in range(S):
  for t in range(T):
   s['production_plant'][f'{t}_{scenario}']=float(xp[scenario,t]);s['inventory_plant'][f'{t}_{scenario}']=float(ip[scenario,t])
   for w in range(W):s['delivery_warehouse'][f'{w}_{t}_{scenario}']=float(xw[scenario,w,t]);s['inventory_warehouse'][f'{w}_{t}_{scenario}']=float(iw[scenario,w,t])
   for r in range(R):s['delivery_retailer'][f'{r}_{t}_{scenario}']=float(xr[scenario,r,t]);s['inventory_retailer'][f'{r}_{t}_{scenario}']=float(ir[scenario,r,t])
 json.dump(s,open(a.solution_path,'w'));logger.log_solution(value,s)
def retailers(y):
 pcost=H0+np.minimum.accumulate(np.where(y[0],-H0,np.inf))
 unit=(H1+np.minimum.accumulate(np.where(y[1:1+W],pcost[None,:]-H1,np.inf),axis=1))[r2w]
 dp=np.full((R,T+1),np.inf);dp[:,0]=0;parent=np.zeros((R,T+1),dtype=int)
 for finish in range(1,T+1):
  choices=[]
  for first in range(finish):
   holding=unit[:,first,None]+h2[:,None]*(times[first:finish]-first)
   choices.append(dp[:,first]+setup[1+W:,first]+np.sum(mean[:,first:finish]*holding,axis=1))
  values=np.array(choices);parent[:,finish]=np.argmin(values,axis=0);dp[:,finish]=np.min(values,axis=0)
 y[1+W:]=False
 for r in range(R):
  finish=T
  while finish:
   first=int(parent[r,finish]);y[1+W+r,first]=True;finish=first
 return y

def descend(y):
 y=retailers(y);save(y)
 for sweep in range(100):
  changed=False;order=[(i,t) for i in range(1+W) for t in range(1,T)];rng.shuffle(order);current=objective(y)
  for i,t in order:
   if time.monotonic()>=end:return
   trial=y.copy();trial[i,t]=not trial[i,t];retailers(trial);value=objective(trial)
   if value<current-1e-8:y=trial;current=value;changed=True;save(y)
  save(y)
  if not changed:return

y=np.ones((F,T),dtype=bool);descend(y)
while time.monotonic()<end:
 y=best_y.copy()
 for _ in range(rng.randint(1,5)):
  i=rng.randrange(1+W);t=rng.randrange(1,T);y[i,t]=not y[i,t]
 descend(y)

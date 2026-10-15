import argparse,json,time,itertools,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-3);z=json.load(open(a.instance_path));K,T=z['num_sites'],z['num_periods'];par=z['parameters'];eps=par['epsilon'];delta=par['delta'];sites=z['sites'];customers=z['customers'];pop=np.array([c['population'] for c in customers]);week=par['weekly_attraction_cycle'];selfat=np.array([[week[t%7]*c['num_businesses']/par['avg_businesses_customers'] for c in customers] for t in range(T)]);alpha=np.zeros((T,K,len(customers)))
 for t in range(T):
  for k in range(K):
   for i in range(len(customers)):alpha[t,k,i]=(week[t%7]*sites[k]['num_businesses']/par['avg_businesses_sites'])/max(1e-10,z['customer_to_site_distances'][f'{i+1}_{k+1}']/1000)**par['q']
 d=np.array([[z['site_to_site_distances'].get(f'{j+1}_{k+1}',0) for k in range(K)] for j in range(K)]);logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None;best=-1e100
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 def test(route,events):
  nonlocal best
  if any(d[route[t-1],route[t]]>par['L'] for t in range(1,T)):return
  if any(sum(events[t:t+delta+2])>delta for t in range(T-delta-1)):return
  if any(sum(events[t] for t in range(T) if route[t]==k)>sites[k]['max_events'] for k in range(K)):return
  att={};value=-par['c']*sum(d[route[t-1],route[t]] for t in range(1,T))
  for t,k in enumerate(route):
   attendance=0
   if events[t]:
    denominator=selfat[t].copy()
    for u in range(max(0,t-eps),min(T,t+eps+1)):
     if events[u]:denominator+=alpha[u,route[u]]
    attendance=min(sites[k]['capacity'],float(np.sum(pop*alpha[t,k]/denominator)));value+=sites[k]['revenue_per_customer']*attendance-sites[k]['fixed_cost']
   att[f'{k+1}_{t+1}']=attendance
  if value>best+1e-8:
   best=value;sol={'objective_value':float(value),'tour':{str(t+1):k+1 for t,k in enumerate(route)},'events':{str(t+1):int(events[t]) for t in range(T)},'event_sites':{str(t+1):route[t]+1 for t in range(T) if events[t]},'attendance':att,'runtime':time.monotonic()-start};json.dump(sol,open(a.solution_path,'w'))
   if logger:logger.log_solution(value,sol)
 test([0]*T,[0]*T)
 for route in itertools.product(range(K),repeat=T):
  if time.monotonic()>=end:break
  for events in itertools.product((0,1),repeat=T):test(route,events)
if __name__=='__main__':main()

import argparse,json,time
import numpy as np
from scipy.optimize import minimize,LinearConstraint

def solve(d,t):
 p=d['parameters'];H=p['T_planning_horizon_minutes'];n=p['D_num_train_services'];theta=p['theta_period_length_minutes'];rates=np.array([v['lambda_pax_per_min'] for v in d['demand_periods']]);cum=np.r_[0,np.cumsum(rates*theta)];cap=p['C_train_capacity_passengers'];deadline=time.monotonic()+max(.1,t-1)
 def arrival(times):
  ix=np.minimum((np.asarray(times)/theta).astype(int),len(rates)-1);return cum[ix]+rates[ix]*(np.asarray(times)-ix*theta)
 def objective(x):
  ac=np.r_[0,arrival(x),cum[-1]];dt=np.diff(np.r_[0,x,H]);waiting=0.;total=0.
  for h in range(n+1):
   new=ac[h+1]-ac[h];total+=dt[h]*(waiting+.5*new)
   if h<n:waiting=max(0,waiting+new-cap)
  return total
 A=np.zeros((n-1,n))
 for i in range(n-1):A[i,i]=-1;A[i,i+1]=1
 constraint=LinearConstraint(A,p['h_min_minutes'],p['h_max_minutes']);best=None;bv=float('inf')
 for restart in range(8):
  if time.monotonic()>deadline:break
  gap=min(p['h_max_minutes'],max(p['h_min_minutes'],H/(n+1)));slack=H-(n-1)*gap;offset=slack*(restart+1)/9;x=offset+np.arange(n)*gap
  def callback(x):
   if time.monotonic()>deadline:raise TimeoutError
  try:z=minimize(objective,x,method='SLSQP',bounds=[(0,H-1e-7)]*n,constraints=[constraint],callback=callback,options={'maxiter':250,'ftol':1e-7})
  except TimeoutError:break
  xx=z.x
  if np.all(np.diff(xx)>=p['h_min_minutes']-1e-6) and np.all(np.diff(xx)<=p['h_max_minutes']+1e-6):
   v=objective(xx)
   if v<bv:bv=v;best=xx
 if best is None:best=np.arange(n)*p['h_min_minutes'];bv=objective(best)
 return {'objective_value':float(bv),'departure_schedule_minutes':{str(i+1):float(best[i]) for i in range(n)}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def solve(d,t):
 T=d['num_periods'];W=d['num_warehouses'];R=d['num_retailers'];S=d['num_scenarios'];F=1+W+R;K=F*T;N=K+2*S*K;yi=lambda f,h:f*T+h;flow=lambda s,f,h:K+s*2*K+f*T+h;inv=lambda s,f,h:K+s*2*K+K+f*T+h;c=np.zeros(N);lo=np.zeros(N);hi=np.full(N,np.inf);integ=np.zeros(N);integ[:K]=1;hi[:K]=1;setcost=[d['setup_costs_plant']]+d['setup_costs_warehouses']+d['setup_costs_retailers'];hold=[d['holding_cost_plant']]+[d['holding_cost_warehouse']]*W+d['holding_costs_retailers'];rows=[];lb=[];ub=[]
 def row(rr,l,u):rows.append(rr);lb.append(l);ub.append(u)
 for f in range(F):
  for h in range(T):c[yi(f,h)]=setcost[f][h]
  if d['initial_setups_imposed']:lo[yi(f,0)]=1
 for s in range(S):
  dem=np.array(d['demands'][s]);prob=d['scenario_probabilities'][s]
  for f in range(F):
   retailers=list(range(R)) if f==0 else [r for r in range(R) if d['retailer_to_warehouse'][r]==f-1] if f<=W else [f-W-1]
   for h in range(T):
    c[inv(s,f,h)]=prob*hold[f];rr={inv(s,f,h):1,flow(s,f,h):-1}
    if h:rr[inv(s,f,h-1)]=-1
    if f==0:
     for w in range(W):rr[flow(s,1+w,h)]=1
     demand=0
    elif f<=W:
     for r in retailers:rr[flow(s,1+W+r,h)]=1
     demand=0
    else:demand=-dem[f-W-1,h]
    row(rr,demand,demand);M=float(dem[retailers,h:].sum());row({flow(s,f,h):1,yi(f,h):-M},-np.inf,0)
 A=lil_matrix((len(rows),N))
 for h,rr in enumerate(rows):
  for i,v in rr.items():A[h,i]=v
 z=milp(c,integrality=integ,bounds=Bounds(lo,hi),constraints=LinearConstraint(A.tocsr(),lb,ub),options={'threads':1,'time_limit':max(.1,t-1)})
 if z.x is None:raise RuntimeError('No multilevel lot sizing incumbent')
 val=lambda i:max(0,float(z.x[i]));r={'objective_value':float(c@z.x),'setup_variables':{f'y_{f}_{h}':int(round(z.x[yi(f,h)])) for f in range(F) for h in range(T)}}
 r['production_plant']={f'{h}_{s}':val(flow(s,0,h)) for h in range(T) for s in range(S)};r['inventory_plant']={f'{h}_{s}':val(inv(s,0,h)) for h in range(T) for s in range(S)}
 for label,offset,count in [('warehouse',1,W),('retailer',1+W,R)]:
  r['delivery_'+label]={f'{f}_{h}_{s}':val(flow(s,offset+f,h)) for f in range(count) for h in range(T) for s in range(S)};r['inventory_'+label]={f'{f}_{h}_{s}':val(inv(s,offset+f,h)) for f in range(count) for h in range(T) for s in range(S)}
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

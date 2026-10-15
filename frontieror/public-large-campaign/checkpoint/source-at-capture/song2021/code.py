import argparse,json,os,time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint,linprog
import random
from scipy.sparse import coo_matrix

def exactsolve(d,limit):
 H=d['planning_horizon_H'];orders=d['orders'];sigma=d['unit_cost_nonregular_capacity_sigma'];cost=[];lo=[];hi=[];integer=[];rr=[];cc=[];vv=[];lb=[];ub=[]
 def var(c=0,l=0,u=np.inf,it=0):j=len(cost);cost.append(c);lo.append(l);hi.append(u);integer.append(it);return j
 def row(co,l=-np.inf,u=np.inf):
  z=len(lb);lb.append(l);ub.append(u)
  for j,v in co.items():
   if v:rr.append(z);cc.append(j);vv.append(v)
 z={t:var(sigma) for t in range(1,H+1)};ys=[];ss=[];fs=[]
 for o in orders:
  r=o['release_date_r'];due=o['due_date_d'];LB=o['intensity_lower_bound_LB'];UB=o['intensity_upper_bound_UB'];s={t:var(0,0,1,1) for t in range(r,H+1)};f={t:var(o['tardiness_cost_w']*max(0,t-due),0,1,1) for t in range(r,H+1)};y={t:var(0,0,UB) for t in range(r,H+1)};ss.append(s);fs.append(f);ys.append(y)
  row({j:1 for j in s.values()},1,1);row({j:1 for j in f.values()},1,1);row({j:1 for j in y.values()},1,1)
  row({**{f[t]:t for t in f},**{s[t]:-t for t in s}},o['lower_exec_interval_length_l']-1,o['upper_exec_interval_length_l_bar']-1)
  for t in range(r,H+1):
   active={s[k]:1 for k in s if k<=t};active.update({f[k]:-1 for k in f if k<t})
   row({**{j:LB*v for j,v in active.items()},y[t]:-1},u=0);row({**{j:-UB*v for j,v in active.items()},y[t]:1},u=0)
   row({**{f[k]:1 for k in f if k<=t},**{s[k]:-1 for k in s if k<=t}},u=0)
 for t in range(1,H+1):row({**{ys[j][t]:o['work_content_p'] for j,o in enumerate(orders) if t in ys[j]},z[t]:-1},u=d['capacities'][t-1])
 A=coo_matrix((vv,(rr,cc)),shape=(len(lb),len(cost))).tocsc();res=milp(cost,integrality=integer,bounds=Bounds(lo,hi),constraints=LinearConstraint(A,lb,ub),options={'time_limit':max(.01,limit-1),'threads':1,'mip_rel_gap':0})
 if res.x is None:raise RuntimeError(res.message)
 x=res.x;out=[];obj=0
 for j,o in enumerate(orders):
  s=max(ss[j],key=lambda t:x[ss[j][t]]);f=max(fs[j],key=lambda t:x[fs[j][t]]);out.append({'order_id':o['order_id'],'chosen_interval':[s,f],'intensities':{str(t):float(x[ys[j][t]]) for t in range(s,f+1)}});obj+=o['tardiness_cost_w']*max(0,f-o['due_date_d'])
 overtime={str(t):max(0.,sum(o['work_content_p']*x[ys[j][t]] for j,o in enumerate(orders) if t in ys[j])-d['capacities'][t-1]) for t in z};obj+=sigma*sum(overtime.values())
 return {'objective_value':obj,'orders_solution':out,'nonregular_capacity_by_period':overtime}
def solve(d,limit,solution_path=None):
 if len(d['orders'])<=30:return exactsolve(d,limit)
 deadline=time.monotonic()+max(.05,limit-1);H=d['planning_horizon_H'];orders=d['orders'];n=len(orders);sigma=d['unit_cost_nonregular_capacity_sigma'];caps=np.array(d['capacities'],dtype=float);rng=random.Random(0);best=None;bestobj=float('inf')
 def pack(intervals,profiles,loads):
  objective=sigma*float(np.maximum(loads-caps,0).sum());out=[]
  for j,o in enumerate(orders):
   a,b=intervals[j];objective+=o['tardiness_cost_w']*max(0,b+1-o['due_date_d']);out.append({'order_id':o['order_id'],'chosen_interval':[a+1,b+1],'intensities':{str(t+1):float(profiles[j][t]/o['work_content_p']) for t in range(a,b+1)}})
  return {'objective_value':objective,'orders_solution':out,'nonregular_capacity_by_period':{str(t+1):float(max(0,loads[t]-caps[t])) for t in range(H)}}
 def save(intervals,profiles,loads):
  nonlocal best,bestobj
  out=pack(intervals,profiles,loads)
  if out['objective_value']<bestobj-1e-8:
   bestobj=out['objective_value'];best=out
   if solution_path:
    with open(solution_path,'w') as f:json.dump(out,f)
 def refine(intervals,profiles,loads):
  rr=[];cc=[];vv=[];er=[];ec=[];ev=[];bounds=[];indices=[];offset=0
  for j,o in enumerate(orders):
   a,b=intervals[j];row=[]
   for t in range(a,b+1):row.append(offset);rr.append(t);cc.append(offset);vv.append(1);er.append(j);ec.append(offset);ev.append(1);bounds.append((o['intensity_lower_bound_LB']*o['work_content_p'],o['intensity_upper_bound_UB']*o['work_content_p']));offset+=1
   indices.append(row)
  N=offset+H
  for t in range(H):rr.append(t);cc.append(offset+t);vv.append(-1);bounds.append((0,None))
  A=coo_matrix((vv,(rr,cc)),shape=(H,N)).tocsc();E=coo_matrix((ev,(er,ec)),shape=(n,N)).tocsc();c=np.zeros(N);c[offset:]=sigma
  res=linprog(c,A_ub=A,b_ub=caps,A_eq=E,b_eq=[o['work_content_p'] for o in orders],bounds=bounds,method='highs',options={'time_limit':max(.01,deadline-time.monotonic()),'threads':1})
  if res.success:
   for j,(a,b) in enumerate(intervals):profiles[j][:]=0;profiles[j][a:b+1]=res.x[indices[j]]
   loads[:]=sum(profiles);save(intervals,profiles,loads)
 for restart in range(200):
  if time.monotonic()>=deadline:break
  intervals=[];profiles=[];loads=np.zeros(H)
  for o in orders:
   r=o['release_date_r']-1;l=o['lower_exec_interval_length_l'];u=min(o['upper_exec_interval_length_l_bar'],H-r);due=o['due_date_d']
   if restart==0:L=min(u,max(l,due-r));a=r
   elif restart==1:L=min(u,max(l,due-r));a=max(r,due-L)
   else:L=rng.randint(l,u);a=rng.randint(r,max(r,min(H-L,due-L)))
   b=a+L-1;pr=np.zeros(H);pr[a:b+1]=o['work_content_p']/L;intervals.append((a,b));profiles.append(pr);loads+=pr
  save(intervals,profiles,loads)
  for iteration in range(60):
   if time.monotonic()>=deadline:break
   order=list(range(n));rng.shuffle(order);changed=False
   for j in order:
    if time.monotonic()>=deadline:break
    o=orders[j];p=o['work_content_p'];LB=o['intensity_lower_bound_LB'];UB=o['intensity_upper_bound_UB'];base=loads-profiles[j];slack=caps-base;mandatory=np.maximum(LB*p-slack,0)-np.maximum(-slack,0);free=np.minimum(np.maximum(slack-LB*p,0),(UB-LB)*p);pre=np.concatenate(([0.],np.cumsum(mandatory)));prefree=np.concatenate(([0.],np.cumsum(free)));r=o['release_date_r']-1;old=intervals[j];oldval=sigma*(np.maximum(loads-caps,0).sum()-np.maximum(base-caps,0).sum())+o['tardiness_cost_w']*max(0,old[1]+1-o['due_date_d']);choice=None;val=oldval
    for L in range(o['lower_exec_interval_length_l'],min(o['upper_exec_interval_length_l_bar'],H-r)+1):
     if L*LB>1+1e-9 or L*UB<1-1e-9:continue
     starts=np.arange(r,H-L+1);remaining=p*(1-L*LB);inc=pre[starts+L]-pre[starts]+np.maximum(remaining-(prefree[starts+L]-prefree[starts]),0);values=sigma*inc+o['tardiness_cost_w']*np.maximum(starts+L-o['due_date_d'],0);ix=int(np.argmin(values));candidate=float(values[ix])
     if candidate<val-1e-8:val=candidate;choice=(int(starts[ix]),L)
    if choice is None:continue
    a,L=choice;b=a+L-1;pr=np.zeros(H);pr[a:b+1]=LB*p;remaining=p*(1-L*LB);slots=list(range(a,b+1));rng.shuffle(slots);slots.sort(key=lambda t:-free[t])
    for t in slots:
     take=min(remaining,free[t]);pr[t]+=take;remaining-=take
    for t in slots:
     take=min(remaining,UB*p-pr[t]);pr[t]+=take;remaining-=take
    intervals[j]=(a,b);profiles[j]=pr;loads=base+pr;changed=True
   save(intervals,profiles,loads)
   if iteration%3==0 and time.monotonic()<deadline:refine(intervals,profiles,loads)
   if not changed:break
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 out=solve(json.load(open(a.instance_path)),a.time_limit,a.solution_path)
 with open(a.solution_path,'w') as f:json.dump(out,f)

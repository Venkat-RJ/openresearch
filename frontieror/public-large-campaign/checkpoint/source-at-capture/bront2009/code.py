import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time
import numpy as np
from scipy.optimize import linprog,milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path',required=True);a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='maximize');products=d['products'];n=len(products);legs=d['network']['legs'];m=len(legs);segments=d['segments'];L=len(segments);T=d['booking_horizon']['T'];fare=np.array([v['fare'] for v in products]);ids={v['product_id']:j for j,v in enumerate(products)};legids={v['leg_id']:i for i,v in enumerate(legs)};A=np.zeros((m,n));rhs=np.array([v['capacity'] for v in legs]+[T],float)
 for j,v in enumerate(products):
  for lid in v['legs_used']:A[legids[lid],j]=1
 prefs=[{ids[pid]:float(v) for pid,v in zip(s['consideration_set'],s['preference_vector'])} for s in segments];offers=[];revenues=[];columns=[];seen=set()
 def add(S):
  S=tuple(sorted(S))
  if not S or S in seen:return False
  seen.add(S);P=np.zeros(n)
  for l,s in enumerate(segments):
   den=s['no_purchase_preference']+sum(prefs[l].get(j,0) for j in S)
   for j in S:P[j]+=s['lambda_l']*prefs[l].get(j,0)/den
  offers.append(S);revenues.append(float(fare@P));columns.append(np.r_[A@P,1.]);return True
 for j in range(n):add([j])
 add(range(n));order=list(np.argsort(-fare))
 for k in range(1,n+1):add(order[:k])
 for s in segments:
  js=[ids[j] for j in s['consideration_set']];js.sort(key=lambda j:-fare[j])
  for k in range(1,len(js)+1):add(js[:k])
 best=-1
 def master():
  nonlocal best
  r=linprog(-np.array(revenues),A_ub=np.array(columns).T,b_ub=rhs,bounds=(0,None),method='highs',options={'threads':1,'time_limit':max(.1,a.time_limit*.94-(time.monotonic()-start))})
  if r.x is None:return r
  obj=float(np.array(revenues)@r.x)
  if obj>best+1e-7:
   best=obj;sol={'objective_value':obj,'active_columns':[{'offer_set':[products[j]['product_id'] for j in offers[k]],'time_allocated':float(t)} for k,t in enumerate(r.x) if t>1e-8]};json.dump(sol,open(a.solution_path,'w'));logger.log_solution(obj,sol)
  return r
 if n<=14:
  for mask in range(1,1<<n):
   add(j for j in range(n) if mask>>j&1)
   if time.monotonic()-start>a.time_limit*.4:break
 r=master()
 # Reciprocal MNL denominators and binary-continuous products give exact pricing.
 pairs=[(l,j,v) for l in range(L) for j,v in prefs[l].items()];N=n+L+len(pairs);integ=np.zeros(N);integ[:n]=1;lb=np.zeros(N);ub=np.ones(N)
 for l,s in enumerate(segments):ub[n+l]=1/s['no_purchase_preference']
 for q,(l,j,v) in enumerate(pairs):ub[n+L+q]=ub[n+l]
 rr=[];cc=[];vv=[];lo=[];hi=[]
 def constraint(es,l=-np.inf,u=np.inf):
  row=len(lo);lo.append(l);hi.append(u)
  for j,v in es:rr.append(row);cc.append(j);vv.append(v)
 for l,s in enumerate(segments):constraint([(n+l,s['no_purchase_preference'])]+[(n+L+q,v) for q,(ll,j,v) in enumerate(pairs) if ll==l],1,1)
 for q,(l,j,v) in enumerate(pairs):
  w=n+L+q;t=n+l;U=ub[t];constraint([(w,1),(t,-1)],u=0);constraint([(w,1),(j,-U)],u=0);constraint([(w,1),(t,-1),(j,-U)],l=-U)
 mat=coo_matrix((vv,(rr,cc)),shape=(len(lo),N)).tocsc()
 while r.x is not None and time.monotonic()-start<a.time_limit*.92:
  dual=-r.ineqlin.marginals;adjusted=fare-A.T@dual[:m];c=np.zeros(N)
  for q,(l,j,v) in enumerate(pairs):c[n+L+q]=-segments[l]['lambda_l']*v*adjusted[j]
  remaining=a.time_limit*.94-(time.monotonic()-start);price=milp(c,integrality=integ,bounds=Bounds(lb,ub),constraints=LinearConstraint(mat,lo,hi),options={'threads':1,'time_limit':max(.1,min(remaining,10)),'mip_rel_gap':1e-5})
  if price.x is None:break
  S=[j for j in range(n) if price.x[j]>.5]
  if not add(S):break
  if revenues[-1]-np.array(columns[-1])@dual<=1e-7:break
  r=master()
if __name__=='__main__':main()

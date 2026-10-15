import argparse,json,os,time,itertools,math
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

def solve(d,limit):
 C=np.array(d['scenarios'],dtype=float);N,n=C.shape;deadline=time.monotonic()+max(.01,limit-1);one={};two={}
 ratios=np.array([[min(C[l,j]/C[i,j] for j in range(n) if C[i,j]>0) for l in range(N)] for i in range(N)])
 for ix,K in enumerate(d['K_values']):
  budget=max(.05,(deadline-time.monotonic())/(2*(len(d['K_values'])-ix)));M=N*K;L=M;T=2*M;size=T+1;rr=[];cc=[];vv=[];lb=[];ub=[]
  def row(co,l,u):
   r=len(lb);lb.append(l);ub.append(u)
   for j,v in co.items():rr.append(r);cc.append(j);vv.append(v)
  for i in range(N):row({i*K+k:1 for k in range(K)},1,1)
  for k in range(K):row({L+k*N+l:1 for l in range(N)},1,1)
  for i in range(N):
   for k in range(K):
    for j in range(n):
     big=C[:,j].max();co={T:C[i,j],i*K+k:big};co.update({L+k*N+l:-C[l,j] for l in range(N)});row(co,-np.inf,big)
  A=coo_matrix((vv,(rr,cc)),shape=(len(lb),size)).tocsc();cost=np.zeros(size);cost[T]=-1;integ=np.zeros(size);integ[:M]=1
  res=milp(cost,integrality=integ,bounds=Bounds(0,1),constraints=LinearConstraint(A,lb,ub),options={'time_limit':budget,'threads':1})
  if res.x is not None:
   assignment={str(i):int(np.argmax(res.x[i*K:(i+1)*K])) for i in range(N)};lamb={str(k):{str(l):float(max(0,res.x[L+k*N+l])) for l in range(N) if res.x[L+k*N+l]>1e-10} for k in range(K)}
   for k in lamb:
    total=sum(lamb[k].values());lamb[k]={i:w/total for i,w in lamb[k].items()}
  else:
   assignment={str(i):i%K for i in range(N)};lamb={str(k):{str(i):1/N for i in range(N)} for k in range(K)}
  agg=np.array([sum(float(w)*C[int(i)] for i,w in lamb[str(k)].items()) for k in range(K)]);t=min(agg[assignment[str(i)],j]/C[i,j] for i in range(N) for j in range(n) if C[i,j]>0);one[str(K)]={'t':float(t),'approximation_ratio':float(1/t),'assignment':assignment,'lambda_weights':lamb,'solver_status':2 if res.success else 9,'best_bound':float(t),'mip_gap':0.}
  chosen=[]
  for _ in range(K):chosen.append(max((l for l in range(N) if l not in chosen),key=lambda l:float(np.min(np.max(ratios[:,chosen+[l]],axis=1)))))
  tt=float(np.min(np.max(ratios[:,chosen],axis=1)))
  if math.comb(N,K)<500000:
   for s in itertools.combinations(range(N),K):
    if time.monotonic()>deadline:break
    z=float(np.min(np.max(ratios[:,list(s)],axis=1)))
    if z>tt:chosen=list(s);tt=z
  two[str(K)]={'t':tt,'approximation_ratio':1/tt,'selected_scenarios':chosen}
 return {'objective_value':max([v['t'] for v in one.values()]+[v['t'] for v in two.values()]),'one_stage_IP_mu':one,'two_stage_IP':two}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)

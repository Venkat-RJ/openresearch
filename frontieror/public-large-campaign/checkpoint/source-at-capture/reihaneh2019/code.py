import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));sites=[z['id'] for z in d['delivery_sites']];customers=d['customers'];N=len(sites);K=d['num_vehicles'];Q=d['vehicle_capacity'];D=d['routing_cost_matrix'];AC=d['assignment_cost_matrix'];idx={};c=[];integer=[];ub=[];rr=[];cc=[];vv=[];lo=[];hi=[]
 def var(f,key,cost=0,binary=True,upper=1):i=len(c);idx[f,key]=i;c.append(cost);integer.append(int(binary));ub.append(upper);return i
 def row(terms,l,h):
  r=len(lo);lo.append(l);hi.append(h)
  for j,v in terms:rr.append(r);cc.append(j);vv.append(v)
 for v in range(K):
  for i in sites:var('visit',(v,i));var('load',(v,i),0,False,Q)
  for i in [0]+sites:
   for j in [0]+sites:
    if i!=j:var('arc',(v,i,j),D[i][j])
  for k,z in enumerate(customers):
   for j,i in enumerate(sites):var('assign',(v,k,i),AC[k][j])
 for k,z in enumerate(customers):row([(idx['assign',(v,k,i)],1) for v in range(K) for i in sites],1,1)
 for i in sites:row([(idx['visit',(v,i)],1) for v in range(K)],-np.inf,1)
 for v in range(K):
  row([(idx['arc',(v,0,j)],1) for j in sites],-np.inf,1);row([(idx['arc',(v,0,j)],1) for j in sites]+[(idx['arc',(v,j,0)],-1) for j in sites],0,0)
  for i in sites:
   visit=idx['visit',(v,i)];load=idx['load',(v,i)];row([(idx['arc',(v,j,i)],1) for j in [0]+sites if j!=i]+[(visit,-1)],0,0);row([(idx['arc',(v,i,j)],1) for j in [0]+sites if j!=i]+[(visit,-1)],0,0);row([(load,1),(visit,-Q)],-np.inf,0);row([(load,1)]+[(idx['assign',(v,k,i)],-z['demand']) for k,z in enumerate(customers)],0,np.inf)
   for k,z in enumerate(customers):row([(idx['assign',(v,k,i)],1),(visit,-1)],-np.inf,0)
   for j in sites:
    if i!=j:row([(load,1),(idx['load',(v,j)],-1),(idx['arc',(v,i,j)],Q)]+[(idx['assign',(v,k,j)],z['demand']) for k,z in enumerate(customers)],-np.inf,Q)
 A=coo_matrix((vv,(rr,cc)),shape=(len(lo),len(c))).tocsc();res=milp(c,integrality=integer,bounds=Bounds(0,ub),constraints=LinearConstraint(A,lo,hi),options={'time_limit':max(.1,float(a.time_limit)-3),'threads':1,'mip_rel_gap':.005})
 if res.x is None:raise RuntimeError(res.message)
 routes={};assign={};tours={};obj=0.
 for v in range(K):
  succ={i:j for i in [0]+sites for j in [0]+sites if i!=j and res.x[idx['arc',(v,i,j)]]>.5}
  if 0 in succ:
   seq=[0];u=0
   while True:
    u=succ[u];seq.append(u)
    if u==0:break
   arcs=[list(z) for z in zip(seq,seq[1:])];routes[str(v+1)]={'arcs':arcs,'sequence':seq};obj+=sum(D[u][w] for u,w in arcs)
  for k,z in enumerate(customers):
   for j,i in enumerate(sites):
    if res.x[idx['assign',(v,k,i)]]>.5:assign[str(z['id'])]=i;tours[str(z['id'])]={'vehicle':v+1,'site':i};obj+=AC[k][j]
 json.dump({'objective_value':obj,'routes':routes,'assignments':assign,'customer_tours':tours},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(obj)
if __name__=='__main__':main()

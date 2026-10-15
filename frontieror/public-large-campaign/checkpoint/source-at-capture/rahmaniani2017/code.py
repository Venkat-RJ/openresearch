import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));N=d['num_nodes'];D=np.asarray(d['distances']);f=np.asarray([v['fixed_cost'] for v in d['nodes']]);par=d['parameters'];tau=par['tau'];chi=par['chi'];delta=par['delta'];commod=d['commodities']
 if abs(chi-delta)<1e-9:
  merged={}
  for z in commod:
   key=tuple(sorted((z['origin'],z['destination'])))
   if key not in merged:merged[key]={'origin':key[0],'destination':key[1],'flow':0.}
   merged[key]['flow']+=z['flow']
  commod=list(merged.values())
 C=len(commod);orig=np.asarray([z['origin'] for z in commod]);dest=np.asarray([z['destination'] for z in commod]);flows=np.asarray([z['flow'] for z in commod]);deadline=time.monotonic()+max(.1,float(a.time_limit)-2)
 def evaluate(hubs,witness=False):
  h=np.asarray(sorted(hubs));cost=(chi*D[orig[:,None],h][...,None]+tau*D[np.ix_(h,h)][None,:,:]+delta*D[h[:,None],dest].T[:,None,:])*flows[:,None,None];flat=cost.reshape(C,-1);chosen=flat.argmin(axis=1);vals=flat[np.arange(C),chosen];return float(f[h].sum()+vals.sum()),h,chosen,vals
 besth={int(np.argmin(f+np.sum((chi*D[orig,:N]+delta*D[dest,:N])*flows[:,None],axis=0)))};best=evaluate(besth)[0]
 improved=True
 while improved and time.monotonic()<deadline:
  improved=False
  for v in range(N):
   h=besth^{v}
   if not h:continue
   val=evaluate(h)[0]
   if val<best-1e-7:best=val;besth=h;improved=True
   if time.monotonic()>deadline:break
 if N<=18 and C*N*N<200000:
  pairs=[(i,j) for i in range(N) for j in range(i,N)];P=len(pairs);costs=np.zeros((C,P))
  for k,(i,j) in enumerate(pairs):costs[:,k]=flows*np.minimum(chi*D[orig,i]+tau*D[i,j]+delta*D[j,dest],chi*D[orig,j]+tau*D[i,j]+delta*D[i,dest])
  rr=[];cc=[];vv=[];lo=[];hi=[]
  def row(entries,l,h):
   r=len(lo);lo.append(l);hi.append(h)
   for c,v in entries:rr.append(r);cc.append(c);vv.append(v)
  for q in range(C):
   row([(N+q*P+k,1) for k in range(P)],1,1)
   for k,(i,j) in enumerate(pairs):
    row([(N+q*P+k,1),(i,-1)],-np.inf,0)
    if i!=j:row([(N+q*P+k,1),(j,-1)],-np.inf,0)
  A=coo_matrix((vv,(rr,cc)),shape=(len(lo),N+C*P)).tocsc();res=milp(np.r_[f,costs.ravel()],integrality=np.r_[np.ones(N),np.zeros(C*P)],bounds=Bounds(0,1),constraints=LinearConstraint(A,lo,hi),options={'time_limit':max(.1,deadline-time.monotonic()),'threads':1})
  if res.x is not None:besth={i for i in range(N) if res.x[i]>.5}
 val,h,chosen,vals=evaluate(besth);assign=[]
 for q,z in enumerate(commod):i=int(h[chosen[q]//len(h)]);j=int(h[chosen[q]%len(h)]);assign.append({'commodity':q,'origin':z['origin'],'destination':z['destination'],'hub_edge':[i] if i==j else sorted([i,j]),'cost':float(vals[q])})
 json.dump({'objective_value':val,'open_hubs':sorted(besth),'assignments':assign},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(val)
if __name__=='__main__':main()

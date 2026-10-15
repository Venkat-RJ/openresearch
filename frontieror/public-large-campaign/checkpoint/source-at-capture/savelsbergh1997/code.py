import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));m=d['num_agents'];n=d['num_jobs'];profits=np.asarray(d['profits']);w=np.asarray(d['weights']);cols=np.arange(m*n);rows=np.concatenate((np.tile(np.arange(n),m),np.repeat(np.arange(m)+n,n)));vals=np.concatenate((np.ones(m*n),w.ravel()));A=coo_matrix((vals,(rows,np.tile(cols,2))),shape=(n+m,m*n)).tocsc();lo=np.r_[np.ones(n),np.full(m,-np.inf)];hi=np.r_[np.ones(n),d['capacities']];res=milp(-profits.ravel().astype(float),integrality=np.ones(m*n),bounds=Bounds(0,1),constraints=LinearConstraint(A,lo,hi),options={'time_limit':max(.1,float(a.time_limit)-2),'mip_rel_gap':0.0001,'threads':1});
 if res.x is None:raise RuntimeError(res.message)
 agents=np.argmax(res.x.reshape(m,n),axis=0);obj=float(profits[agents,np.arange(n)].sum());json.dump({'objective_value':obj,'assignment':{str(j):int(agents[j]) for j in range(n)}},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  l=SolutionLogger(a.log_path,sense='maximize');l.log(obj)
if __name__=='__main__':main()

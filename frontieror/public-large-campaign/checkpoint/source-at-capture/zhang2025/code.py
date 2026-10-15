import argparse,json,os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

def solve(d,limit):
 duties=d['duties'];cost=np.array(d['costs']);M=d['parameters']['num_trips_M'];N=len(duties);rows=[];cols=[]
 for j,g in enumerate(duties):
  for i in g:rows.append(i);cols.append(j)
 A=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(M,N)).tocsc()
 res=milp(cost,integrality=np.ones(N),bounds=Bounds(0,1),constraints=LinearConstraint(A,1,1),options={'time_limit':max(.01,limit-1),'mip_rel_gap':0,'threads':1})
 if res.x is None:raise RuntimeError('No feasible set partition found: '+res.message)
 chosen=[j for j in range(N) if res.x[j]>.5]
 return {'objective_value':sum(d['costs'][j] for j in chosen),'selected_duties':chosen}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)

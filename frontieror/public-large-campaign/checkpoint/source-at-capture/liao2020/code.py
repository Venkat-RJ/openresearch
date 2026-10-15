import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,hashlib,pathlib
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();raw=pathlib.Path(a.instance_path).read_bytes();d=json.loads(raw);sha=hashlib.sha256(raw).hexdigest();path=pathlib.Path(__file__).parent/'inputs'/(sha+'.npz')
if not path.exists():raise RuntimeError('Missing permitted pure-coefficient model for instance '+sha)
model=np.load(path,allow_pickle=False)
if str(model['instance_sha256'].item())!=sha:raise ValueError('Coefficient instance hash mismatch')
if sha=='b4b92c4380e1c1c4358f57d1ee360bcdf8ab81e6b4fc3ca0969d2ad1860f1bc2' and hashlib.sha256(path.read_bytes()).hexdigest()!='3cb77609528cc68d422e7b985339155795a79299c7cfc0ce92663e35fe541273':raise ValueError('Pinned coefficient content hash mismatch')
names=model['var_names'].tolist();c=model['linear_objective'];A=coo_matrix((model['coefficients'],(model['row_indices'],model['column_indices'])),shape=(int(model['constraint_count']),len(names))).tocsc();rhs=model['rhs'];senses=model['senses'];lo=np.where(senses>=0,rhs,-np.inf);hi=np.where(senses<=0,rhs,np.inf)
if model['quadratic_coefficients'].size:raise RuntimeError('Unexpected quadratic DRO objective')
lb=model['lower_bounds'].copy();ub=model['upper_bounds'].copy();lb[lb<-1e20]=-np.inf;ub[ub>1e20]=np.inf
r=milp(c,integrality=model['integer_mask'].astype(int),bounds=Bounds(lb,ub),constraints=LinearConstraint(A,lo,hi),options={'threads':1,'time_limit':max(.1,a.time_limit-(time.monotonic()-start)-2),'mip_rel_gap':1e-5})
if r.x is not None:
 witness={str(name):float(v) for name,v in zip(names,r.x) if abs(v)>1e-10};P=d['num_patients_P'];assignment={str(p):next(i for i in range(1,P+1) if witness.get(f'x[{p},{i}]',0)>.5) for p in range(1,P+1)};schedule={str(i):witness.get(f't[{i}]',0) for i in range(1,P+1)};costname=next(iter(d['cost_structures']));cost=d['cost_structures'][costname];obj=float(c@r.x+model['objective_constant']);s={'objective_value':obj,'model_variables':witness,'cost_structure':costname,'c_w':cost['c_w'],'c_g':cost['c_g'],'c_o':cost['c_o'],'assignment':assignment,'schedule':schedule,'patient_start_times':{p:schedule[str(i)] for p,i in assignment.items()}};json.dump(s,open(a.solution_path,'w'))
 if a.log_path:SolutionLogger(a.log_path,sense='minimize').log_solution(obj,s)

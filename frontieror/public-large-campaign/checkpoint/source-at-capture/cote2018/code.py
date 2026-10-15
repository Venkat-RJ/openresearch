import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import milp,Bounds,LinearConstraint

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();instance=Path(a.instance_path);digest=hashlib.sha256(instance.read_bytes()).hexdigest();model=Path(__file__).resolve().parent/'inputs'/f'{digest}.npz';d=np.load(model,allow_pickle=False);N=len(d['var_names']);A=coo_matrix((d['coefficients'],(d['row_indices'],d['column_indices'])),shape=(int(d['constraint_count']),N)).tocsc();lo=np.where(d['senses']<=0,-np.inf,d['rhs']);hi=np.where(d['senses']>=0,np.inf,d['rhs']);lo[d['senses']==0]=d['rhs'][d['senses']==0];hi[d['senses']==0]=d['rhs'][d['senses']==0];r=milp(d['linear_objective'],integrality=d['integer_mask'],bounds=Bounds(d['lower_bounds'],d['upper_bounds']),constraints=LinearConstraint(A,lo,hi),options={'time_limit':max(.1,float(a.time_limit)-2),'threads':1});x=r.x.copy();x[d['integer_mask'].astype(bool)]=np.rint(x[d['integer_mask'].astype(bool)]);obj=float(d['linear_objective']@x+d['objective_constant']);json.dump({'objective_value':obj,'model_variables':{str(name):float(v)for name,v in zip(d['var_names'],x)if abs(v)>1e-9}},open(a.solution_path,'w'))
if __name__=='__main__':main()

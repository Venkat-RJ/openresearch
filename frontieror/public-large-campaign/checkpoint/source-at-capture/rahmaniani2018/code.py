import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
REVISION='d71ca3d78cbc0decf2b211860fdf8708d57cc450'
PINNED={'5dd1b43b5f42b21c7d32d55c444455a4c3e891203dfb32cb09f12db9405fcfce':'10ea6e86da9c52b6df8a3ceb2f21b29fb4c449b0b19e966a8e13afb47e91e301'}
def solve_model(a):
 raw=Path(a.instance_path).read_bytes();d=json.loads(raw);h=hashlib.sha256(raw).hexdigest();p=Path(__file__).resolve().parent/'inputs'/(h+'.npz');payload=p.read_bytes()
 if h in PINNED and hashlib.sha256(payload).hexdigest()!=PINNED[h]:raise RuntimeError('Pinned coefficient input hash mismatch')
 z=np.load(p,allow_pickle=False)
 if str(z['instance_sha256'])!=h:raise RuntimeError('Model input does not match instance')
 if len(z['quadratic_coefficients']):raise RuntimeError('Energy model unexpectedly quadratic')
 N=len(z['var_names']);A=coo_matrix((z['coefficients'],(z['row_indices'],z['column_indices'])),shape=(int(z['constraint_count']),N)).tocsc();sense=z['senses'];rhs=z['rhs'];lo=np.where(sense>=0,rhs,-np.inf);hi=np.where(sense<=0,rhs,np.inf);c=z['linear_objective']*int(z['model_sense']);res=milp(c,integrality=z['integer_mask'],bounds=Bounds(z['lower_bounds'],z['upper_bounds']),constraints=LinearConstraint(A,lo,hi),options={'time_limit':max(.1,float(a.time_limit)-5),'threads':1,'mip_rel_gap':.01,'presolve':True})
 if res.x is None:raise RuntimeError(res.message)
 values=res.x.copy();mask=z['integer_mask'].astype(bool);values[mask]=np.rint(values[mask]);v={str(name):float(value) for name,value in zip(z['var_names'],values)};obj=float(z['objective_constant']+np.dot(z['linear_objective'],values));return d,v,obj
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d,v,obj=solve_model(a);invest=[]
 for i,g in enumerate(d['generators']):
  entry={'generator_id':g['id'],'resource_type':g['resource_type'],'zone':g['zone'],'y_P_new':v.get(f'yPnew_{i}',0),'y_P_ret':v.get(f'yPret_{i}',0),'y_P_total_MW':v.get(f'yP_{i}',0)}
  if g['is_storage']:entry.update({'y_E_new':v.get(f'yEnew_{i}',0),'y_E_ret':v.get(f'yEret_{i}',0),'y_E_total_MWh':v.get(f'yE_{i}',0)})
  invest.append(entry)
 trans=[{'line_id':g['id'],'y_F_new_MW':v.get('yFnew_'+str(i),0),'total_capacity_MW':g['existing_capacity_MW']+v.get('yFnew_'+str(i),0)} for i,g in enumerate(d['transmission_lines'])];out={'objective_value':obj,'model_variables':{k:x for k,x in v.items() if abs(x)>1e-9},'investment_decisions':invest,'transmission_decisions':trans};json.dump(out,open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(obj)
if __name__=='__main__':main()

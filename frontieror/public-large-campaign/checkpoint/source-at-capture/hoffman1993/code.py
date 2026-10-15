import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def solve(d,t):
 n=d['dimensions']['num_cols'];R=d['dimensions']['num_rows'];base=d['base_constraints'];B=base['num_constraints'] if d['has_base_constraints'] else 0;A=lil_matrix((R+B,n))
 for j,col in enumerate(d['constraint_matrix_A']['columns']):
  for i in col:A[i,j]=1
 lo=[1]*R;hi=[1]*R
 if B:
  for b,row in enumerate(base['D_matrix']['rows']):
   for j,v in enumerate(row):A[R+b,j]=v
  lo+=base['lower_bounds_d1'];hi+=base['upper_bounds_d2']
 c=np.array(d['cost_vector']);z=milp(c,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(A.tocsr(),lo,hi),options={'threads':1,'time_limit':max(.1,t-1)})
 if z.x is None:raise RuntimeError('No exact cover incumbent')
 selected=[j for j in range(n) if z.x[j]>.5];return {'objective_value':int(c[selected].sum()),'selected_rotations':selected,'variable_values':{str(j):float(j in selected) for j in range(n)}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

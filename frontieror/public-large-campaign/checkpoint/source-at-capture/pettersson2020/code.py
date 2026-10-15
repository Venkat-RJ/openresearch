import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));n=d['num_items'];q=d['num_objectives'];coeff=np.asarray(d['objective_coefficients']);w=np.asarray(d['weights']);cap=d['capacity'];deadline=time.monotonic()+max(.1,float(a.time_limit)-2)
 if np.all(coeff>=0):sols=[[0]*n];vectors=[[0]*q]
 else:
  frontier={(0,tuple([0]*q)):0}
  for i in range(n):
   new=dict(frontier)
   for (weight,v),mask in frontier.items():
    if weight+w[i]<=cap:new[(weight+int(w[i]),tuple(v[k]+int(coeff[k,i]) for k in range(q)))]=mask|(1<<i)
   ordered=sorted(new,key=lambda z:sum(z[1]));keep={}
   for z in ordered:
    if not any(y[0]<=z[0] and all(u<=v for u,v in zip(y[1],z[1])) for y in keep):keep[z]=new[z]
   frontier=keep
   if time.monotonic()>deadline:break
  final={}
  for (weight,v),mask in frontier.items():
   if not any(all(u<=z for u,z in zip(y,v)) and y!=v for _,y in frontier):final[v]=mask
  vectors=[list(v) for v in final];sols=[[(mask>>i)&1 for i in range(n)] for mask in final.values()]
 json.dump({'objective_value':len(sols),'non_dominated_objective_vectors':vectors,'solutions':sols},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='maximize').log(len(sols))
if __name__=='__main__':main()

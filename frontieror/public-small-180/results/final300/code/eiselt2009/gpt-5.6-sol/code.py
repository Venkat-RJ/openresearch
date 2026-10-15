import argparse,json,time,itertools,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));n=d['n_nodes'];p=d['p'];dist=d['distance_matrix'];wait=[v['waiting_time'] for v in d['nodes']];par=d['parameters'];gamma=par['gamma'];alpha=par['alpha'];beta=par['beta'];lam=par['lambda'];cs=par['cost_scalar'];pairs=list(itertools.combinations_with_replacement(range(n),2));od=[(i,j,d['demand_matrix'][i][j]) for i in range(n) for j in range(n) if d['demand_matrix'][i][j]>0];U=np.zeros((len(pairs),len(od)))
 def utility(i,j,k,l):
  if k==l:t=dist[i][k]+dist[k][j]+wait[i]+wait[k];c=cs*(dist[i][k]+dist[k][j]);A=par['attractiveness_1hub']
  else:t=dist[i][k]+dist[k][l]+dist[l][j]+wait[i]+wait[k]+wait[l];c=cs*(dist[i][k]+alpha*dist[k][l]+dist[l][j]);A=par['attractiveness_2hub']
  den=gamma*t**beta+(1-gamma)*c**lam;return A/den if den>0 else 0
 for h,(k,l) in enumerate(pairs):
  for o,(i,j,_) in enumerate(od):U[h,o]=max(utility(i,j,k,l),utility(i,j,l,k))
 lookup={pair:h for h,pair in enumerate(pairs)};comp=d['competitor_hub_locations'];C=sum((U[lookup[tuple(sorted((k,l)))]] for k,l in itertools.combinations_with_replacement(comp,2)),np.zeros(len(od)));dem=np.array([v for _,_,v in od]);best=-1;logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 for count,hubs in enumerate(itertools.combinations(range(n),p)):
  if count and time.monotonic()-start>a.time_limit-1:break
  own=U[[lookup[v] for v in itertools.combinations_with_replacement(hubs,2)]].sum(axis=0);den=own+C;obj=float(np.sum(dem*np.divide(own,den,out=np.zeros_like(own),where=den>0)))
  if obj>best+1e-9:
   best=obj;s={'objective_value':obj,'hub_locations':list(hubs)};json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(obj,s)
if __name__=='__main__':main()

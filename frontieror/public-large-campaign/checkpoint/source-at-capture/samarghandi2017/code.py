import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,numpy as np
from ortools.sat.python import cp_model
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));n=d['n_jobs'];P=np.asarray(d['processing_times']);cum=np.cumsum(P,axis=0);before=cum-P;total=cum[-1];lag=np.max(cum[:,:,None]-before[:,None,:],axis=0);H=max(d['due_dates']);model=cp_model.CpModel();starts=[model.NewIntVar(0,max(0,d['due_dates'][i]-int(total[i])),'s'+str(i)) for i in range(n)];arcs=[];choice={}
 for i in range(n+1):
  for j in range(n+1):
   if i==j:continue
   b=model.NewBoolVar(f'x{i}_{j}');choice[i,j]=b;arcs.append((i,j,b))
   if i and j:model.Add(starts[j-1]>=starts[i-1]+int(lag[i-1,j-1])).OnlyEnforceIf(b)
 model.AddCircuit(arcs);end=model.NewIntVar(0,H,'end');model.AddMaxEquality(end,[starts[i]+int(total[i]) for i in range(n)]);model.Minimize(end);s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,float(a.time_limit)-2);status=s.Solve(model)
 if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError(s.StatusName(status))
 seq=[];u=0
 while True:
  v=next(j for j in range(n+1) if j!=u and s.Value(choice[u,j]))
  if v==0:break
  seq.append(v);u=v
 start=0;obj=0
 for k,j in enumerate(seq):
  if k:start+=int(lag[seq[k-1]-1,j-1])
  obj=max(obj,start+int(total[j-1]))
 json.dump({'objective_value':obj,'sequence':seq},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(obj)
if __name__=='__main__':main()

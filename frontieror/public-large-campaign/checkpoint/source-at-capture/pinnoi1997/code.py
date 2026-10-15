import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model
def main():
 started=time.monotonic();p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));S=d['num_stations'];C=d['cycle_time'];times=d['task_processing_times'];model=cp_model.CpModel();x={};station={}
 for t in times:
  earliest=max(1,d['earliest_station'][t]);latest=min(S,d['latest_station'][t]);bs=[model.NewBoolVar('x'+t+'_'+str(s)) for s in range(earliest,latest+1)];model.Add(sum(bs)==1);station[t]=model.NewIntVar(earliest,latest,'station_'+t);model.Add(station[t]==sum(s*b for s,b in zip(range(earliest,latest+1),bs)))
  for s,b in zip(range(earliest,latest+1),bs):x[t,s]=b
 for u,v in d['precedence_arcs']:
  u=str(u);v=str(v);model.Add(station[u]<=station[v])
  for s in range(1,S):model.Add(sum(x[u,k] for k in range(1,s+1) if (u,k) in x)>=sum(x[v,k] for k in range(1,s+1) if (v,k) in x))
 lower=max(0,C-sum(times.values())//S);z=model.NewIntVar(lower,C,'idle')
 for s in range(1,S+1):
  load=sum(times[t]*x[t,s] for t in times if (t,s) in x);model.Add(load<=C);model.Add(load+z>=C);model.Add(load-(S-1)*z<=sum(times.values())-(S-1)*C)
 model.Minimize(z);target=model.Clone();target.ClearObjective();target.Add(z==lower);solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.linearization_level=2;deadline=started+max(.1,float(a.time_limit)-2);solver.parameters.max_time_in_seconds=max(.01,(deadline-time.monotonic())*.5);status=solver.Solve(target)
 if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
  solver.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());status=solver.Solve(model)
 if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError(solver.StatusName(status))
 out={'objective_value':solver.Value(z),'task_assignments':{t:solver.Value(station[t]) for t in times}};json.dump(out,open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(out['objective_value'])
if __name__=='__main__':main()

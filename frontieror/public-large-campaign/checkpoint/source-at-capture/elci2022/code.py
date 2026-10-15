import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));n=d['n'];M=d['m'];scenarios=d['scenarios'];m=cp_model.CpModel();assign=[[m.NewBoolVar('a%d_%d'%(i,k))for k in range(M)]for i in range(n)]
 for row in assign:m.Add(sum(row)==1)
 for i in range(n):
  for k in range(M):
   if d['capacity_requirements'][k][i]>d['facility_capacity_limits'][k]:m.Add(assign[i][k]==0)
 starts=[];ends=[];spans=[]
 for s,scenario in enumerate(scenarios):
  H=max(d['release_times'])+sum(max(scenario['processing_times'][k][i]for k in range(M))for i in range(n));ss=[];ee=[];ints=[[]for _ in range(M)];cons=[[]for _ in range(M)]
  for i in range(n):
   st=m.NewIntVar(d['release_times'][i],H,'s%d_%d'%(s,i));en=m.NewIntVar(0,H,'e%d_%d'%(s,i));ss.append(st);ee.append(en)
   for k in range(M):
    interval=m.NewOptionalIntervalVar(st,scenario['processing_times'][k][i],en,assign[i][k],'int%d_%d_%d'%(s,i,k));ints[k].append(interval);cons[k].append(d['capacity_requirements'][k][i])
  for k in range(M):m.AddCumulative(ints[k],cons[k],d['facility_capacity_limits'][k])
  span=m.NewIntVar(0,H,'span'+str(s));m.AddMaxEquality(span,ee);starts.append(ss);ends.append(ee);spans.append(span)
 m.Minimize(sum(int(round(scenario['probability']*1000000))*span for scenario,span in zip(scenarios,spans)));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.1,float(a.time_limit)-2-(time.monotonic()-start));status=solver.Solve(m)
 if status not in(cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible stochastic resource schedule found')
 selected=[next(k for k in range(M)if solver.Value(assign[i][k]))for i in range(n)];schedule={str(s):{str(i):{'facility':selected[i],'start':solver.Value(starts[s][i]),'finish':solver.Value(ends[s][i])}for i in range(n)}for s in range(len(scenarios))};value=sum(v['probability']*solver.Value(spans[s])for s,v in enumerate(scenarios));json.dump({'objective_value':value,'assignment':{str(i):selected[i]for i in range(n)},'schedule':schedule},open(a.solution_path,'w'))
if __name__=='__main__':main()

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();deadline=time.monotonic()+max(.01,float(a.time_limit)-2);d=json.load(open(a.instance_path));n=d['num_jobs'];M=d['num_machines'];P=d['jobs']['processing_times'];w=d['jobs']['weights'];due=d['jobs']['due_dates'];tardy=d['problem_type']=='weighted_tardy_jobs';sched=[[]for _ in range(M)];loads=[0]*M;left=[]
 order=sorted(range(n),key=lambda j:due[j] if tardy else min(P[j])/max(w[j],1))
 for j in order:
  choices=[(loads[k]+P[j][k],k)for k in range(M)];end,k=min(choices)
  if tardy and end>due[j]:left.append(j)
  else:sched[k].append(j);loads[k]=end
 if n*M<=5000 and time.monotonic()<deadline:
  m=cp_model.CpModel();H=sum(max(row)for row in P);ends=[];assign=[];intervals=[[]for _ in range(M)];late=[]
  for j in range(n):
   start=m.NewIntVar(0,H,'s'+str(j));end=m.NewIntVar(0,H,'e'+str(j));ends.append(end);row=[]
   for k in range(M):
    x=m.NewBoolVar('x%d_%d'%(j,k));row.append(x);intervals[k].append(m.NewOptionalIntervalVar(start,int(P[j][k]),end,x,'i%d_%d'%(j,k)))
   if tardy:
    z=m.NewBoolVar('late'+str(j));late.append(z);m.Add(sum(row)+z==1);m.Add(end<=int(due[j])).OnlyEnforceIf(z.Not())
   else:m.Add(sum(row)==1)
   assign.append(row)
  for ints in intervals:m.AddNoOverlap(ints)
  m.Minimize(sum(w[j]*(late[j]if tardy else ends[j])for j in range(n)));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());status=s.Solve(m)
  if status in(cp_model.FEASIBLE,cp_model.OPTIMAL):
   candidate=[sorted([j for j in range(n)if s.Value(assign[j][k])],key=lambda j:s.Value(ends[j]))for k in range(M)];sched=candidate;left=[j for j in range(n)if tardy and s.Value(late[j])]
 completion={}
 for k,row in enumerate(sched):
  t=0
  for j in row:t+=P[j][k];completion[str(j)]=t
 objective=sum(w[j]for j in left)if tardy else sum(w[j]*completion[str(j)]for j in range(n));sol={'objective_value':objective,'schedule':{str(k):row for k,row in enumerate(sched)},'completion_times':completion,'tardy_jobs':left,'on_time_jobs':[j for row in sched for j in row]};json.dump(sol,open(a.solution_path,'w'))
if __name__=='__main__':main()

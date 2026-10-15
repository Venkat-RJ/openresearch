import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json
from ortools.sat.python import cp_model
from release_contract import ReleasedInstance,completion_options,RELEASED_VARIANT
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));inst=ReleasedInstance(d);duties=d['active_duties']+d['reserve_duties'];model=cp_model.CpModel();options={u['duty_id']:completion_options(inst,u) for u in duties};x={};cover={t:[] for t in inst.tasks};cost=[]
 for duty,opts in options.items():
  bs=[]
  for k,o in enumerate(opts):
   b=model.NewBoolVar(duty+'_'+str(k));x[duty,k]=b;bs.append(b);cost.append(round(o['cost'])*b)
   for t in o['covered_tasks']:cover[t].append(b)
  model.Add(sum(bs)==1)
 cancel={}
 for t,bs in cover.items():
  b=model.NewBoolVar('cancel'+t);cancel[t]=b;model.Add(sum(bs)+b==1);kind=inst.tasks[t]['task_type'];cost.append(inst.params['cost_cancel_A_to_A' if kind=='A_to_A' else 'cost_cancel_A_to_B']*b)
 model.Minimize(sum(cost));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.1,float(a.time_limit)-2);status=solver.Solve(model)
 if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError(solver.StatusName(status))
 assignments={}
 for duty,opts in options.items():
  k=next(k for k in range(len(opts)) if solver.Value(x[duty,k]));assignments[duty]=dict(opts[k],completion_index=k)
 out={'released_problem_variant':RELEASED_VARIANT,'objective_value':solver.ObjectiveValue(),'canceled_tasks':[t for t,b in cancel.items() if solver.Value(b)],'duty_assignments':assignments};json.dump(out,open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(out['objective_value'])
if __name__=='__main__':main()

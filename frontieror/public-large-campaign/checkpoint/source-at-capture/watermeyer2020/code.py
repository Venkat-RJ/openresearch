import argparse,json,os,time
os.environ['OMP_NUM_THREADS']='1'
from ortools.sat.python import cp_model

def solve(d,limit):
 if 'processing_times' in d:
  durations=d['processing_times'];H=d['deadline'];constraints=[(a['from'],a['to'],a['delta']) for a in d['temporal_constraints']];resources=[(r['time_periods'],r['capacity'],r['demands']) for r in d['resources']];kind=2023
 elif isinstance(d['activities'],dict):
  durations=[d['activities'][str(i)]['processing_time'] for i in range(d['num_activities'])];H=d['d_bar'];constraints=[(a['i'],a['j'],a['delta']) for a in d['temporal_constraints']];resources=[(r['periods'],r['capacity'],[d['activities'][str(i)]['resource_demands'].get(str(k),0) for i in range(len(durations))]) for k,r in d['resources'].items()];kind=2020
 else:
  durations=[a['processing_time'] for a in d['activities']];H=d['max_project_duration'];constraints=[(a['from_activity'],a['to_activity'],a['time_lag']) for a in d['temporal_constraints']];resources=[(r['Pi_k'],r['capacity'],[r['demands'].get(str(i),0) for i in range(len(durations))]) for r in d['resources']];kind=2022
 n=len(durations);m=cp_model.CpModel();S=[m.new_int_var(0,H,f'S{i}') for i in range(n)];m.add(S[0]==0)
 for i,j,lag in constraints:m.add(S[j]>=S[i]+lag)
 for periods,capacity,demands in resources:
  terms=[]
  for i in range(n):
   if not demands[i] or not durations[i]:continue
   lookup=[sum(s<t<=s+durations[i] for t in periods)*demands[i] for s in range(H+1)];u=m.new_int_var(min(lookup),max(lookup),f'u{len(terms)}');m.add_element(S[i],lookup,u);terms.append(u)
  m.add(sum(terms)<=capacity)
 m.minimize(S[-1]);s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,limit-1);st=s.solve(m)
 if st not in (cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible project schedule '+s.status_name(st))
 start=[s.value(v) for v in S];out={'objective_value':start[-1]};out['schedule' if kind==2022 else 'start_times']=start if kind==2023 else {str(i):v for i,v in enumerate(start)};return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)

import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path',required=True);a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path);I=d['num_facilities'];J=d['num_customers'];m=cp_model.CpModel();y=[m.new_bool_var('y%d'%i) for i in range(I)];x=[[m.new_bool_var('x%d_%d'%(i,j)) for j in range(J)] for i in range(I)]
 for j in range(J):m.add_exactly_one(x[i][j] for i in range(I))
 for i in range(I):
  m.add(sum(d['demands'][j]*x[i][j] for j in range(J))<=d['capacities'][i]*y[i])
  for j in range(J):m.add(x[i][j]<=y[i])
 z=[]
 for q in [1,2]:z.append(sum(d['fixed_costs_obj%d'%q][i]*y[i] for i in range(I))+sum(d['assignment_costs_obj%d'%q][i][j]*x[i][j] for i in range(I) for j in range(J)))
 sols=[]
 def solve(objective,budget):
  m.minimize(objective);s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,budget);status=s.solve(m)
  if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):return None,status
  sol={'z1':int(s.value(z[0])),'z2':int(s.value(z[1])),'open_facilities':[i for i in range(I) if s.value(y[i])],'assignments':{str(j):next(i for i in range(I) if s.value(x[i][j])) for j in range(J)}};return sol,status
 def save(sol):
  if sol is not None and not any(v['z1']==sol['z1'] and v['z2']==sol['z2'] for v in sols):sols.append(sol)
  front=[v for v in sols if not any(w['z1']<=v['z1'] and w['z2']<=v['z2'] and (w['z1']<v['z1'] or w['z2']<v['z2']) for w in sols)];obj=min((v['z1']+v['z2'])/2 for v in front);json.dump({'objective_value':obj,'pareto_front':[[v['z1'],v['z2']] for v in front],'solutions':front},open(a.solution_path,'w'));logger.log_solution(obj,{'objective_value':obj,'pareto_front':[[v['z1'],v['z2']] for v in front],'solutions':front})
 sol,status=solve(z[0]+z[1],a.time_limit*.35)
 if sol is None:raise RuntimeError('No feasible single-source assignment')
 save(sol);B=sum(d['fixed_costs_obj2'])+sum(max(d['assignment_costs_obj2'][i][j] for i in range(I)) for j in range(J))+1
 while time.monotonic()-start<a.time_limit*.94:
  sol,status=solve(B*z[0]+z[1],a.time_limit*.94-(time.monotonic()-start))
  if sol is None:break
  save(sol)
  if status!=cp_model.OPTIMAL:break
  m.add(z[1]<=sol['z2']-1)
if __name__=='__main__':main()

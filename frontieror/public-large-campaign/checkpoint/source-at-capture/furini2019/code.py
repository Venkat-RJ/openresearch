import argparse,json,time
from ortools.sat.python import cp_model

def solve(d,t):
 n=d['num_vertices'];E={tuple(sorted(e)) for e in d['edges']};m=cp_model.CpModel();x=[m.new_bool_var(f'x{i}') for i in range(n)]
 for i in range(n):
  for j in range(i):
   if (j,i) not in E:m.add(x[i]+x[j]<=1)
 m.maximize(sum(x));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,t-1);r=s.solve(m)
 if r not in (cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No clique incumbent')
 vs=[i for i in range(n) if s.value(x[i])];return {'objective_value':len(vs),'clique_vertices':vs}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

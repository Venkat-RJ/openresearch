import argparse,json,time,os
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));n=d['n'];m=d['m'];c=d['linear_costs'];q=d['quadratic_costs'];u=d['arc_capacity'];model=gp.Model();model.Params.Seed=0;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;model.Params.Threads=1;model.Params.TimeLimit=max(.1,a.time_limit-3)
 x=model.addVars(n,m,lb=0,ub=u)
 model.setObjective(gp.quicksum(c[i][j]*x[i,j]+q[i][j]*x[i,j]*x[i,j] for i in range(n) for j in range(m)))
 for j in range(m):model.addConstr(gp.quicksum(x[i,j] for i in range(n))==d['demands'][j])
 for i in range(n):model.addConstr(gp.quicksum(x[i,j] for j in range(m))<=d['supplies'][i])
 model.optimize()
 if not model.SolCount:raise RuntimeError('No transportation solution')
 flows={f'x_{i}_{j}':float(v.X) for (i,j),v in x.items() if v.X>1e-10};obj=sum(c[int(k.split('_')[1])][int(k.split('_')[2])]*v+q[int(k.split('_')[1])][int(k.split('_')[2])]*v*v for k,v in flows.items());s={'objective_value':obj,'flows':flows};json.dump(s,open(a.solution_path,'w'))
 if a.log_path:
  logger=SolutionLogger(a.log_path,sense='minimize');logger.min_interval=0;logger.start_time=_logger_start;logger.log_solution(obj,s)
if __name__=='__main__':main()

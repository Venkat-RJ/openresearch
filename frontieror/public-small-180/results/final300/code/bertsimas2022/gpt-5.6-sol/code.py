import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import argparse,json,time,numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 n,k=data['n'],data['k'];mu=np.array(data['mu']);F=np.array(data['factor_loadings']);idio=np.array(data['idiosyncratic_variance']);gamma=data['gamma'];kappa=data['kappa'];cons=data.get('constraints',{})
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;x=m.addVars(n);z=m.addVars(n,vtype=GRB.BINARY);v=m.addVars(F.shape[1],lb=-GRB.INFINITY)
 m.addConstr(x.sum()==1);m.addConstr(z.sum()<=k)
 for i in range(n):
  m.addConstr(x[i]<=((cons.get('u_max_investment') or [1]*n)[i])*z[i])
  if cons.get('has_min_investment_constraint') and cons.get('l_min_investment') is not None:m.addConstr(x[i]>=cons['l_min_investment'][i]*z[i])
 for j in range(F.shape[1]):m.addConstr(v[j]==gp.quicksum(float(F[i,j])*x[i] for i in range(n)))
 if cons.get('has_min_return_constraint'):m.addConstr(gp.quicksum(float(mu[i])*x[i] for i in range(n))>=cons['r_bar'])
 for A,lo,hi in [(data.get('A'),data.get('l'),data.get('u')),(cons.get('A_lin'),cons.get('l_lin'),cons.get('u_lin'))]:
  if A is None:continue
  for j,row in enumerate(A):
   expr=gp.quicksum(coef*x[i] for i,coef in enumerate(row))
   if lo is not None:m.addConstr(expr>=lo[j])
   if hi is not None:m.addConstr(expr<=hi[j])
 m.setObjective(.5*gp.quicksum(v[j]*v[j] for j in range(F.shape[1]))+gp.quicksum(.5*(float(idio[i])+1/gamma)*x[i]*x[i]-kappa*float(mu[i])*x[i] for i in range(n)))
 def emit(values):
  arr=np.array(values);obj=.5*float(np.sum((F.T@arr)**2))+.5*float(np.dot(idio+1/gamma,arr*arr))-kappa*float(mu@arr);return {'objective_value':obj,'x':arr.tolist()}
 best=None
 def callback(model,where):
  nonlocal best
  if where==GRB.Callback.MIPSOL:
   result=emit(model.cbGetSolution([x[i] for i in range(n)]))
   if best is None or result['objective_value']<best['objective_value']:best=result
   if logger:logger.log_solution(result['objective_value'],result)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic()-1);m.optimize(callback)
 if not m.SolCount:
  if best is None:raise RuntimeError('no complete feasible portfolio incumbent')
  return best
 result=emit([x[i].X for i in range(n)])
 if best is None or result['objective_value']<best['objective_value']:best=result
 if logger:logger.log_solution(best['objective_value'],best)
 support=[i for i in range(n) if z[i].X>.5]
 for i in range(n):z[i].LB=int(i in support);z[i].UB=int(i in support);z[i].VType=GRB.CONTINUOUS
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize()
 if m.SolCount:
  result=emit([max(0,x[i].X) if i in support else 0 for i in range(n)])
  if best is None or result['objective_value']<best['objective_value']:best=result
 if logger:logger.log_solution(best['objective_value'],best)
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

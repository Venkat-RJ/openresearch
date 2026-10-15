import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 I=data['num_facilities'];J=data['num_customers'];m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;y=m.addVars(I,vtype=GRB.BINARY);x=m.addVars(I,J,vtype=GRB.BINARY)
 for j in range(J):m.addConstr(gp.quicksum(x[i,j] for i in range(I))==1)
 for i in range(I):
  m.addConstr(gp.quicksum(data['demands'][j]*x[i,j] for j in range(J))<=data['capacities'][i]*y[i])
  for j in range(J):m.addConstr(x[i,j]<=y[i])
 z=[gp.quicksum(data['fixed_costs_obj'+str(k)][i]*y[i] for i in range(I))+gp.quicksum(data['assignment_costs_obj'+str(k)][i][j]*x[i,j] for i in range(I) for j in range(J)) for k in [1,2]]
 m.setObjectiveN(z[0],0,priority=2,abstol=0,reltol=0);m.setObjectiveN(z[1],1,priority=1,abstol=0,reltol=0)
 sols=[];epsilon=None;best=float('inf')
 while time.monotonic()<deadline:
  m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize()
  if not m.SolCount:break
  a,b=[int(round(expr.getValue())) for expr in z];sol={'z1':a,'z2':b,'open_facilities':[i for i in range(I) if y[i].X>.5],'assignments':{str(j):next(i for i in range(I) if x[i,j].X>.5) for j in range(J)}};sols.append(sol)
  result={'objective_value':min((s['z1']+s['z2'])/2 for s in sols),'pareto_front':[[s['z1'],s['z2']] for s in sols],'solutions':sols[:]}
  if result['objective_value']<best:
   best=result['objective_value']
   if logger:logger.log_solution(best,result)
  if m.Status!=GRB.OPTIMAL:break
  if epsilon is None:epsilon=m.addConstr(z[1]<=b-1)
  else:epsilon.RHS=b-1
 if not sols:raise RuntimeError('no feasible Pareto point')
 sols=[s for s in sols if not any(t['z1']<=s['z1'] and t['z2']<=s['z2'] and (t['z1'],t['z2'])!=(s['z1'],s['z2']) for t in sols)];sols.sort(key=lambda s:s['z1'])
 return {'objective_value':min((s['z1']+s['z2'])/2 for s in sols),'pareto_front':[[s['z1'],s['z2']] for s in sols],'solutions':sols}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

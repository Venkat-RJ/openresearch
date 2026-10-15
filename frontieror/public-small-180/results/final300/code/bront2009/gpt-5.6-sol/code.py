import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 products=data['products'];legs=data['network']['legs'];ids=[p['product_id'] for p in products];segments=[(dict(zip(s['consideration_set'],s['preference_vector'])),s['no_purchase_preference'],s['lambda_l']) for s in data['segments']];columns=[];revenues=[];usage=[]
 for mask in range(1,1<<len(ids)):
  if time.monotonic()>=deadline:break
  S={ids[j] for j in range(len(ids)) if mask>>j&1};prob={pid:0. for pid in S}
  for weights,v0,arrival in segments:
   denom=v0+sum(weights.get(pid,0) for pid in S)
   for pid in S:prob[pid]+=arrival*weights.get(pid,0)/denom
  columns.append(sorted(S));revenues.append(sum(p['fare']*prob.get(p['product_id'],0) for p in products));usage.append([sum(prob.get(p['product_id'],0) for p in products if leg['leg_id'] in p['legs_used']) for leg in legs])
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;m.ModelSense=GRB.MAXIMIZE;t=m.addVars(len(columns),obj={k:revenues[k] for k in range(len(columns))})
 for i,leg in enumerate(legs):m.addConstr(gp.quicksum(usage[k][i]*t[k] for k in range(len(columns)))<=leg['capacity'])
 m.addConstr(t.sum()<=data['booking_horizon']['T']);m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize()
 if not m.SolCount:raise RuntimeError('no feasible offer allocation')
 active=[{'offer_set':columns[k],'time_allocated':max(0,t[k].X)} for k in range(len(columns)) if t[k].X>1e-9];sol={'objective_value':sum(revenues[k]*max(0,t[k].X) for k in range(len(columns))),'active_columns':active}
 if logger:logger.log_solution(sol['objective_value'],sol)
 return sol

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

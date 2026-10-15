import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 n=data['num_items'];T=data['num_periods'];d=data['demand'];fc=data['initial_inventory_cost'];cap=data['capacity']
 def expand(key):
  r=data[key];return r if isinstance(r[0],list) else [[v]*T for v in r]
 sc=expand('setup_cost');hc=expand('holding_cost');vc=expand('variable_cost');st=expand('setup_time');vt=expand('variable_time')
 model=gp.Model();model.Params.Threads=1;model.Params.Seed=0;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;model.Params.OutputFlag=0
 x=model.addVars(n,T,lb=0,name='x');y=model.addVars(n,T,vtype=GRB.BINARY,name='y');s0=model.addVars(n,lb=0,name='s0')
 inv={}
 for i in range(n):
  for t in range(T):
   inv[i,t]=s0[i]+gp.quicksum(x[i,u]-d[i][u] for u in range(t+1));model.addConstr(inv[i,t]>=0)
   ub=max(0,min(sum(d[i][t:]),(cap[t]-st[i][t])/vt[i][t] if vt[i][t]>0 else sum(d[i][t:])))
   model.addConstr(x[i,t]<=ub*y[i,t])
  model.addConstr(inv[i,T-1]==0)
 for t in range(T):model.addConstr(gp.quicksum(st[i][t]*y[i,t]+vt[i][t]*x[i,t] for i in range(n))<=cap[t])
 model.setObjective(gp.quicksum(fc[i]*s0[i]+gp.quicksum(sc[i][t]*y[i,t]+vc[i][t]*x[i,t]+hc[i][t]*inv[i,t] for t in range(T)) for i in range(n)))
 for i in range(n):
  s0[i].Start=sum(d[i]);
  for t in range(T):x[i,t].Start=0;y[i,t].Start=0
 def extract(get):
  prod=[[max(0,get(x[i,t])) for t in range(T)] for i in range(n)];setup=[[round(get(y[i,t])) for t in range(T)] for i in range(n)];stock=[]
  for i in range(n):
   row=[max(0,get(s0[i]))]
   for t in range(T):row.append(max(0,row[-1]+prod[i][t]-d[i][t]))
   stock.append(row)
  obj=sum(fc[i]*stock[i][0]+sum(sc[i][t]*setup[i][t]+vc[i][t]*prod[i][t]+hc[i][t]*stock[i][t+1] for t in range(T)) for i in range(n))
  return {'objective_value':obj,'production':prod,'setup':setup,'inventory':stock}
 def cb(m,where):
  if where==GRB.Callback.MIPSOL and logger:
   result=extract(m.cbGetSolution);logger.log_solution(result['objective_value'],result)
 model.Params.TimeLimit=max(.01,deadline-time.monotonic());model.optimize(cb)
 return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(0.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

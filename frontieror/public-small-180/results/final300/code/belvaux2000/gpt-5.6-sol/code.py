import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 pars=data['parameters'];n,K,T=pars['NI'],pars['NK'],pars['NT'];cost=data['costs'];f=cost['setup_cost_f'];g=cost['startup_cost_g'];h=cost['holding_cost_h'];e=cost['backlogging_cost_e'];cap=data['machine_data']['capacity_C'];sigma=data['machine_data']['startup_time_sigma'];d=data['demand'];s0=data['initial_conditions']['initial_stock_s0'];r0=data['initial_conditions']['initial_backlog_r0']
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0
 x=m.addVars(n,K,T);y=m.addVars(n,K,T,vtype=GRB.BINARY);z=m.addVars(n,K,T,vtype=GRB.BINARY);r=m.addVars(n,T);stock={}
 for i in range(n):
  for t in range(T):
   stock[i,t]=s0[i]-r0[i]+gp.quicksum(x[i,k,u] for k in range(K) for u in range(t+1))-sum(d[i][:t+1])+r[i,t];m.addConstr(stock[i,t]>=0);r[i,t].Start=max(0,r0[i]-s0[i]+sum(d[i][:t+1]))
   for k in range(K):
    prev=y[i,k,t-1] if t else 0;m.addConstr(x[i,k,t]+sigma[k]*z[i,k,t]<=cap[k]*y[i,k,t]);m.addConstr(z[i,k,t]>=y[i,k,t]-prev);m.addConstr(z[i,k,t]<=y[i,k,t]);m.addConstr(prev+z[i,k,t]+gp.quicksum(y[j,k,t]-z[j,k,t] for j in range(n) if j!=i)<=1);x[i,k,t].Start=0;y[i,k,t].Start=0;z[i,k,t].Start=0
 for k in range(K):
  for t in range(T):m.addConstr(gp.quicksum(y[i,k,t] for i in range(n))<=1)
 m.setObjective(gp.quicksum(f*y[i,k,t]+g*z[i,k,t] for i in range(n) for k in range(K) for t in range(T))+gp.quicksum(h[i]*stock[i,t]+e[i]*r[i,t] for i in range(n) for t in range(T)))
 def extract(get):
  out={'production':{},'stock':{},'backlog':{},'setup':{},'startup':{}};obj=0
  for i in range(n):
   net=s0[i]-r0[i]
   for t in range(T):
    prod=0
    for k in range(K):
     key=f'{i}_{k}_{t}';qty=max(0,get(x[i,k,t]));yv=round(get(y[i,k,t]));zv=round(get(z[i,k,t]));out['production'][key]=qty;out['setup'][key]=yv;out['startup'][key]=zv;prod+=qty;obj+=f*yv+g*zv
    net+=prod-d[i][t];back=max(0,get(r[i,t]));stk=max(0,net+back);key=f'{i}_{t}';out['backlog'][key]=back;out['stock'][key]=stk;obj+=h[i]*stk+e[i]*back
  out['objective_value']=obj;return out
 def cb(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb);return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

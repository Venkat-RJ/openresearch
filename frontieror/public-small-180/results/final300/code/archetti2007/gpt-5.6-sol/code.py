import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 n=data['num_retailers'];H=data['time_horizon'];sup=data['supplier'];ret={a['id']:a for a in data['retailers']};cap=data['vehicle_capacity'];dist=data['distance_matrix'];nodes=range(n+1);times=range(1,H+1);ext=range(1,H+2);edges=[(i,j) for i in nodes for j in range(i)]
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;m.Params.LazyConstraints=1
 q=m.addVars(range(1,n+1),times);stock=m.addVars(nodes,ext);z=m.addVars(nodes,times,vtype=GRB.BINARY)
 x={(i,j,t):m.addVar(vtype=GRB.INTEGER if j==0 else GRB.BINARY,ub=2 if j==0 else 1) for i,j in edges for t in times}
 for i in nodes:m.addConstr(stock[i,1]==(sup['starting_inventory'] if i==0 else ret[i]['starting_inventory']))
 for t in times:
  qty=gp.quicksum(q[i,t] for i in range(1,n+1));m.addConstr(stock[0,t]>=qty);m.addConstr(stock[0,t+1]==stock[0,t]+sup['production_rate']-qty);m.addConstr(qty<=cap*z[0,t])
  for i in range(1,n+1):
   a=ret[i];m.addConstr(stock[i,t+1]==stock[i,t]+q[i,t]-a['demand']);m.addConstr(q[i,t]>=a['max_inventory']*z[i,t]-stock[i,t]);m.addConstr(q[i,t]<=a['max_inventory']-stock[i,t]);m.addConstr(q[i,t]<=a['max_inventory']*z[i,t]);m.addConstr(z[i,t]<=z[0,t])
  for i in nodes:m.addConstr(gp.quicksum(x[u,v,t] for u,v in edges if i in (u,v))==2*z[i,t])
 m.setObjective(gp.quicksum((sup['inventory_cost'] if i==0 else ret[i]['inventory_cost'])*stock[i,t] for i in nodes for t in ext)+gp.quicksum(dist[i][j]*x[i,j,t] for i,j in edges for t in times))
 def extract(get):
  delivery={str(t):{str(i):max(0,get(q[i,t])) for i in range(1,n+1)} for t in times};routes={str(t):[[j,i,round(get(x[i,j,t]))] for i,j in edges if get(x[i,j,t])>.5] for t in times};inv={i:{str(t):max(0,get(stock[i,t])) for t in ext} for i in nodes};obj=sum((sup['inventory_cost'] if i==0 else ret[i]['inventory_cost'])*sum(inv[i].values()) for i in nodes)+sum(dist[j][i]*k for route in routes.values() for j,i,k in route)
  return {'objective_value':obj,'deliveries':delivery,'routes':routes,'supplier_inventory':inv[0],'retailer_inventory':{str(i):inv[i] for i in range(1,n+1)}}
 def cb(model,where):
  if where!=GRB.Callback.MIPSOL:return
  vals=model.cbGetSolution(x);zv=model.cbGetSolution(z);bad=False
  for t in times:
   adj={i:set() for i in nodes}
   for i,j in edges:
    if vals[i,j,t]>.5:adj[i].add(j);adj[j].add(i)
   remaining={i for i in nodes if zv[i,t]>.5}
   while remaining:
    todo=[next(iter(remaining))];comp=set()
    while todo:
     i=todo.pop()
     if i in comp:continue
     comp.add(i);todo.extend(adj[i]-comp)
    remaining-=comp
    if 0 not in comp:
     bad=True;k=min(comp);model.cbLazy(gp.quicksum(x[i,j,t] for i,j in edges if i in comp and j in comp)<=gp.quicksum(z[i,t] for i in comp)-z[k,t])
  if not bad and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

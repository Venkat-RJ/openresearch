import argparse,json,time,os,random
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 pars=data['parameters'];n=pars['num_requests_n'];s=pars['num_stacks_s'];height=pars['stack_height_l'];cost=[data['pickup_region']['distance_matrix'],data['delivery_region']['distance_matrix']];N=range(n+1)
 def total(tours):return sum(cost[k][tour[i]][tour[(i+1)%len(tour)]] for k,tour in enumerate(tours) for i in range(len(tour)))
 remaining=set(range(1,n+1));p=[0]
 while remaining:
  v=min(remaining,key=lambda j:cost[0][p[-1]][j]+cost[1][j][p[-1]]);p.append(v);remaining.remove(v)
 best={'objective_value':total([p,[0]+p[:0:-1]]),'pickup_tour':p,'delivery_tour':[0]+p[:0:-1]}
 if logger:logger.log_solution(best['objective_value'],best)
 model=gp.Model();model.Params.Threads=1;model.Params.Seed=0;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;model.Params.OutputFlag=0;model.Params.LazyConstraints=1
 x={};
 for k in range(2):
  for i in N:
   for j in N:
    if i!=j:x[k,i,j]=model.addVar(vtype=GRB.BINARY,obj=cost[k][i][j])
 for k in range(2):
  for i in N:
   model.addConstr(gp.quicksum(x[k,i,j] for j in N if i!=j)==1);model.addConstr(gp.quicksum(x[k,j,i] for j in N if i!=j)==1)
  for i in N:
   for j in range(i+1,n+1):model.addConstr(x[k,i,j]+x[k,j,i]<=1)
 for k,tour in enumerate([best['pickup_tour'],best['delivery_tour']]):
  edges=set(zip(tour,tour[1:]+tour[:1]))
  for i in N:
   for j in N:
    if i!=j:x[k,i,j].Start=int((i,j) in edges)
 def loading(p,d):
  pos={v:i for i,v in enumerate(d[1:])};stacks=[[] for _ in range(s)]
  def dfs(i):
   if i==n:return True
   v=p[i+1]
   for stack in stacks:
    if len(stack)<height and (not stack or pos[v]<pos[stack[-1]]):
     stack.append(v)
     if dfs(i+1):return True
     stack.pop()
    if not stack:break
   return False
  return dfs(0)
 def cb(m,where):
  nonlocal best
  if where!=GRB.Callback.MIPSOL:return
  vals=m.cbGetSolution(x);tours=[];sub=False
  for k in range(2):
   nxt={i:j for i in N for j in N if i!=j and vals[k,i,j]>.5};seen=set();cycles=[]
   for v in N:
    if v in seen:continue
    cycle=[];u=v
    while u not in seen:cycle.append(u);seen.add(u);u=nxt[u]
    cycles.append(cycle)
   if len(cycles)>1:
    sub=True
    for cycle in cycles:m.cbLazy(gp.quicksum(x[k,i,j] for i in cycle for j in cycle if i!=j)<=len(cycle)-1)
   else:
    tour=[0];u=nxt[0]
    while u!=0:tour.append(u);u=nxt[u]
    tours.append(tour)
  if sub:return
  if not loading(*tours):
   m.cbLazy(gp.quicksum(x[k,tours[k][i],tours[k][i+1]] for k in range(2) for i in range(1,n))<=2*(n-1)-1);return
  obj=total(tours)
  if obj<best['objective_value']:
   best={'objective_value':obj,'pickup_tour':tours[0],'delivery_tour':tours[1]}
   if logger:logger.log_solution(obj,best)
 model.Params.TimeLimit=max(.01,deadline-time.monotonic());model.optimize(cb)
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(0.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

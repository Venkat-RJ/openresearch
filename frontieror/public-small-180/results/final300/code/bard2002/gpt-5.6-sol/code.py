import argparse,json,time,os,math
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 nodes=[data['depot']]+data['customers'];n=len(nodes);N=range(n);I=range(1,n);cap=data['vehicle_capacity'];ready=[a['ready_time'] for a in nodes];due=[a['due_date'] for a in nodes];dem=[a['demand'] for a in nodes]
 dist=data.get('distance_matrix') or [[0 if i==j else math.floor(math.hypot(nodes[i]['x']-nodes[j]['x'],nodes[i]['y']-nodes[j]['y'])*10)/10 for j in N] for i in N]
 tau=[[0 if i==j else max(dist[i][j]+(0 if data.get('travel_time_includes_service_time',False) else nodes[i]['service_time']),ready[j]-due[i]) for j in N] for i in N]
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0
 arcs=[(i,j) for i in N for j in N if i!=j and ready[i]+tau[i][j]<=due[j]+1e-8 and (i==0 or j==0 or dem[i]+dem[j]<=cap)]
 x=m.addVars(arcs,vtype=GRB.BINARY);t={i:m.addVar(lb=ready[i],ub=due[i]) for i in I};load={i:m.addVar(lb=dem[i],ub=cap) for i in I}
 for i in I:m.addConstr(gp.quicksum(x[i,j] for j in N if (i,j) in x)==1)
 for i in N:m.addConstr(gp.quicksum(x[i,j] for j in N if (i,j) in x)==gp.quicksum(x[j,i] for j in N if (j,i) in x))
 for i in I:
  for j in I:
   if i==j:continue
   var=x.get((i,j),0);m.addConstr(t[j]>=t[i]+tau[i][j]*var-(due[i]-ready[j])*(1-var));m.addConstr(load[j]>=load[i]+dem[j]-(cap-dem[j])*(1-var))
 m.setObjective(gp.quicksum(x[0,j] for j in I if (0,j) in x))
 for i,j in arcs:x[i,j].Start=int(i==0 or j==0)
 for i in I:t[i].Start=max(tau[0][i],ready[i]);load[i].Start=dem[i]
 def extract(get):
  succ={i:j for i,j in arcs if i!=0 and get(x[i,j])>.5};routes=[]
  for j in I:
   if (0,j) not in x or get(x[0,j])<.5:continue
   route=[0,j];u=succ[j]
   while u:route.append(u);u=succ[u]
   route.append(0);routes.append(route)
  distance=round(sum(dist[a][b] for route in routes for a,b in zip(route,route[1:])),1)
  return {'objective_value':len(routes),'num_vehicles':len(routes),'routes':routes,'total_distance':distance,'departure_times':{str(i):get(t[i]) for i in I},'loads':{str(i):get(load[i]) for i in I}}
 def cb(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

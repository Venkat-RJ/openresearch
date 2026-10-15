import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def config(m):
 m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0

def solve(data,deadline,logger):
 I=data['num_facilities'];J=data['num_customers'];sc=data['scenarios'];K=len(sc);cost=data['transportation_costs'];cap=[f['capacity'] for f in data['facilities']];fixed=[f['opening_cost'] for f in data['facilities']];G=min(10,K);groups=[[k for k in range(K) if k%G==g] for g in range(G)]
 master=gp.Model();config(master);x=master.addVars(I,vtype=GRB.BINARY);theta=master.addVars(G);master.addConstr(gp.quicksum(cap[i]*x[i] for i in range(I))>=max(sum(s['demands']) for s in sc));master.setObjective(gp.quicksum(fixed[i]*x[i] for i in range(I))+theta.sum())
 lp=gp.Model();config(lp);y=lp.addVars(I,J,obj={(i,j):cost[i][j] for i in range(I) for j in range(J)});dem=[lp.addConstr(gp.quicksum(y[i,j] for i in range(I))==sc[0]['demands'][j]) for j in range(J)];caps=[lp.addConstr(gp.quicksum(y[i,j] for j in range(J))<=cap[i]) for i in range(I)];lp.Params.Method=1
 best=None;opened=[1]*I;duration=0.;seen=set()
 for iteration in range(170):
  tick=time.monotonic();witness={};values=[0.]*G;constant=[0.]*G;coeff=[[0.]*I for _ in range(G)];complete=True
  for k,s in enumerate(sc):
   if time.monotonic()>=deadline:complete=False;break
   for i in range(I):caps[i].RHS=cap[i]*opened[i]
   for j in range(J):dem[j].RHS=s['demands'][j]
   lp.Params.TimeLimit=max(.01,deadline-time.monotonic());lp.optimize()
   if lp.Status!=GRB.OPTIMAL:complete=False;break
   prob=s['probability'];g=k%G;values[g]+=prob*lp.ObjVal;constant[g]+=prob*sum(dem[j].Pi*s['demands'][j] for j in range(J))
   for i in range(I):coeff[g][i]+=prob*caps[i].Pi*cap[i]
   witness[str(k)]={str(i):{str(j):y[i,j].X for j in range(J) if y[i,j].X>1e-10} for i in range(I) if opened[i]}
  if not complete:break
  obj=sum(fixed[i]*opened[i] for i in range(I))+sum(values)
  if best is None or obj<best['objective_value']-1e-7:
   best={'objective_value':obj,'open_facilities':[i for i in range(I) if opened[i]],'x':{str(i):opened[i] for i in range(I)},'y':witness}
   if logger:logger.log_solution(obj,best)
  for g in range(G):master.addConstr(theta[g]>=constant[g]+gp.quicksum(coeff[g][i]*x[i] for i in range(I)))
  duration=max(duration,time.monotonic()-tick);seen.add(tuple(opened))
  if deadline-time.monotonic()<duration+0.3:break
  for i in range(I):x[i].Start=opened[i]
  master.Params.TimeLimit=max(.01,min(deadline-time.monotonic()-duration,2.));master.optimize()
  if not master.SolCount:break
  if best['objective_value']-master.ObjBound<=1e-4*max(1,abs(best['objective_value'])):break
  opened=[int(x[i].X>.5) for i in range(I)]
  if tuple(opened) in seen:break
 if best is None:raise RuntimeError('no complete scenario witness within deadline')
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

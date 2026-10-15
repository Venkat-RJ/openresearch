import argparse,json,math,os,time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));n=d['num_users'];K=d['num_vehicles'];end=2*n+1;V=range(1,end);nd={v['node_id']:v for v in d['nodes']};dist={(i,j):math.hypot(nd[i]['x']-nd[j]['x'],nd[i]['y']-nd[j]['y']) for i in range(end+1) for j in range(end+1)};Q=d['vehicle_capacity'];L=d['maximum_ride_time'];A=[(i,j) for i in range(end) for j in range(1,end+1) if i!=j and not(i==0 and n<j<end) and not(1<=i<=n and j==end) and not(n<i<end and j==i-n) and nd[i]['earliest_time']+nd[i]['service_duration']+dist[i,j]<=nd[j]['latest_time']+1e-9];inc={i:[e for e in A if e[1]==i] for i in range(end+1)};out={i:[e for e in A if e[0]==i] for i in range(end+1)}
 model=gp.Model();model.Params.Threads=1;model.Params.Seed=0;model.Params.NumericFocus=0;model.Params.MIPGap=1e-4;x=model.addVars([(k,i,j) for k in range(K) for i,j in A],vtype=GRB.BINARY);B={i:model.addVar(lb=nd[i]['earliest_time'],ub=nd[i]['latest_time']) for i in V};load={i:model.addVar(lb=max(0,nd[i]['load']),ub=min(Q,Q+nd[i]['load'])) for i in V};bs={k:model.addVar(lb=nd[0]['earliest_time'],ub=nd[0]['latest_time']) for k in range(K)};be={k:model.addVar(lb=nd[end]['earliest_time'],ub=nd[end]['latest_time']) for k in range(K)}
 for i in V:model.addConstr(gp.quicksum(x[k,j,i] for k in range(K) for j,_ in inc[i])==1)
 for k in range(K):
  model.addConstr(gp.quicksum(x[k,0,j] for _,j in out[0])==1);model.addConstr(gp.quicksum(x[k,i,end] for i,_ in inc[end])==1);model.addConstr(be[k]-bs[k]<=d['maximum_route_duration'])
  for i in V:model.addConstr(gp.quicksum(x[k,j,i] for j,_ in inc[i])==gp.quicksum(x[k,i,j] for _,j in out[i]))
  for i in range(1,n+1):model.addConstr(gp.quicksum(x[k,i,j] for _,j in out[i])==gp.quicksum(x[k,n+i,j] for _,j in out[n+i]))
 for i,j in A:
  if i==0 and j==end:
   for k in range(K):model.addConstr(be[k]>=bs[k]+dist[i,j]-(nd[i]['latest_time']+dist[i,j]-nd[j]['earliest_time'])*(1-x[k,i,j]))
  elif i==0:
   for k in range(K):model.addConstr(B[j]>=bs[k]+dist[i,j]-max(0,nd[i]['latest_time']+dist[i,j]-nd[j]['earliest_time'])*(1-x[k,i,j]))
  elif j==end:
   for k in range(K):model.addConstr(be[k]>=B[i]+nd[i]['service_duration']+dist[i,j]-max(0,nd[i]['latest_time']+nd[i]['service_duration']+dist[i,j]-nd[j]['earliest_time'])*(1-x[k,i,j]));model.addConstr(0>=load[i]-Q*(1-x[k,i,j]))
  else:
   used=gp.quicksum(x[k,i,j] for k in range(K));model.addConstr(B[j]>=B[i]+nd[i]['service_duration']+dist[i,j]-max(0,nd[i]['latest_time']+nd[i]['service_duration']+dist[i,j]-nd[j]['earliest_time'])*(1-used));model.addConstr(load[j]>=load[i]+nd[j]['load']-(Q+abs(nd[j]['load']))*(1-used))
 for i in range(1,n+1):model.addConstr(B[n+i]-B[i]-nd[i]['service_duration']>=dist[i,n+i]);model.addConstr(B[n+i]-B[i]-nd[i]['service_duration']<=L)
 model.setObjective(gp.quicksum(dist[i,j]*x[k,i,j] for k in range(K) for i,j in A));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  routes={};obj=0
  for k in range(K):
   succ={i:j for i,j in A if get(x[k,i,j])>.5};r=[0]
   while r[-1]!=end:r.append(succ[r[-1]])
   routes[str(k)]=r;obj+=sum(dist[i,j] for i,j in zip(r,r[1:]))
  times={str(i):get(B[i]) for i in V};times.update({f'depot_start_{k}':get(bs[k]) for k in range(K)});times.update({f'depot_end_{k}':get(be[k]) for k in range(K)});return {'objective_value':obj,'routes':routes,'service_times':times,'ride_times':{str(i):times[str(n+i)]-times[str(i)]-nd[i]['service_duration'] for i in range(1,n+1)}}
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   s=extract(md.cbGetSolution);json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(s['objective_value'],s)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No DARP incumbent')
if __name__=='__main__':main()

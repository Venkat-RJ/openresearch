import argparse,json,os,time,math,random
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();begin=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));n=d['num_customers'];Q=d['vehicle_capacity'];dist=d['distance_matrix'];nodes={v['id']:v for v in [d['depot']]+d['customers']};pool={};rng=random.Random(0);deadline=begin+min(12,a.time_limit*.22)
 def save(seq,mask,cost):
  if not seq:return
  total=cost+dist[seq[-1]][0]
  if mask not in pool or total<pool[mask][0]:pool[mask]=(total,tuple(seq))
 def dfs(seq,mask,last,t,load,cost):
  if time.monotonic()>deadline or len(pool)>100000:return
  if seq and t+nodes[last]['service_time']+dist[last][0]<=nodes[0]['time_window'][1]+1e-8:save(seq,mask,cost)
  for j in range(1,n+1):
   if mask>>(j-1)&1 or load+nodes[j]['demand']>Q:continue
   nt=max(nodes[j]['time_window'][0],t+nodes[last]['service_time']+dist[last][j])
   if nt<=nodes[j]['time_window'][1]+1e-8 and nt+nodes[j]['service_time']+dist[j][0]<=nodes[0]['time_window'][1]+1e-8:dfs(seq+[j],mask|1<<(j-1),j,nt,load+nodes[j]['demand'],cost+dist[last][j])
 for j in range(1,n+1):
  t=max(nodes[j]['time_window'][0],nodes[0]['time_window'][0]+nodes[0]['service_time']+dist[0][j]);save([j],1<<(j-1),dist[0][j])
 dfs([],0,0,nodes[0]['time_window'][0],0,0);single=[(mask,v) for mask,v in pool.items() if mask.bit_count()==1];other=sorted(((mask,v) for mask,v in pool.items() if mask.bit_count()>1),key=lambda z:z[1][0]/z[0].bit_count());chosen=dict(single+other[:1800-len(single)]);routes=list(chosen.values());masks=list(chosen);model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;y=model.addVars(len(routes),vtype=GRB.BINARY)
 for j in range(n):model.addConstr(gp.quicksum(y[r] for r,mask in enumerate(masks) if mask>>j&1)==1)
 model.setObjective(gp.quicksum(cost*y[r] for r,(cost,seq) in enumerate(routes)));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  selected=[r for r in range(len(routes)) if get(y[r])>.5];result=[{'vehicle':k,'route':[0]+list(routes[r][1])+[n+1],'deliveries':{str(j):nodes[j]['demand'] for j in routes[r][1]}} for k,r in enumerate(selected)];return {'objective_value':sum(routes[r][0] for r in selected),'num_vehicles_used':len(result),'routes':result,'deliveries':{str(j):{'demand':nodes[j]['demand'],'total_delivered':nodes[j]['demand']} for j in range(1,n+1)}}
 for r,mask in enumerate(masks):y[r].Start=int(mask.bit_count()==1)
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   s=extract(md.cbGetSolution);json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(s['objective_value'],s)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-begin)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No route partition incumbent')
if __name__=='__main__':main()

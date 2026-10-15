import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def config(m,deadline):
 m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;m.Params.TimeLimit=max(.01,deadline-time.monotonic())

def solve(data,deadline,logger):
 jobs={j['job_id']:j for j in data['jobs']};projects={p['project_id']:p for p in data['projects']};renew=data['resources'].get('renewable',[]);nonrenew=data['resources'].get('nonrenewable',[]);pred={j:[] for j in jobs}
 for j,a in jobs.items():
  for s in a['successors']:pred[s].append(j)
 order=[];left=set(jobs)
 while left:
  j=min(j for j in left if all(v in order for v in pred[j]));order.append(j);left.remove(j)
 est={j:projects[a['project_id']]['release_date'] for j,a in jobs.items()};mind={j:min(m['duration'] for m in a['modes']) for j,a in jobs.items()}
 for j in order:
  for s in jobs[j]['successors']:est[s]=max(est[s],est[j]+mind[j])
 cpd={p:est[a['artificial_sink_job_id']]-a['release_date'] for p,a in projects.items()}
 mode=gp.Model();config(mode,min(deadline,time.monotonic()+3));v={(j,k):mode.addVar(vtype=GRB.BINARY,obj=b['duration']) for j,a in jobs.items() for k,b in enumerate(a['modes'])}
 for j,a in jobs.items():mode.addConstr(gp.quicksum(v[j,k] for k in range(len(a['modes'])))==1)
 for res in nonrenew:mode.addConstr(gp.quicksum(b['nonrenewable_consumption'][res['resource_id']]*v[j,k] for j,a in jobs.items() for k,b in enumerate(a['modes']))<=res['capacity'])
 for j,a in jobs.items():
  for k,b in enumerate(a['modes']):
   if any(b['renewable_consumption'][res['resource_id']]>res['capacity'] for res in renew):v[j,k].UB=0
 mode.optimize();selected={j:next(k for k in range(len(a['modes'])) if v[j,k].X>.5) for j,a in jobs.items()};mode.dispose()
 starts={};ends={};usage={res['resource_id']:{} for res in renew}
 for j in order:
  b=jobs[j]['modes'][selected[j]];dur=b['duration'];st=max([projects[jobs[j]['project_id']]['release_date']]+[ends[v] for v in pred[j]])
  while any(usage[res['resource_id']].get(t,0)+b['renewable_consumption'][res['resource_id']]>res['capacity'] for res in renew for t in range(st,st+dur)):st+=1
  starts[j]=st;ends[j]=st+dur
  for res in renew:
   for t in range(st,st+dur):usage[res['resource_id']][t]=usage[res['resource_id']].get(t,0)+b['renewable_consumption'][res['resource_id']]
 def emit(assignment):
  end={j:t+jobs[j]['modes'][k]['duration'] for j,(k,t) in assignment.items()};obj=sum(end[p['artificial_sink_job_id']]-p['release_date']-cpd[pid] for pid,p in projects.items());return {'objective_value':obj,'schedule':[{'job_id':j,'mode_id':jobs[j]['modes'][k]['mode_id'],'start_time':t} for j,(k,t) in sorted(assignment.items())],'makespan':max(end.values())}
 best=emit({j:(selected[j],starts[j]) for j in jobs})
 if logger:logger.log_solution(best['objective_value'],best)
 H=max(ends.values());tail={j:0 for j in jobs}
 for j in reversed(order):tail[j]=max([0]+[mind[s]+tail[s] for s in jobs[j]['successors']])
 m=gp.Model();config(m,deadline);x={}
 for j,a in jobs.items():
  for k,b in enumerate(a['modes']):
   if any(b['renewable_consumption'][res['resource_id']]>res['capacity'] for res in renew):continue
   for t in range(est[j],H-b['duration']-tail[j]+1):x[j,k,t]=m.addVar(vtype=GRB.BINARY)
 if len(x)>1900:return best
 start={j:gp.quicksum(t*v for (i,k,t),v in x.items() if i==j) for j in jobs};end={j:gp.quicksum((t+jobs[j]['modes'][k]['duration'])*v for (i,k,t),v in x.items() if i==j) for j in jobs}
 for j in jobs:
  m.addConstr(gp.quicksum(v for (i,k,t),v in x.items() if i==j)==1)
  for s in jobs[j]['successors']:m.addConstr(end[j]<=start[s])
 for res in nonrenew:m.addConstr(gp.quicksum(jobs[j]['modes'][k]['nonrenewable_consumption'][res['resource_id']]*v for (j,k,t),v in x.items())<=res['capacity'])
 for res in renew:
  for timepoint in range(H):m.addConstr(gp.quicksum(jobs[j]['modes'][k]['renewable_consumption'][res['resource_id']]*v for (j,k,t),v in x.items() if t<=timepoint<t+jobs[j]['modes'][k]['duration'])<=res['capacity'])
 m.setObjective(gp.quicksum(end[p['artificial_sink_job_id']]-p['release_date']-cpd[pid] for pid,p in projects.items()))
 for (j,k,t),v in x.items():v.Start=int(k==selected[j] and t==starts[j])
 def extract(get):return emit({j:(k,t) for (j,k,t),v in x.items() if get(v)>.5})
 def cb(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 if m.SolCount:best=extract(lambda v:v.X)
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

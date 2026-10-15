import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 jobs=data['jobs'];ops={(j['job_id'],k):op for j in jobs for k,op in enumerate(j['operations'])};H=sum(op['processing_time'] for op in ops.values())+max(j['release_date'] for j in jobs)
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;start=m.addVars(list(ops),ub=H);tard=m.addVars([j['job_id'] for j in jobs]);m.setObjective(gp.quicksum(j['weight']*tard[j['job_id']] for j in jobs))
 machine={}
 for key,op in ops.items():machine.setdefault(op['machine'],[]).append(key)
 for j in jobs:
  jid=j['job_id'];m.addConstr(start[jid,0]>=j['release_date'])
  for k in range(1,len(j['operations'])):m.addConstr(start[jid,k]>=start[jid,k-1]+ops[jid,k-1]['processing_time'])
  last=len(j['operations'])-1;m.addConstr(tard[jid]>=start[jid,last]+ops[jid,last]['processing_time']-j['due_date'])
 disj={}
 for keys in machine.values():
  for ai,a in enumerate(keys):
   for b in keys[ai+1:]:
    v=m.addVar(vtype=GRB.BINARY);disj[a,b]=v;m.addConstr(start[b]>=start[a]+ops[a]['processing_time']-H*(1-v));m.addConstr(start[a]>=start[b]+ops[b]['processing_time']-H*v)
 availability={m:0 for m in machine};seed={}
 for j in sorted(jobs,key=lambda j:j['due_date']/j['weight']):
  t=j['release_date']
  for k,op in enumerate(j['operations']):
   t=max(t,availability[op['machine']]);seed[j['job_id'],k]=t;t+=op['processing_time'];availability[op['machine']]=t
 for key,v in start.items():v.Start=seed[key]
 for (a,b),v in disj.items():v.Start=int(seed[a]<seed[b])
 for j in jobs:
  key=j['job_id'],len(j['operations'])-1;tard[j['job_id']].Start=max(0,seed[key]+ops[key]['processing_time']-j['due_date'])
 def extract(get):
  schedule=[];obj=0
  for j in jobs:
   jid=j['job_id'];last=jid,len(j['operations'])-1;completion=get(start[last])+ops[last]['processing_time'];late=max(0,completion-j['due_date']);obj+=j['weight']*late;schedule.append({'job_id':jid,'completion_time':completion,'tardiness':late,'operations':[{'machine':op['machine'],'start_time':max(0,get(start[jid,k]))} for k,op in enumerate(j['operations'])]})
  return {'objective_value':obj,'schedule':schedule}
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

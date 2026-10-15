import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
_PROCESS_START=time.monotonic()
import highspy
from solution_logger import SolutionLogger

def main():
 start=_PROCESS_START;p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','log_path'):p.add_argument('--'+k,required=True)
 p.add_argument('--time_limit',type=int,default=60);a=p.parse_args();deadline=start+max(0.,a.time_limit-5.);data=json.load(open(a.instance_path));jobs=data['jobs'];projects=data['projects'];res=data['resources'];logger=SolutionLogger(a.log_path,sense='minimize');by={j['job_id']:j for j in jobs};release={p['project_id']:p['release_date'] for p in projects};H=max(release.values(),default=0)+sum(max(m['duration'] for m in j['modes']) for j in jobs);constants={};pred={i:[] for i in by}
 for j in jobs:
  for s in j['successors']:pred[s].append(j['job_id'])
 order=[];remaining=set(by)
 while remaining:
  ready=[i for i in remaining if not any(k in remaining for k in pred[i])]
  if not ready:raise ValueError('cyclic precedence')
  order.extend(sorted(ready));remaining.difference_update(ready)
 for project in projects:
  pid=project['project_id'];ids={j['job_id'] for j in jobs if j['project_id']==pid};est={i:project['release_date'] for i in ids}
  for i in order:
   if i not in ids:continue
   for s in by[i]['successors']:
    if s in ids:est[s]=max(est[s],est[i]+min(m['duration'] for m in by[i]['modes']))
  constants[pid]=est[project['artificial_sink_job_id']]
 best=float('inf');best_schedule=None
 def save(schedule):
  nonlocal best,best_schedule
  mapping={e['job_id']:e['start_time'] for e in schedule};obj=sum(mapping[p['artificial_sink_job_id']]-constants[p['project_id']] for p in projects)
  if obj>=best:return
  best=obj;best_schedule=schedule;sol={'objective_value':obj,'schedule':schedule,'makespan':max(mapping[p['artificial_sink_job_id']] for p in projects)};json.dump(sol,open(a.solution_path,'w'))
  if hasattr(logger,'log_solution'):logger.log_solution(obj,sol)
  else:logger.log(obj)
 allowed={i:[m for m in j['modes'] if all(m['renewable_consumption'][r['resource_id']]<=r['capacity'] for r in res['renewable'])] for i,j in by.items()}
 selected={i:min(row,key=lambda m:(m['duration'],sum(m['renewable_consumption']),sum(m['nonrenewable_consumption']))) for i,row in allowed.items()}
 if any(sum(selected[i]['nonrenewable_consumption'][r['resource_id']] for i in by)>r['capacity'] for r in res['nonrenewable']):
  mode_model=highspy.Highs();mode_model.setOptionValue('threads',1);mode_model.setOptionValue('output_flag',False);mode_vars={i:[(mode_model.addVariable(lb=0,ub=1,obj=m['duration'],type=highspy.HighsVarType.kInteger),m) for m in allowed[i]] for i in by}
  for row in mode_vars.values():mode_model.addConstr(sum(v for v,m in row)==1)
  for r in res['nonrenewable']:mode_model.addConstr(sum(m['nonrenewable_consumption'][r['resource_id']]*v for row in mode_vars.values() for v,m in row)<=r['capacity'])
  mode_model.setOptionValue('time_limit',max(.01,min(3.,deadline-time.monotonic())));mode_model.run()
  if mode_model.getSolution().value_valid:
   values=mode_model.getSolution().col_value;selected={i:next(m for v,m in row if values[int(v)]>.5) for i,row in mode_vars.items()}
 if all(sum(selected[i]['nonrenewable_consumption'][r['resource_id']] for i in by)<=r['capacity'] for r in res['nonrenewable']):
  calendar={r['resource_id']:[0]*(H+1) for r in res['renewable']};tail={};finish={};schedule=[]
  for i in reversed(order):tail[i]=selected[i]['duration']+max((tail[s] for s in by[i]['successors']),default=0)
  remaining=set(by)
  while remaining:
   ready=[i for i in remaining if all(k in finish for k in pred[i])];i=max(ready,key=lambda i:(tail[i],-i));mode=selected[i];duration=mode['duration'];t=max([release[by[i]['project_id']]]+[finish[k] for k in pred[i]])
   while not all(all(calendar[r['resource_id']][tau]+mode['renewable_consumption'][r['resource_id']]<=r['capacity'] for tau in range(t,t+duration)) for r in res['renewable']):t+=1
   for r in res['renewable']:
    for tau in range(t,t+duration):calendar[r['resource_id']][tau]+=mode['renewable_consumption'][r['resource_id']]
   finish[i]=t+duration;schedule.append({'job_id':i,'mode_id':mode['mode_id'],'start_time':t});remaining.remove(i)
  save(schedule)
 print('INCUMBENT_END elapsed',time.monotonic()-start,'objective',best,flush=True)
 if best==0:return
 earliest={i:release[by[i]['project_id']] for i in by};latest_finish={i:H for i in by}
 for i in order:
  for s in by[i]['successors']:earliest[s]=max(earliest[s],earliest[i]+min(m['duration'] for m in allowed[i]))
 if best_schedule is not None:
  for project in projects:latest_finish[project['artificial_sink_job_id']]=constants[project['project_id']]+int(best)
 for i in reversed(order):
  for s in by[i]['successors']:latest_finish[i]=min(latest_finish[i],latest_finish[s]-min(m['duration'] for m in allowed[s]))
 horizon=max(latest_finish.values());pulse=highspy.Highs();pulse.setOptionValue('threads',1);pulse.setOptionValue('output_flag',True);entries={};start_expr={};finish_expr={};sinks={p['artificial_sink_job_id'] for p in projects};warm={e['job_id']:(e['mode_id'],e['start_time']) for e in best_schedule or []};warm_indices=[];warm_values=[]
 print('BUILD_START elapsed',time.monotonic()-start,'horizon',horizon,'jobs',len(jobs),flush=True)
 for job in jobs:
  if time.monotonic()>=deadline:
   print('BUILD_DEADLINE elapsed',time.monotonic()-start,flush=True);return
  jid=job['job_id'];entries[jid]=[]
  for mode in allowed[jid]:
   for t in range(earliest[jid],latest_finish[jid]-mode['duration']+1):
    v=pulse.addVariable(lb=0,ub=1,obj=t if jid in sinks else 0,type=highspy.HighsVarType.kInteger);entries[jid].append((v,mode,t))
    if warm.get(jid)==(mode['mode_id'],t):warm_indices.append(int(v));warm_values.append(1.)
  pulse.addConstr(sum(v for v,mode,t in entries[jid])==1);start_expr[jid]=sum(t*v for v,mode,t in entries[jid]);finish_expr[jid]=sum((t+mode['duration'])*v for v,mode,t in entries[jid])
 for job in jobs:
  for successor in job['successors']:pulse.addConstr(finish_expr[job['job_id']]<=start_expr[successor])
 for r in res['nonrenewable']:pulse.addConstr(sum(mode['nonrenewable_consumption'][r['resource_id']]*v for row in entries.values() for v,mode,t in row)<=r['capacity'])
 for r in res['renewable']:
  rid=r['resource_id']
  for tau in range(horizon):
   if tau%16==0 and time.monotonic()>=deadline:
    print('RESOURCE_BUILD_DEADLINE elapsed',time.monotonic()-start,flush=True);return
   active=[mode['renewable_consumption'][rid]*v for row in entries.values() for v,mode,t in row if t<=tau<t+mode['duration'] and mode['renewable_consumption'][rid]]
   if active:pulse.addConstr(sum(active)<=r['capacity'])
 if warm_indices:pulse.setSolution(len(warm_indices),warm_indices,warm_values)
 remaining_time=deadline-time.monotonic();print('BUILD_END elapsed',time.monotonic()-start,'solve_budget',remaining_time,flush=True)
 if remaining_time<=0:return
 pulse.setOptionValue('time_limit',remaining_time);pulse.run();print('PULSE_STATUS',pulse.modelStatusToString(pulse.getModelStatus()),'OBJECTIVE',pulse.getObjectiveValue()-sum(constants.values()),'BOUND',pulse.getInfo().mip_dual_bound-sum(constants.values()),'elapsed',time.monotonic()-start,flush=True)
 if pulse.getSolution().value_valid:
  values=pulse.getSolution().col_value;save([{'job_id':jid,'mode_id':mode['mode_id'],'start_time':t} for jid,row in entries.items() for v,mode,t in row if values[int(v)]>.5])
if __name__=='__main__':main()

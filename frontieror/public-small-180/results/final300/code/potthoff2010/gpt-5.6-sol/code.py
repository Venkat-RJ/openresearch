import time
_wall_start=time.time()
import argparse,json,os
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
P=d['parameters'];res=d['disruption']['rescheduling_time_minutes'];bad=set(d['disruption'].get('canceled_tasks',[]));tasks={t['task_id']:t for t in d['tasks'] if t['departure_time_minutes']>=res and t['task_id'] not in bad};duties={u['duty_id']:u for u in d.get('active_duties',[])+d.get('reserve_duties',[])};stations={s['id'] for s in d['network']['stations']};taxi={}
for z in d['network'].get('taxi_connections',[]):
 taxi[z['from_station'],z['to_station']]=z['travel_time_minutes']
 if z.get('bidirectional'):taxi[z['to_station'],z['from_station']]=z['travel_time_minutes']
def minutes(s):h,m=s.split(':');return int(h)*60+int(m)
def completion(u,r):
 station=u.get('standby_station',u['crew_base']) if u['duty_type']=='reserve' else u['crew_base'];clock=max(res,u['start_time_minutes']) if u['duty_type']=='reserve' else u['start_time_minutes'];maxend=u['end_time_minutes']+P['max_overtime_minutes'];uses=False
 if u['duty_type']!='reserve':
  for z in u.get('task_details',[]):
   if minutes(z['departure_time'])>=res:break
   station=z['arrival_station'];clock=max(clock,minutes(z['arrival_time']))
 for pos,tid in enumerate(r):
  t=tasks[tid]
  if t['departure_station'] not in u.get('route_knowledge',[]) or t['arrival_station'] not in u.get('route_knowledge',[]) or (t.get('rolling_stock_type') and t['rolling_stock_type'] not in u.get('rolling_stock_licenses',[])):return
  if t['departure_station']==station:
   if clock+(P['min_connection_time_minutes'] if pos else 0)>t['departure_time_minutes']:return
  else:
   gap=taxi.get((station,t['departure_station']))
   if gap is None or clock+gap>t['departure_time_minutes']:return
   uses=True
  if t['arrival_time_minutes']>maxend:return
  station=t['arrival_station'];clock=t['arrival_time_minutes']
 release=False
 if station!=u['crew_base']:
  travel=taxi.get((station,u['crew_base']))
  if travel is not None and clock+travel<=maxend:uses=True;clock+=travel;station=u['crew_base']
  elif station in stations:release=True
  else:return
 original=[q for q in u.get('tasks',[]) if q in tasks]
 cost=0 if r==original and not uses and not release else int(P['cost_duty_changed'])+int(P['cost_task_reassigned'])*sum(q not in original for q in r)+int(P['cost_new_transfer'])*sum(pair not in set(zip(original,original[1:])) for pair in zip(r,r[1:]))+(int(P['cost_taxi']) if uses else 0)
 return {'covered_tasks':r[:],'uses_taxi':uses,'release_at_relief_point':release,'return_station':station,'return_time_minutes':clock,'cost':cost}
lib={}
for k,u in sorted(duties.items()):
 routes=[[],[t for t in u.get('tasks',[]) if t in tasks]]+[[t] for t in sorted(tasks)];options=[];seen=set()
 for r in routes:
  if tuple(r) in seen:continue
  q=completion(u,r)
  if q is not None:seen.add(tuple(r));q['completion_index']=len(options);options.append(q)
 lib[k]=options
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
x=m.addVars([(k,i) for k,opts in lib.items() for i in range(len(opts))],vtype=GRB.BINARY);cancel=m.addVars(list(tasks),vtype=GRB.BINARY);pen={t:P['cost_cancel_A_to_A'] if q['task_type']=='A_to_A' else P['cost_cancel_A_to_B'] for t,q in tasks.items()}
for k,opts in lib.items():m.addConstr(gp.quicksum(x[k,i] for i in range(len(opts)))==1)
for t in tasks:m.addConstr(gp.quicksum(x[k,i] for k,opts in lib.items() for i,q in enumerate(opts) if t in q['covered_tasks'])+cancel[t]>=1)
m.setObjective(gp.quicksum(q['cost']*x[k,i] for k,opts in lib.items() for i,q in enumerate(opts))+gp.quicksum(pen[t]*cancel[t] for t in tasks))
for k,opts in lib.items():
 for i in range(len(opts)):x[k,i].Start=int(i==0)
for t in tasks:cancel[t].Start=1
def pack(v):
 ass={k:dict(opts[next(i for i in range(len(opts)) if v(x[k,i])>.5)]) for k,opts in lib.items()};canceled=sorted(t for t in tasks if v(cancel[t])>.5)
 return {'released_problem_variant':'relief_point_completion_v1','objective_value':sum(q['cost'] for q in ass.values())+sum(pen[t] for t in canceled),'canceled_tasks':canceled,'duty_assignments':ass}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible crew completion cover')
s=pack(lambda q:q.X)
if log:log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

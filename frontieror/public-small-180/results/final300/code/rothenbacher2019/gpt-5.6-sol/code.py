import time
_wall_start=time.time()
import argparse,json,os,math,itertools
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
customers={q['id']:q for q in d['customers']};depot=d['depot'];base=depot['id'];nodes={base:depot,**customers};distance={(i,j):math.hypot(q['x']-r['x'],q['y']-r['y']) for i,q in nodes.items() for j,r in nodes.items()};pool={};Q=d['vehicle_capacity'];duration=d['max_route_duration'];days=range(d['num_days']);enumend=start+min(20,a.time_limit*.4)
for day in days:
 enumend=time.monotonic()+min(20,a.time_limit*.4)/len(days)
 opts={}
 for i,c in customers.items():
  ps={}
  for sc in c['schedules']:
   for part in sc['parts']:
    if part['visit_day']==day:ps[part['days_covered']]=part
  if ps:opts[i]=list(ps.values())
 def extend(seq,parts,load,et,ed,eh,cost,only=False):
  if time.monotonic()>enumend:return
  prev=seq[-1] if seq else base
  if seq:
   tt=distance[prev,base]+nodes[prev]['service_time'];dt=max(ed+tt,eh+depot['time_window'][0]);ct=cost+distance[prev,base]
   if et+tt<=depot['time_window'][1] and dt<=duration:
    key=(day,tuple(sorted((i,parts[i]['days_covered']) for i in seq)));r={'customers':seq[:],'cost':ct,'demand':load,'parts':{str(i):dict(parts[i]) for i in seq}}
    if key not in pool or ct<pool[key]['cost']:pool[key]=r
  if only or sum(key[0]==day for key in pool)>1500:return
  for i,variants in opts.items():
   if i in parts:continue
   tt=distance[prev,i]+nodes[prev]['service_time'];e,l=customers[i]['time_window'];nt=max(e,et+tt);nd=max(ed+tt,eh+e);nh=max(ed+tt-l,eh)
   if nt>l or nd>duration:continue
   for part in variants:
    nl=load+part['demand_per_visit']
    if nl>Q:continue
    parts[i]=part;extend(seq+[i],parts,nl,nt,nd,nh,cost+distance[prev,i]);del parts[i]
 for i,variants in opts.items():
  tt=distance[base,i]+depot['service_time'];e,l=customers[i]['time_window'];nt=max(e,depot['time_window'][0]+tt);nd=max(tt,-depot['time_window'][1]+e);nh=max(tt-l,-depot['time_window'][1])
  for part in variants:
   if nt<=l and part['demand_per_visit']<=Q:extend([i],{i:part},part['demand_per_visit'],nt,nd,nh,distance[base,i],True)
 extend([],{},0,depot['time_window'][0],0,-depot['time_window'][1],0)
keys=list(pool)
if len(keys)>1800:
 singles=[k for k in keys if len(k[1])==1];others=[k for k in keys if len(k[1])!=1];others.sort(key=lambda k:pool[k]['cost']/len(k[1]));keys=singles+others[:max(0,1800-len(singles))]
R=range(len(keys));m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
x=m.addVars(R,vtype=GRB.BINARY);z=m.addVars([(i,k) for i,c in customers.items() for k in range(len(c['schedules']))],vtype=GRB.BINARY)
for i,c in customers.items():
 m.addConstr(gp.quicksum(z[i,k] for k in range(len(c['schedules'])))==1)
 pairs={(part['visit_day'],part['days_covered']) for sc in c['schedules'] for part in sc['parts']}
 for day,l in pairs:
  left=gp.quicksum(x[r] for r,key in enumerate(keys) if key[0]==day and (i,l) in key[1]);right=gp.quicksum(z[i,k] for k,sc in enumerate(c['schedules']) if any(pt['visit_day']==day and pt['days_covered']==l for pt in sc['parts']));m.addConstr(left==right)
for day in days:m.addConstr(gp.quicksum(x[r] for r,key in enumerate(keys) if key[0]==day)<=d['num_vehicles'])
m.setObjective(gp.quicksum(pool[key]['cost']*x[r] for r,key in enumerate(keys)))
def pack(v):
 routes={str(day):[pool[key] for r,key in enumerate(keys) if key[0]==day and v(x[r])>.5] for day in days};schedules={}
 for i,c in customers.items():
  k=next(k for k in range(len(c['schedules'])) if v(z[i,k])>.5);sc=c['schedules'][k];schedules[str(i)]={'schedule_index':k,'days':sc['days'],'visit_frequency':sc['visit_frequency']}
 return {'objective_value':sum(r['cost'] for rr in routes.values() for r in rr),'routes':routes,'schedules':schedules}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible periodic route pool incumbent')
s=pack(lambda q:q.X)
if log:log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

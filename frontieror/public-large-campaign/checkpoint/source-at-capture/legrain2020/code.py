import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));N=d['num_nurses'];D=d['num_days'];shifts=[s['name'] for s in d['shifts']];skills=d['skills'];model=cp_model.CpModel();x={};y={};work={};cost=[];seq=0
def positive(expr,weight,bound=N*D):
 global seq
 if not weight:return
 v=model.new_int_var(0,max(bound,1),'pen'+str(seq));seq+=1;model.add_max_equality(v,[expr,0]);cost.append(weight*v)
def AND(vs):
 global seq
 v=model.new_bool_var('and'+str(seq));seq+=1;model.add_min_equality(v,vs);return v
pw=d['penalty_weights'];nurse_cost={}
for nurse in d['nurses']:
 i=nurse['id'];first_cost=len(cost)
 for t in range(D):
  work[i,t]=model.new_bool_var(f'w{i}_{t}')
  for sn in shifts:
   x[i,t,sn]=model.new_bool_var(f'x{i}_{t}_{sn}');vs=[]
   for sk in nurse['skills']:v=model.new_bool_var(f'y{i}_{t}_{sn}_{sk}');y[i,t,sn,sk]=v;vs.append(v)
   model.add(sum(vs)==x[i,t,sn])
  model.add(sum(x[i,t,sn] for sn in shifts)==work[i,t])
 for t in range(D-1):
  for sn,sn2 in d['forbidden_shift_successions']:model.add(x[i,t,sn]+x[i,t+1,sn2]<=1)
 c=nurse['contract'];ww=[work[i,t] for t in range(D)]
 for arr,mn,mx,weight in [(ww,c['CD_minus'],c['CD_plus'],pw['c_S2a']),([1-v for v in ww],c['CR_minus'],c['CR_plus'],pw['c_S3'])]+[([x[i,t,s['name']] for t in range(D)],s['CS_minus'],s['CS_plus'],pw['c_S2b']) for s in d['shifts']]:
  for t in range(D-mx):positive(sum(arr[t:t+mx+1])-mx,weight)
  for t in range(D):
   start=arr[t] if not t else AND([arr[t],1-arr[t-1]])
   for j in range(1,mn):
    if t+j<D:positive(start-arr[t+j],weight)
 for pref in nurse.get('preferences',[]):cost.append(pw['c_S4']*x[i,pref['day'],pref['shift']])
 weekends=[]
 for t in range(5,D,7):
  if t+1>=D:continue
  v=model.new_int_var(0,1,f'we{i}_{t}');model.add_max_equality(v,[ww[t],ww[t+1]]);weekends.append(v);z=model.new_int_var(0,1,f'diff{i}_{t}');model.add_abs_equality(z,ww[t]-ww[t+1]);cost.append(pw['c_S5']*z)
 positive(c['L_minus']-sum(ww),pw['c_S6']);positive(sum(ww)-c['L_plus'],pw['c_S6']);positive(sum(weekends)-c['WE_plus'],pw['c_S7'])
 nurse_cost[i]=cost[first_cost:]
for t in range(D):
 for sn in shifts:
  for sk in skills:
   total=sum(y.get((i,t,sn,sk),0) for i in range(N));req=d['demand'][t][sn][sk];model.add(total>=req['minimum']);positive(req['optimal']-total,pw['c_S1'])
model.minimize(sum(cost));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.random_seed=0;solver.parameters.max_time_in_seconds=max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
class CB(cp_model.CpSolverSolutionCallback):
 def on_solution_callback(self):
  schedule=[]
  for nurse in d['nurses']:
   i=nurse['id'];assign=[{'day':t,'shift':sn,'skill':sk} for (ii,t,sn,sk),v in y.items() if ii==i and self.value(v)];schedule.append({'nurse_id':i,'nurse_name':nurse['name'],'assignments':assign})
  counts={key:0 for key in pw}
  for nurse in d['nurses']:
   i=nurse['id'];ww=[self.value(work[i,t]) for t in range(D)];co=nurse['contract']
   arrays=[(ww,co['CD_minus'],co['CD_plus'],'c_S2a'),([1-v for v in ww],co['CR_minus'],co['CR_plus'],'c_S3')]+[([self.value(x[i,t,sn['name']]) for t in range(D)],sn['CS_minus'],sn['CS_plus'],'c_S2b') for sn in d['shifts']]
   for arr,mn,mx,key in arrays:
    counts[key]+=sum(max(0,sum(arr[t:t+mx+1])-mx) for t in range(D-mx))
    for t in range(D):
     indicator=arr[t] if t==0 else max(0,arr[t]-arr[t-1])
     counts[key]+=sum(max(0,indicator-arr[t+j]) for j in range(1,mn) if t+j<D)
   counts['c_S4']+=sum(self.value(x[i,pr['day'],pr['shift']]) for pr in nurse.get('preferences',[]))
   counts['c_S5']+=sum(abs(ww[t]-ww[t+1]) for t in range(5,D-1,7))
   counts['c_S6']+=max(0,co['L_minus']-sum(ww))+max(0,sum(ww)-co['L_plus'])
   counts['c_S7']+=max(0,sum(max(ww[t],ww[t+1]) for t in range(5,D-1,7))-co['WE_plus'])
  coverage={}
  for (i,t,sn,sk),v in y.items():
   if self.value(v):coverage[t,sn,sk]=coverage.get((t,sn,sk),0)+1
  for t in range(D):
   for sn in shifts:
    for sk in skills:counts['c_S1']+=max(0,d['demand'][t][sn][sk]['optimal']-coverage.get((t,sn,sk),0))
  component_names=['S1_optimal_coverage','S2a_consecutive_days','S2b_consecutive_shifts','S3_consecutive_rest','S4_preferences','S5_complete_weekend','S6_total_assignments','S7_working_weekends'];keys=['c_S1','c_S2a','c_S2b','c_S3','c_S4','c_S5','c_S6','c_S7'];breakdown={name:counts[key]*pw[key] for name,key in zip(component_names,keys)}
  s={'objective_value':sum(breakdown.values()),'schedule':schedule,'penalty_breakdown':breakdown};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(s['objective_value'],s)
import random
started=time.monotonic();deadline=started+max(.1,a.time_limit-(6 if N>40 or D>28 else 2));best=float('inf');incumbent=None;badness={}
base_callback=CB.on_solution_callback
def capture(self):
 global best,incumbent
 if self.objective_value < best:
  best=self.objective_value
  badness.update({i:self.value(sum(terms)) for i,terms in nurse_cost.items()})
  incumbent=[self.value(model.get_int_var_from_proto_index(j)) for j in range(len(model.proto.variables))]
  base_callback(self)
CB.on_solution_callback=capture
def constructive():
 from ortools.graph.python import min_cost_flow
 slots=[(sn,sk) for sn in shifts for sk in skills];forbidden=set(map(tuple,d['forbidden_shift_successions']));shift_specs={s['name']:s for s in d['shifts']};previous={n['id']:None for n in d['nurses']};wr={i:0 for i in previous};sr=wr.copy();totals=wr.copy();weekend_counts=wr.copy();assignments={}
 for t in range(D):
  flow=min_cost_flow.SimpleMinCostFlow();source=0;sink=1+N+len(slots);arcs={};required=[]
  for nurse in d['nurses']:
   i=nurse['id'];co=nurse['contract'];prev=previous[i];working=prev is not None
   offcost=(pw['c_S2a']*max(0,co['CD_minus']-wr[i]) if working else pw['c_S3']*int(wr[i]>=co['CR_plus']))
   if working:offcost+=pw['c_S2b']*max(0,shift_specs[prev]['CS_minus']-sr[i])
   if t%7==6 and working:offcost+=pw['c_S5']
   if D-t<=co['L_minus']-totals[i]:offcost+=pw['c_S6']
   flow.add_arc_with_capacity_and_unit_cost(source,1+i,1,0);arcs[i,None]=flow.add_arc_with_capacity_and_unit_cost(1+i,sink,1,int(offcost))
   prefs={pr['shift'] for pr in nurse.get('preferences',[]) if pr['day']==t}
   for j,(sn,sk) in enumerate(slots):
    if sk not in nurse['skills'] or (prev,sn) in forbidden:continue
    value=pw['c_S4']*int(sn in prefs)+pw['c_S6']*int(totals[i]>=co['L_plus'])
    value+=pw['c_S2a']*int(working and wr[i]>=co['CD_plus'])
    if not working:value+=pw['c_S3']*max(0,co['CR_minus']-wr[i]) if t else 0
    if sn==prev:value+=pw['c_S2b']*int(sr[i]>=shift_specs[sn]['CS_plus'])
    elif working:value+=pw['c_S2b']*max(0,shift_specs[prev]['CS_minus']-sr[i])
    if t%7==6 and not working:value+=pw['c_S5']
    if t%7 in (5,6) and (t%7==5 or not working):value+=pw['c_S7']*int(weekend_counts[i]>=co['WE_plus'])
    arcs[i,(sn,sk)]=flow.add_arc_with_capacity_and_unit_cost(1+i,1+N+j,1,int(value))
  for j,(sn,sk) in enumerate(slots):
   req=d['demand'][t][sn][sk];arc=flow.add_arc_with_capacity_and_unit_cost(1+N+j,sink,req['minimum'],-1000000);required.append((arc,req['minimum']))
   flow.add_arc_with_capacity_and_unit_cost(1+N+j,sink,max(0,req['optimal']-req['minimum']),-pw['c_S1'])
  flow.set_node_supply(source,N);flow.set_node_supply(sink,-N)
  if flow.solve()!=flow.OPTIMAL or any(flow.flow(arc)!=minimum for arc,minimum in required):return None
  for nurse in d['nurses']:
   i=nurse['id'];chosen=next(option for (ii,option),arc in arcs.items() if ii==i and flow.flow(arc));assignments[i,t]=chosen;sn=chosen[0] if chosen is not None else None;prev=previous[i]
   if sn is not None:
    totals[i]+=1
    if t%7==5 or (t%7==6 and prev is None):weekend_counts[i]+=int(t%7 in (5,6))
   wr[i]=wr[i]+1 if (sn is None)==(prev is None) else 1;sr[i]=sr[i]+1 if sn==prev else 1;previous[i]=sn
 return assignments
def polish(seed,until):
 from ortools.graph.python import min_cost_flow
 rows={n['id']:[seed[n['id'],t] for t in range(D)] for n in d['nurses']};slots=[(sn,sk) for sn in shifts for sk in skills];specs={s['name']:s for s in d['shifts']};forbidden=set(map(tuple,d['forbidden_shift_successions']));prefs={n['id']:[(pr['day'],pr['shift']) for pr in n.get('preferences',[])] for n in d['nurses']}
 def sequence(arr,mn,mx):
  total=0;run=0
  for t,value in enumerate(arr):
   if value:
    if not run:total+=sum(1-arr[j] for j in range(t+1,min(D,t+mn)))
    run+=1;total+=int(run>mx)
   else:run=0
  return total
 def penalty(nurse,row):
  i=nurse['id'];co=nurse['contract'];names=[v[0] if v is not None else None for v in row];ww=[int(v is not None) for v in row]
  value=pw['c_S2a']*sequence(ww,co['CD_minus'],co['CD_plus'])+pw['c_S3']*sequence([1-v for v in ww],co['CR_minus'],co['CR_plus'])
  for sn in shifts:value+=pw['c_S2b']*sequence([int(v==sn) for v in names],specs[sn]['CS_minus'],specs[sn]['CS_plus'])
  value+=pw['c_S4']*sum(names[t]==sn for t,sn in prefs[i]);value+=pw['c_S5']*sum(abs(ww[t]-ww[t+1]) for t in range(5,D-1,7));total=sum(ww)
  value+=pw['c_S6']*(max(0,co['L_minus']-total)+max(0,total-co['L_plus']))
  value+=pw['c_S7']*max(0,sum(max(ww[t],ww[t+1]) for t in range(5,D-1,7))-co['WE_plus'])
  return int(value)
 sweep=0;order=list(range(D))
 while time.monotonic()<until:
  sweep+=1
  for t in order if sweep%2 else reversed(order):
   if time.monotonic()>=until:break
   flow=min_cost_flow.SimpleMinCostFlow();sink=1+N+len(slots);arcs={};required=[]
   for nurse in d['nurses']:
    i=nurse['id'];row=rows[i];previous=row[t-1][0] if t and row[t-1] is not None else None;following=row[t+1][0] if t+1<D and row[t+1] is not None else None
    flow.add_arc_with_capacity_and_unit_cost(0,1+i,1,0);trial=row.copy();trial[t]=None;arcs[i,None]=flow.add_arc_with_capacity_and_unit_cost(1+i,sink,1,penalty(nurse,trial))
    scores={}
    for j,(sn,sk) in enumerate(slots):
     if sk not in nurse['skills'] or (previous,sn) in forbidden or (sn,following) in forbidden:continue
     if sn not in scores:trial[t]=(sn,sk);scores[sn]=penalty(nurse,trial)
     arcs[i,(sn,sk)]=flow.add_arc_with_capacity_and_unit_cost(1+i,1+N+j,1,scores[sn])
   for j,(sn,sk) in enumerate(slots):
    req=d['demand'][t][sn][sk];arc=flow.add_arc_with_capacity_and_unit_cost(1+N+j,sink,req['minimum'],-1000000);required.append((arc,req['minimum']))
    flow.add_arc_with_capacity_and_unit_cost(1+N+j,sink,max(0,req['optimal']-req['minimum']),-pw['c_S1'])
   flow.set_node_supply(0,N);flow.set_node_supply(sink,-N)
   if flow.solve()!=flow.OPTIMAL or any(flow.flow(arc)!=minimum for arc,minimum in required):continue
   for (i,option),arc in arcs.items():
    if flow.flow(arc):rows[i][t]=option
 return {(i,t):row[t] for i,row in rows.items() for t in range(D)}
large=N>40 or D>28
if large:
 seed=constructive()
 if seed is not None:
  seed=polish(seed,min(deadline-8,time.monotonic()+25))
  fixed=model.clone()
  for (i,t,sn,sk),v in y.items():fixed.add(fixed.get_int_var_from_proto_index(v.index)==int(seed[i,t]==(sn,sk)))
  solver.parameters.max_time_in_seconds=max(.1,min(6,deadline-time.monotonic()));solver.solve(fixed,CB())
if incumbent is None:
 solver.parameters.max_time_in_seconds=max(.1,deadline-time.monotonic() if large else min(8,(deadline-started)*.2))
 solver.solve(model,CB())
rng=random.Random(0);iteration=0
while incumbent is not None and time.monotonic()<deadline-.15:
 neighborhood=model.clone();iteration+=1
 if iteration%3:
  ids=[n['id'] for n in d['nurses']];free=set()
  while len(free)<min(N,4+iteration%3):
   available=[i for i in ids if i not in free]
   free.add(rng.choices(available,weights=[30+badness[i] for i in available])[0])
  unfixed=lambda i,t:i in free
 else:
  length=min(D,3);first=rng.randrange(D-length+1)
  unfixed=lambda i,t:first<=t<first+length
 for (i,t,sn,sk),v in y.items():
  if not unfixed(i,t):neighborhood.add(neighborhood.get_int_var_from_proto_index(v.index)==incumbent[v.index])
 neighborhood.add(sum(cost)<=int(best)-1)
 neighborhood.clear_hints()
 for j,value in enumerate(incumbent):neighborhood.add_hint(neighborhood.get_int_var_from_proto_index(j),value)
 solver.parameters.max_time_in_seconds=max(.05,min(1.5,deadline-time.monotonic()))
 solver.parameters.random_seed=iteration
 solver.solve(neighborhood,CB())

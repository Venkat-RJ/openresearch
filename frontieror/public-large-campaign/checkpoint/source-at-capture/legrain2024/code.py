import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
started=time.monotonic()
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));D=d['instance_metadata']['num_days'];shifts=d['shifts'];skills=d['skills'];nurses=d['nurses'];hard=d['constraints']['hard_constraints'];soft=d['constraints']['soft_constraints'];model=cp_model.CpModel();x={};y={};work={};cost=[];seq=0

def AND(vs):
 global seq
 z=model.new_bool_var('a'+str(seq));seq+=1;model.add_min_equality(z,vs);return z
def positive(expr,weight):
 global seq
 if not weight:return
 z=model.new_int_var(0,100000,'p'+str(seq));seq+=1;model.add_max_equality(z,[expr,0]);cost.append(int(weight)*z)
def bound(expr,limit,weight,is_hard):
 if is_hard:model.add(expr<=limit)
 else:positive(expr-limit,weight)
for nurse in nurses:
 i=nurse['id'];allowed=nurse['skills'] if hard.get('skill_requirement',False) else skills
 for t in range(1,D+1):
  vs=[]
  for sn in shifts+['None']:x[i,t,sn]=model.new_bool_var(f'x{i}_{t}_{sn}');vs.append(x[i,t,sn])
  model.add(sum(vs)==1);work[i,t]=1-x[i,t,'None']
  for sn in shifts:
   vs=[]
   for sk in allowed:
    v=model.new_bool_var(f'y{i}_{t}_{sn}_{sk}');y[i,t,sn,sk]=v;vs.append(v)
    if sk not in nurse['skills']:cost.append(int(soft.get('skill_requirement_penalty',0))*v)
   model.add(sum(vs)==x[i,t,sn])
 ww=[work[i,t] for t in range(1,D+1)];wkdays=[[wk[k] for k in ('friday','saturday','sunday') if k in wk] for wk in d['weekends']];we=[]
 for days in wkdays:
  z=model.new_bool_var('we'+str(seq));seq+=1;model.add_max_equality(z,[work[i,t] for t in days]);we.append(z)
 for rules,is_hard in [(hard,True),(soft,False)]:
  if 'total_working_days' in rules:
   q=rules['total_working_days'];bound(sum(ww),q.get('max',D),q.get('penalty_over',0),is_hard);bound(-sum(ww),-q.get('min',0),q.get('penalty_under',0),is_hard)
  if 'total_worked_weekends' in rules:
   q=rules['total_worked_weekends'];bound(sum(we),q['max'],q.get('penalty',0),is_hard)
  for name,arr in [('consecutive_working_days',ww),('consecutive_days_off',[1-z for z in ww]),('consecutive_worked_weekends',we)]+[('consecutive_same_shift',[x[i,t,sn] for t in range(1,D+1)]) for sn in shifts]:
   if name not in rules:continue
   q=rules[name];N=len(arr);mx=q.get('max',N);mn=q.get('min',0);pen=q.get('penalty_over',q.get('penalty',0))
   for t in range(N-mx):bound(sum(arr[t:t+mx+1]),mx,pen,is_hard)
   if name in ('consecutive_working_days','consecutive_days_off'):
    for size in range(1,mn):
     for t in range(1,N-size):
      vs=[1-arr[t-1]]+arr[t:t+size]+[1-arr[t+size]]
      if is_hard:model.add(sum(vs)<=len(vs)-1)
      else:cost.append(int(q.get('penalty_under',0))*AND(vs))
  if rules.get('identical_weekend',{}).get('enabled',False):
   for days in wkdays:
    equal=[AND([x[i,t,sn] for t in days]) for sn in shifts+['None']]
    if is_hard:model.add(sum(equal)==1)
    else:cost.append(int(rules['identical_weekend'].get('penalty',0))*(1-sum(equal)))
  for fp in rules.get('forbidden_patterns',[]):
   pattern=fp['pattern']
   for t in range(1,D-len(pattern)+2):
    vs=[x[i,t+j,sn] for j,(_,sn) in enumerate(pattern)]
    if is_hard:model.add(sum(vs)<=len(vs)-1)
    else:cost.append(int(fp.get('penalty',0))*AND(vs))
 for pref in nurse.get('preferences',[]):
  t=pref['day'];sn=pref.get('shift','None');kind=pref['type']
  if kind=='day_off':violation=work[i,t]
  elif kind=='day_on':violation=x[i,t,'None']
  elif kind=='shift_on':violation=1-x[i,t,sn]
  else:violation=x[i,t,sn]
  cost.append(int(pref.get('penalty',0))*violation)
  if 'preferences' in hard or hard.get('hard_preferences',False):model.add(violation==0)
for t in range(1,D+1):
 for sn,reqs in d['demand'].get(f'day_{t}',{}).items():
  for sk,req in reqs.items():
   total=sum(y.get((nu['id'],t,sn,sk),0) for nu in nurses)
   if hard.get('demand_equality',False):model.add(total==req)
   elif hard.get('demand_lower_bound',False):model.add(total>=req)
   positive(req-total,d['unsatisfied_demand_penalty'])
model.minimize(sum(cost));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.random_seed=0;deadline=started+max(.1,a.time_limit-5);solver.parameters.max_time_in_seconds=max(.1,deadline-time.monotonic());logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
best=float('inf');incumbent=None
class CB(cp_model.CpSolverSolutionCallback):
 def on_solution_callback(self):
  global best,incumbent
  if self.objective_value>=best:return
  best=self.objective_value;incumbent=[self.value(model.get_int_var_from_proto_index(j)) for j in range(len(model.proto.variables))]
  schedule={};skill={}
  for nurse in nurses:
   i=nurse['id'];schedule[i]={};skill[i]={}
   for t in range(1,D+1):
    sn=next(sn for sn in shifts+['None'] if self.value(x[i,t,sn]));sk='None' if sn=='None' else next(sk for sk in skills if (i,t,sn,sk) in y and self.value(y[i,t,sn,sk]));schedule[i][f'day_{t}']=sn;skill[i][f'day_{t}']=sk
  s={'objective_value':self.objective_value,'schedule':schedule,'skill_assignments':skill};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(s['objective_value'],s)
optimization_rules=dict(soft);hard_roster=False
for name in ('total_working_days','total_worked_weekends','consecutive_working_days','consecutive_days_off','consecutive_same_shift','consecutive_worked_weekends','identical_weekend'):
 if name in hard:
  hard_roster=True;optimization_rules[name]={**hard[name],'penalty':1000000,'penalty_under':1000000,'penalty_over':1000000}
def roster_cost(nurse,row,focus=None):
 total=0;shift_bits={sn:0 for sn in shifts+['None']}
 for t,sn in enumerate(row):shift_bits[sn]|=1<<t
 work_bits=((1<<D)-1)^shift_bits['None'];working=work_bits.bit_count()
 weekends=[[row[t-1] for t in days] for days in wkdays]
 weekend_bits=sum(1<<k for k,days in enumerate(weekends) if any(sn!='None' for sn in days));worked_weekends=weekend_bits.bit_count()
 q=optimization_rules.get('total_working_days',{});total+=max(0,q.get('min',0)-working)*q.get('penalty_under',0)+max(0,working-q.get('max',D))*q.get('penalty_over',0)
 q=optimization_rules.get('total_worked_weekends',{});total+=max(0,worked_weekends-q.get('max',len(weekends)))*q.get('penalty',0)
 for name,bits,N in [('consecutive_working_days',work_bits,D),('consecutive_days_off',shift_bits['None'],D),('consecutive_worked_weekends',weekend_bits,len(weekends))]+[('consecutive_same_shift',shift_bits[sn],D) for sn in shifts]:
  q=optimization_rules.get(name,{});mx=q.get('max',N);mn=q.get('min',0)
  if mx<N:
   hits=bits
   for j in range(1,mx+1):hits&=bits>>j
   if focus is not None and name!='consecutive_worked_weekends':
    lo=max(0,focus-mx);hi=min(focus+1,N-mx);hits&=((1<<hi)-1)^((1<<lo)-1) if hi>lo else 0
   total+=hits.bit_count()*q.get('penalty_over',q.get('penalty',0))
  if name in ('consecutive_working_days','consecutive_days_off'):
   for size in range(1,min(mn,N)):
    hits=bits
    for j in range(1,size):hits&=bits>>j
    hits&=~(bits<<1)&~(bits>>size)
    lo=1 if focus is None else max(1,focus-size);hi=N-size if focus is None else min(focus+2,N-size)
    hits&=((1<<hi)-1)^((1<<lo)-1) if hi>lo else 0
    total+=hits.bit_count()*q.get('penalty_under',0)
 q=optimization_rules.get('identical_weekend',{})
 if q.get('enabled',False):total+=sum(int(len(set(days))>1) for days in weekends)*q.get('penalty',0)
 for fp in soft.get('forbidden_patterns',[]):
  pattern=[sn for _,sn in fp['pattern']];total+=sum(int(row[t:t+len(pattern)]==pattern) for t in range(D-len(pattern)+1))*fp.get('penalty',0)
 for pref in nurse.get('preferences',[]):
  sn=row[pref['day']-1];kind=pref['type'];wanted=pref.get('shift','None')
  violation=(sn!='None') if kind=='day_off' else (sn=='None') if kind=='day_on' else (sn!=wanted) if kind=='shift_on' else (sn==wanted)
  total+=int(violation)*pref.get('penalty',0)
 return total

def seed_rosters(until):
 from ortools.graph.python import min_cost_flow
 rows={};assigned={};N=len(nurses);iteration=0
 weekend_days={t-1 for days in wkdays for t in days};weekend_start=min(t%7 for t in weekend_days)
 for j,nu in enumerate(nurses):
  row=[];shift=shifts[j%len(shifts)]
  for t in range(D):
   week=t//7;weekday=t%7;working_weekend=(week+j//len(shifts))%2==0
   off=(t in weekend_days and not working_weekend) or (t not in weekend_days and weekday==(weekend_start-2+(j//2)%2 if working_weekend else (j//4)%2))
   if off:row.append('None')
   else:row.append(shift)
  working=sum(sn!='None' for sn in row);maximum=min(soft.get('total_working_days',{}).get('max',D),hard.get('total_working_days',{}).get('max',D))
  for t in range(D):
   if working<=maximum:break
   if t not in weekend_days and row[t]!='None' and (t==0 or row[t-1]=='None') and t+2<D and row[t+1]!='None' and row[t+2]!='None':row[t]='None';working-=1
  rows[nu['id']]=row
  for t,sn in enumerate(row):assigned[nu['id'],t+1]='None' if sn=='None' else (nu['skills'][0] if hard.get('skill_requirement',False) else skills[0])
 while time.monotonic()<until:
  for t in (range(D) if iteration%2==0 else range(D-1,-1,-1)):
   if time.monotonic()>=until:return rows,assigned
   slots=[(sn,sk,int(req)) for sn,reqs in d['demand'].get(f'day_{t+1}',{}).items() for sk,req in reqs.items()]
   source=0;first=N+1;off=first+len(slots);sink=off+1;flow=min_cost_flow.SimpleMinCostFlow();arcs=[]
   for j,nu in enumerate(nurses):
    i=nu['id'];row=rows[i];old=row[t];base=roster_cost(nu,row,t);flow.add_arc_with_capacity_and_unit_cost(source,j+1,1,0)
    prices={}
    for sn in shifts+['None']:
     row[t]=sn
     valid=True
     for fp in hard.get('forbidden_patterns',[]):
      pattern=[v for _,v in fp['pattern']];length=len(pattern)
      if any(row[start:start+length]==pattern for start in range(max(0,t-length+1),min(t,D-length)+1)):valid=False;break
     if valid:prices[sn]=roster_cost(nu,row,t)-base
    row[t]=old
    if 'None' in prices:arcs.append((flow.add_arc_with_capacity_and_unit_cost(j+1,off,1,int(prices['None'])),i,'None','None'))
    for k,(sn,sk,req) in enumerate(slots):
     if sn not in prices or (hard.get('skill_requirement',False) and sk not in nu['skills']):continue
     price=prices[sn]+(soft.get('skill_requirement_penalty',0) if sk not in nu['skills'] else 0)
     arcs.append((flow.add_arc_with_capacity_and_unit_cost(j+1,first+k,1,int(price)),i,sn,sk))
   flow.add_arc_with_capacity_and_unit_cost(off,sink,N,0)
   for k,(sn,sk,req) in enumerate(slots):
    flow.add_arc_with_capacity_and_unit_cost(first+k,sink,req,-int(10000000 if hard.get('demand_lower_bound',False) else d['unsatisfied_demand_penalty']))
    if not hard.get('demand_equality',False):flow.add_arc_with_capacity_and_unit_cost(first+k,sink,N,0)
   flow.set_node_supply(source,N);flow.set_node_supply(sink,-N)
   if flow.solve()!=flow.OPTIMAL:return rows,assigned
   for arc,i,sn,sk in arcs:
    if flow.flow(arc):rows[i][t]=sn;assigned[i,t+1]=sk
  iteration+=1
 return rows,assigned

def exchange_blocks(rows,assigned,until):
 import random,math
 rng=random.Random(1);ids=[nu['id'] for nu in nurses];by_id={nu['id']:nu for nu in nurses}
 skill_weight=int(soft.get('skill_requirement_penalty',0));skill_cost={i:skill_weight*sum(rows[i][t-1]!='None' and assigned[i,t] not in by_id[i]['skills'] for t in range(1,D+1)) for i in ids}
 penalties={i:roster_cost(by_id[i],rows[i])+skill_cost[i] for i in ids};total=sum(penalties.values());lowest=total
 saved_rows={i:row[:] for i,row in rows.items()};saved_skills=dict(assigned);began=time.monotonic()
 patterns=[[sn for _,sn in fp['pattern']] for fp in hard.get('forbidden_patterns',[])]
 while time.monotonic()<until:
  left,right=rng.sample(ids,2);length=min(D,rng.choice((1,2,3,4,5,7,14)));first=rng.randrange(D-length+1);end=first+length
  arow,brow=rows[left],rows[right];old_a=arow[first:end];old_b=brow[first:end]
  if old_a==old_b and all(assigned[left,t]==assigned[right,t] for t in range(first+1,end+1)):continue
  if hard.get('skill_requirement',False) and any((brow[t]!='None' and assigned[right,t+1] not in by_id[left]['skills']) or (arow[t]!='None' and assigned[left,t+1] not in by_id[right]['skills']) for t in range(first,end)):continue
  arow[first:end],brow[first:end]=old_b,old_a
  valid=True
  for row in (arow,brow):
   for pattern in patterns:
    size=len(pattern)
    if any(row[t:t+size]==pattern for t in range(max(0,first-size+1),min(D-size+1,end))):valid=False;break
   if not valid:break
  if not valid:arow[first:end],brow[first:end]=old_a,old_b;continue
  delta_a=delta_b=0
  if skill_weight:
   delta_a=skill_weight*sum(int(old_b[t-first]!='None' and assigned[right,t+1] not in by_id[left]['skills'])-int(old_a[t-first]!='None' and assigned[left,t+1] not in by_id[left]['skills']) for t in range(first,end))
   delta_b=skill_weight*sum(int(old_a[t-first]!='None' and assigned[left,t+1] not in by_id[right]['skills'])-int(old_b[t-first]!='None' and assigned[right,t+1] not in by_id[right]['skills']) for t in range(first,end))
  ca=roster_cost(by_id[left],arow)+skill_cost[left]+delta_a;cb=roster_cost(by_id[right],brow)+skill_cost[right]+delta_b;change=ca+cb-penalties[left]-penalties[right]
  temperature=max(.1,8*(until-time.monotonic())/max(.1,until-began))
  if change<=0 or rng.random()<math.exp(-change/temperature):
   total+=change;penalties[left],penalties[right]=ca,cb;skill_cost[left]+=delta_a;skill_cost[right]+=delta_b
   for t in range(first+1,end+1):assigned[left,t],assigned[right,t]=assigned[right,t],assigned[left,t]
   if total<lowest:lowest=total;saved_rows={i:row[:] for i,row in rows.items()};saved_skills=dict(assigned)
  else:arow[first:end],brow[first:end]=old_a,old_b
 return saved_rows,saved_skills

loose_soft=not hard_roster and hard.get('skill_requirement',False) and max(sum(v for req in day.values() for v in req.values()) for day in d['demand'].values())<=.9*len(nurses)
if len(nurses)>40 or D>28:
 rows,assigned=seed_rosters(min(deadline-8,time.monotonic()+(12 if hard_roster else 10)))
 rows,assigned=exchange_blocks(rows,assigned,min(deadline-8,time.monotonic()+(20 if hard_roster or loose_soft else 32)))
 if hard_roster:
  for (i,t,sn),v in x.items():model.add_hint(v,int(rows[i][t-1]==sn))
  for (i,t,sn,sk),v in y.items():model.add_hint(v,int(rows[i][t-1]==sn and assigned.get((i,t))==sk))
  solver.parameters.repair_hint=True;solver.parameters.hint_conflict_limit=10000
 fixed=model.clone()
 for (i,t,sn),v in x.items():fixed.add(fixed.get_int_var_from_proto_index(v.index)==int(rows[i][t-1]==sn))
 for (i,t,sn,sk),v in y.items():fixed.add(fixed.get_int_var_from_proto_index(v.index)==int(rows[i][t-1]==sn and assigned.get((i,t))==sk))
 solver.parameters.max_time_in_seconds=max(.1,min(5,deadline-time.monotonic()));status=solver.solve(fixed,CB())
 if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
  model.add(sum(cost)<=int(round(solver.objective_value)));model.clear_hints()
  for j in range(len(model.proto.variables)):model.add_hint(model.get_int_var_from_proto_index(j),solver.value(fixed.get_int_var_from_proto_index(j)))
if incumbent is None or not (len(nurses)>40 or D>28):
 solver.parameters.max_time_in_seconds=max(.1,deadline-time.monotonic());solver.solve(model,CB())
else:
 import random
 rng=random.Random(0);iteration=0
 if loose_soft:solver.parameters.linearization_level=2;solver.parameters.search_branching=cp_model.LP_SEARCH
 while time.monotonic()<deadline-1:
  ids=[nu['id'] for nu in nurses];weights=[]
  for nu in nurses:
   row=[next(sn for sn in shifts+['None'] if incumbent[x[nu['id'],t,sn].index]) for t in range(1,D+1)]
   weights.append(30+roster_cost(nu,row))
  free=set();days=set()
  if loose_soft and iteration%2==0:
   anchor=rng.choices(ids,weights=weights,k=1)[0];free.add(anchor)
   anchor_skills=set(next(nu['skills'] for nu in nurses if nu['id']==anchor))
   partner_weights=[weight*(1+3*len(anchor_skills.intersection(nu['skills']))) for nu,weight in zip(nurses,weights)]
   while len(free)<min(8,len(ids)):free.add(rng.choices(ids,weights=partner_weights,k=1)[0])
  elif hard_roster or loose_soft:
   violations=[]
   for nu in nurses:
    i=nu['id']
    for pref in nu.get('preferences',[]):
     t=pref['day'];sn=next(sn for sn in shifts+['None'] if incumbent[x[i,t,sn].index]);kind=pref['type'];wanted=pref.get('shift','None')
     violation=(sn!='None') if kind=='day_off' else (sn=='None') if kind=='day_on' else (sn!=wanted) if kind=='shift_on' else (sn==wanted)
     if violation and pref.get('penalty',0)>0:violations.append((i,t,pref['penalty']))
   if violations:
    i,t,_=rng.choices(violations,weights=[v[2] for v in violations],k=1)[0];free.add(i);width=4 if loose_soft else 3;first=max(1,min(D-width+1,t-1));days.update(range(first,first+width))
   elif loose_soft:
    weekend=rng.choice(wkdays);days.update(range(max(1,min(weekend)-1),min(D,max(weekend)+1)+1))
   while len(free)<min(3 if loose_soft else 4,len(ids)):free.add(rng.choices(ids,weights=weights,k=1)[0])
  elif iteration%3==2:
   while len(free)<min(4,len(ids)):free.add(rng.choices(ids,weights=weights,k=1)[0])
  else:
   ranks=[]
   for k,weekend in enumerate(wkdays):
    mismatch=0
    for nu in nurses:
     choices=[next(sn for sn in shifts+['None'] if incumbent[x[nu['id'],t,sn].index]) for t in weekend]
     mismatch+=int(len(set(choices))>1)
    ranks.append((mismatch+rng.random(),k))
   ranks.sort(reverse=True);weekend=wkdays[ranks[(iteration//3)%len(ranks)][1]]
   days=set(weekend)
   if iteration%3==1:days.update(t for t in (min(weekend)-1,max(weekend)+1) if 1<=t<=D)
  repair=model.clone();repair.clear_hints()
  for (i,t,sn,sk),v in y.items():
   if i not in free and t not in days:repair.add(repair.get_int_var_from_proto_index(v.index)==incumbent[v.index])
  repair.add(sum(cost)<=int(best)-1)
  for j,value in enumerate(incumbent):repair.add_hint(repair.get_int_var_from_proto_index(j),value)
  solver.parameters.random_seed=iteration+1;solver.parameters.max_time_in_seconds=max(.1,min(5,deadline-time.monotonic()));solver.solve(repair,CB());iteration+=1

import os
license=os.environ.get('GRB_LICENSE_FILE')
if license and not os.path.exists(license):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
import argparse,json,time,random,collections
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-4);z=json.load(open(a.instance_path));trips=sorted(z['trips'],key=lambda t:t['start_time']);records={t['trip_id']:t for t in trips};ids=list(records);depots=[d['name'] for d in z['depots']];tt=z['travel_times'];cp=z['crew_parameters'];depotset=set(depots);relief=set(z['relief_locations']);rng=random.Random(0);short={};long={}
for dep in depots:
 short[dep]=set();long[dep]=set()
 for i in ids:
  for j in ids:
   x=records[i];y=records[j]
   if y['start_time']>=x['end_time']+tt[x['end_location']][y['start_location']]:
    target=long[dep] if y['start_time']-x['end_time']>=tt[x['end_location']][dep]+tt[dep][y['start_location']] else short[dep];target.add((i,j))
def prepost(loc,dep,is_start):
 if loc in depotset:return cp['sign_on_time_depot_minutes'] if is_start else cp['sign_off_time_depot_minutes']
 return cp['extra_time_non_depot_relief_minutes']+tt[loc][dep]
def valid(pieces,dep,dtype):
 if len(pieces)!=dtype['num_pieces']:return False
 spans=[records[q[-1]]['end_time']-records[q[0]]['start_time'] for q in pieces]
 if any((dtype['piece_length_min'] is not None and s<dtype['piece_length_min']) or (dtype['piece_length_max'] is not None and s>dtype['piece_length_max']) for s in spans):return False
 first=records[pieces[0][0]];last=records[pieces[-1][-1]];ds=first['start_time']-prepost(first['start_location'],dep,True);de=last['end_time']+prepost(last['end_location'],dep,False)
 if dtype['duty_length_max'] is not None and de-ds>dtype['duty_length_max']:return False
 if dtype['work_time_max'] is not None and sum(spans)>dtype['work_time_max']:return False
 if dtype['start_time_min'] is not None and ds<dtype['start_time_min']:return False
 if dtype['start_time_max'] is not None and ds>dtype['start_time_max']:return False
 if dtype['end_time_max'] is not None and de>dtype['end_time_max']:return False
 if len(pieces)==2:
  end1=records[pieces[0][-1]];start2=records[pieces[1][0]]
  if end1['end_location'] not in relief|depotset or start2['start_location'] not in relief|depotset:return False
  if start2['start_time']-end1['end_time']<(dtype['break_length_min'] or 0):return False
  if (pieces[0][-1],pieces[1][0]) in short[dep]:return False
 return True
pieces={d:{} for d in depots}
for dep in depots:
 adjacency={i:sorted([j for j in ids if (i,j) in short[dep]],key=lambda j:records[j]['end_time']) for i in ids}
 for seed in ids:
  stack=[[seed]];count=0
  while stack and count<400 and time.monotonic()<start+a.time_limit*.15:
   q=stack.pop();count+=1;pieces[dep][tuple(q)]=q
   for j in reversed(adjacency[q[-1]]):
    if records[j]['end_time']-records[seed]['start_time']<=300:stack.append(q+[j])
  pieces[dep][(seed,)]=[seed]

def duty(dep,qs):
 for name,dt in z['duty_types'].items():
  if valid(qs,dep,dt):return {'dep':dep,'pieces':[q.copy() for q in qs],'type':name,'trips':[i for q in qs for i in q]}
 return None

def key(d):return d['dep'],tuple(tuple(q) for q in d['pieces'])
duties=[];x={};links={};cache=set()
model=gp.Model('priced_integrated_master');model.Params.Seed=0;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;model.Params.Threads=1;model.Params.OutputFlag=0
source=model.addVars(depots,ids,obj=1);sink=model.addVars(depots,ids)
cover={i:model.addConstr(gp.LinExpr()==1) for i in ids}
starts={(dep,i):model.addConstr(source[dep,i]==0) for dep in depots for i in ids}
ends={(dep,i):model.addConstr(sink[dep,i]==0) for dep in depots for i in ids}
def insert(d):
 k=len(duties);duties.append(d);cache.add(key(d));cons=[cover[i] for i in d['trips']]+[starts[d['dep'],q[0]] for q in d['pieces']]+[ends[d['dep'],q[-1]] for q in d['pieces']];coef=[1]*len(d['trips'])+[-1]*(2*len(d['pieces']));x[k]=model.addVar(obj=1,column=gp.Column(coef,cons))
def link(dep,i,j):
 links[dep,i,j]=model.addVar(column=gp.Column([1,1],[ends[dep,i],starts[dep,j]]))
for dep in depots:
 for i in ids:
  d=duty(dep,[[i]])
  if d:insert(d)
 candidates=sorted(pieces[dep].values(),key=lambda q:len(q),reverse=True)
 frequency=collections.Counter()
 for q in candidates:
  if len(duties)>=500:break
  if len(q)==1 or any(frequency[i]>=8 for i in q):continue
  d=duty(dep,[q])
  if d:insert(d);frequency.update(q)
 for i in ids:
  for j in sorted([j for j in ids if (i,j) in long[dep]],key=lambda j:records[j]['start_time'])[:2]:link(dep,i,j)

for iteration in range(30):
 if time.monotonic()>=start+a.time_limit*.55:break
 model.Params.TimeLimit=max(.1,start+a.time_limit*.55-time.monotonic());model.optimize()
 if model.Status!=gp.GRB.OPTIMAL:break
 pi={i:cover[i].Pi for i in ids};ps={k:c.Pi for k,c in starts.items()};pe={k:c.Pi for k,c in ends.items()};fresh=[]
 for dep in depots:
  scored=sorted([(sum(pi[i] for i in q)-ps[dep,q[0]]-pe[dep,q[-1]],q) for q in pieces[dep].values()],key=lambda v:v[0],reverse=True)
  for reward,q in scored[:300]:
   d=duty(dep,[q])
   if reward>1+1e-6 and d and key(d) not in cache:fresh.append((1-reward,d))
  options=scored[:120];included={tuple(q) for reward,q in options};by_start=collections.defaultdict(list);by_end=collections.defaultdict(list)
  for reward,q in scored:by_start[q[0]].append((reward,q));by_end[q[-1]].append((reward,q))
  for pool in list(by_start.values())+list(by_end.values()):
   for reward,q in pool[:2]:
    if tuple(q) not in included:options.append((reward,q));included.add(tuple(q))
  options.sort(key=lambda v:v[0],reverse=True)
  for reward1,q1 in options:
   if time.monotonic()>=start+a.time_limit*.55:break
   for reward2,q2 in options:
    if reward1+reward2<1-1e-5:break
    if records[q1[-1]]['end_time']>=records[q2[0]]['start_time'] or (q1[-1],q2[0]) in short[dep]:continue
    d=duty(dep,[q1,q2])
    if d and key(d) not in cache:fresh.append((1-reward1-reward2+rng.random()*1e-5,d))
 arc_candidates=sorted([(-pe[dep,i]-ps[dep,j],dep,i,j) for dep in depots for i,j in long[dep] if (dep,i,j) not in links and pe[dep,i]+ps[dep,j]>=-1e-5])
 fresh.sort(key=lambda v:v[0]);fresh=fresh[:180];arc_candidates=arc_candidates[:80]
 if not fresh and not arc_candidates:break
 if len(x)+len(links)+len(source)+len(sink)+len(fresh)+len(arc_candidates)>1950:
  remove_x=[k for k in x if x[k].X<1e-8 and len(duties[k]['trips'])>1];remove_x.sort(key=lambda k:x[k].RC,reverse=True)
  remove_l=[k for k in links if links[k].X<1e-8];remove_l.sort(key=lambda k:links[k].RC,reverse=True)
  for k in remove_x[:min(len(remove_x),250)]:model.remove(x.pop(k));cache.discard(key(duties[k]))
  for k in remove_l[:min(len(remove_l),100)]:model.remove(links.pop(k))
  model.update()
 for reduced,d in fresh:
  if len(x)+len(links)+len(source)+len(sink)>=1900:break
  if key(d) not in cache:insert(d)
 for reduced,dep,i,j in arc_candidates:
  if len(x)+len(links)+len(source)+len(sink)>=1950:break
  if (dep,i,j) not in links:link(dep,i,j)
model.update()
for var in model.getVars():var.VType=gp.GRB.BINARY
model.Params.OutputFlag=1;model.Params.TimeLimit=max(.1,end-time.monotonic());logger=SolutionLogger(a.log_path,sense='minimize');best=float('inf')
logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
def save(xv,sv,lv):
 global best
 chosen=[k for k,value in xv.items() if value>.5];schedule={d:[] for d in depots};succ={}
 for k in chosen:
  d=duties[k]
  for q in d['pieces']:
   for i,j in zip(q,q[1:]):succ[d['dep'],i]=j
 for (dep,i,j),value in lv.items():
  if value>.5:succ[dep,i]=j
 for (dep,i),value in sv.items():
  if value>.5:
   q=[i]
   while (dep,q[-1]) in succ:q.append(succ[dep,q[-1]])
   schedule[dep].append(q)
 value=sum(len(q) for q in schedule.values())+len(chosen)
 if value>=best:return
 best=value;s={'objective_value':value,'vehicle_schedule':schedule,'crew_schedule':[{'depot':duties[k]['dep'],'type':duties[k]['type'],'trips':duties[k]['trips']} for k in chosen],'runtime':time.monotonic()-start};json.dump(s,open(a.solution_path,'w'));logger.log_solution(value,s)
def cb(m,where):
 if where==gp.GRB.Callback.MIPSOL:save(m.cbGetSolution(x),m.cbGetSolution(source),m.cbGetSolution(links))
model.optimize(cb)
if model.SolCount:save({k:v.X for k,v in x.items()},{k:v.X for k,v in source.items()},{k:v.X for k,v in links.items()})

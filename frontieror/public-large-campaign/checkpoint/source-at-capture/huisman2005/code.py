import argparse,json,time
from ortools.sat.python import cp_model

def solve(d,t):
 trips=sorted(d['trips'],key=lambda v:v['start_time']);tt=d['travel_times'];cp=d['crew_parameters'];deadline=time.monotonic()+max(.1,t-1);columns=[];limit=10 if len(trips)<=20 else 5
 def duty(path,dep):
  pieces=[[path[0]]]
  for prev,cur in zip(path,path[1:]):
   gap=cur['start_time']-prev['end_time'];back=tt[prev['end_location']][dep]+tt[dep][cur['start_location']]
   if gap>=back:pieces.append([cur])
   else:pieces[-1].append(cur)
  start=path[0]['start_time']-tt[dep][path[0]['start_location']]-cp['sign_on_time_depot_minutes'];end=path[-1]['end_time']+tt[path[-1]['end_location']][dep]+cp['sign_off_time_depot_minutes'];lengths=[p[-1]['end_time']-p[0]['start_time']+tt[dep][p[0]['start_location']]+tt[p[-1]['end_location']][dep] for p in pieces]
  for typ,v in d['duty_types'].items():
   if len(pieces)!=v['num_pieces']:continue
   if any(x<v['piece_length_min'] or x>v['piece_length_max'] for x in lengths):continue
   if v['duty_length_max'] is not None and end-start>v['duty_length_max']:continue
   if v['work_time_max'] is not None and sum(lengths)+cp['sign_on_time_depot_minutes']+cp['sign_off_time_depot_minutes']>v['work_time_max']:continue
   if v['start_time_min'] is not None and start<v['start_time_min']:continue
   if v['start_time_max'] is not None and start>v['start_time_max']:continue
   if v['end_time_max'] is not None and end>v['end_time_max']:continue
   if len(pieces)==2:
    a,b=pieces;breaktime=b[0]['start_time']-tt[dep][b[0]['start_location']]-(a[-1]['end_time']+tt[a[-1]['end_location']][dep])
    if breaktime<v['break_length_min']:continue
   return typ
  return None
 def expand(path,dep,pos):
  if time.monotonic()>deadline-t*.5 or len(columns)>100000:return
  typ=duty(path,dep)
  if typ:columns.append((dep,typ,[v['trip_id'] for v in path]))
  if len(path)>=limit:return
  prev=path[-1]
  for j in range(pos+1,len(trips)):
   cur=trips[j]
   if prev['end_time']+tt[prev['end_location']][cur['start_location']]<=cur['start_time']:expand(path+[cur],dep,j)
 for dep in d['depots']:
  for j,trip in enumerate(trips):expand([trip],dep['name'],j)
 m=cp_model.CpModel();x=[m.new_bool_var(str(i)) for i in range(len(columns))]
 for trip in trips:
  inds=[i for i,c in enumerate(columns) if trip['trip_id'] in c[2]]
  if not inds:raise RuntimeError(f'No valid integrated vehicle/crew chain for trip {trip["trip_id"]} under enumerated duty rules')
  m.add_exactly_one(x[i] for i in inds)
 m.minimize(sum(x));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,deadline-time.monotonic());r=s.solve(m)
 if r not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError('No integrated crew/vehicle exact cover')
 schedule={dep['name']:[] for dep in d['depots']};crew=[];pieces={dep['name']:[] for dep in d['depots']};byid={v['trip_id']:v for v in trips}
 for i,(dep,typ,path) in enumerate(columns):
  if not s.value(x[i]):continue
  crew.append({'depot':dep,'type':typ,'trips':path});current=[path[0]]
  for prev,cur in zip(path,path[1:]):
   a=byid[prev];b=byid[cur];gap=b['start_time']-a['end_time'];back=tt[a['end_location']][dep]+tt[dep][b['start_location']]
   if gap>=back:pieces[dep].append(current);current=[cur]
   else:current.append(cur)
  pieces[dep].append(current)
 for dep,chains in pieces.items():
  count=len(chains);adj=[[] for _ in chains]
  for i,path in enumerate(chains):
   a=byid[path[-1]]
   for j,other in enumerate(chains):
    if i==j:continue
    b=byid[other[0]]
    if a['end_time']+tt[a['end_location']][dep]+tt[dep][b['start_location']]<=b['start_time']:adj[i].append(j)
  matched={}
  def augment(i,seen):
   for j in adj[i]:
    if j in seen:continue
    seen.add(j)
    if j not in matched or augment(matched[j],seen):matched[j]=i;return True
   return False
  for i in range(count):augment(i,set())
  successor={i:j for j,i in matched.items()}
  for i in range(count):
   if i in matched:continue
   path=list(chains[i]);j=i
   while j in successor:j=successor[j];path.extend(chains[j])
   schedule[dep].append(path)
 return {'objective_value':sum(len(v) for v in schedule.values())+len(crew),'vehicle_schedule':schedule,'crew_schedule':crew}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

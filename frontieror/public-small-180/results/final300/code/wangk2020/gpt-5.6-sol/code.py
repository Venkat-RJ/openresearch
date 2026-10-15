import argparse,json,time,collections
wall_started=time.time()
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True)
a=p.parse_args(); end=time.monotonic()+a.time_limit-1; d=json.load(open(a.instance_path)); logger=configured_logger(a.log_path,sense='minimize'); fs=d['flights']; by={f['id']:f for f in fs}; delta=d['maximum_displacement_delta']; T=d['num_time_periods']; rho=d.get('rho',.67)
sd={f['id']:f['dep_period']-min(1,delta) for f in fs}; con=d['aircraft_connections']+d['passenger_connections']
for _ in range(len(fs)):
 change=False
 for c in con:
  i,j=c['flight_i'],c['flight_j']; needed=sd[i]+by[i]['delta_sch']+c['min_connection_time']
  if sd[j]<needed: sd[j]=needed; change=True
 if not change: break
assert all(abs(sd[f['id']]-f['dep_period'])<=delta for f in fs)
schedule={str(f['id']):{'scheduled_dep':sd[f['id']],'scheduled_arr':sd[f['id']]+f['delta_sch']} for f in fs}
pred=collections.defaultdict(list); succ=collections.defaultdict(list); indeg={f['id']:0 for f in fs}
for c in d['aircraft_connections']:
 i,j=c['flight_i'],c['flight_j']; pred[j].append((i,c['min_connection_time'])); succ[i].append(j); indeg[j]+=1
q=[f['id'] for f in fs if indeg[f['id']]==0]; order=[]
while q:
 i=min(q,key=lambda v:sd[v]); q.remove(i); order.append(i)
 for j in succ[i]:
  indeg[j]-=1
  if indeg[j]==0: q.append(j)
assert len(order)==len(fs)
scs=[]; delay=0.
for sc in d['scenarios']:
 counts=collections.defaultdict(lambda:[0,0]); dep={}; arr={}
 def fits(airport,t,kind):
  # Step indicators drop at operated time plus one.
  t=t+1
  if t<1 or t>T: return True
  x,y=counts[airport,t]; x+=kind==0; y+=kind==1
  return all(s['a']*x+s['b']*y<=s['Q'] for s in d['capacity_envelopes'][airport][sc['operating_conditions'][airport][t-1]])
 for fid in order:
  if time.monotonic()>=end: raise TimeoutError('Flight schedule incomplete')
  f=by[fid]; lo=max(sd[fid],max((arr[i]+tau for i,tau in pred[fid]),default=sd[fid])); options=[]
  for td in range(lo,sd[fid]+f['max_dep_delay']+1):
   if not fits(f['dep_airport'],td,0): continue
   low=max(td+f['delta_min'],sd[fid]+f['delta_min'])
   high=min(td+f['delta_max'],sd[fid]+f['delta_sch']+f['max_arr_delay'])
   for ta in range(low,high+1):
    if fits(f['arr_airport'],ta,1): options.append((f['cost_dep_delay']*(td-sd[fid])+f['cost_arr_delay']*max(0,ta-sd[fid]-f['delta_sch']),td,ta))
  if not options: raise RuntimeError('Greedy recourse cannot schedule flight '+str(fid))
  _,td,ta=min(options,key=lambda v:(v[0],abs(v[2]-(sd[fid]+f['delta_sch'])),v[1])); dep[fid]=td; arr[fid]=ta; counts[f['dep_airport'],td+1][0]+=1; counts[f['arr_airport'],ta+1][1]+=1
 sol={'scenario_id':sc['id'],'x_dep':{},'x_arr':{},'v_dep':{},'v_arr':{}}
 for f in fs:
  i=f['id']; sol['v_dep'][str(i)]=dep[i]-sd[i]; sol['v_arr'][str(i)]=max(0,arr[i]-sd[i]-f['delta_sch'])
  for t in range(f['dep_period']-delta+1,f['dep_period']+delta+f['max_dep_delay']+1): sol['x_dep'][f'{i}_{t}']=int(t<=dep[i])
  for t in range(f['dep_period']-delta+f['delta_min']+1,f['arr_period']+delta+f['max_arr_delay']+1): sol['x_arr'][f'{i}_{t}']=int(t<=arr[i])
  delay+=sc['probability']*(f['cost_dep_delay']*sol['v_dep'][str(i)]+f['cost_arr_delay']*sol['v_arr'][str(i)])
 scs.append(sol)
disp=sum(abs(sd[f['id']]+1-f['dep_period']) for f in fs if f['dep_period']-delta+1<=sd[f['id']]+1<=f['dep_period']+delta)
z=rho*disp+(1-rho)*delay; sol={'objective_value':z,'schedule':schedule,'scenario_solutions':scs}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(z,sol)

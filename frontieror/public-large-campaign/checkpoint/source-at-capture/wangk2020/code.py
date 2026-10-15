import argparse,json,os
os.environ['OMP_NUM_THREADS']='1'
from ortools.sat.python import cp_model

def solve(d,limit):
 flights=d['flights'];delta=d['maximum_displacement_delta'];T=d['num_time_periods'];rho=d.get('rho',.5);m=cp_model.CpModel();SD={};SA={};OD={};OA={};VD={};VA={};cost=[];scale=100000
 for f in flights:
  i=f['id'];lo=f['dep_period']-delta;hi=f['dep_period']+delta;SD[i]=m.new_int_var(lo,hi,f'sd{i}');SA[i]=m.new_int_var(f['arr_period']-delta,f['arr_period']+delta,f'sa{i}');m.add(SA[i]-SD[i]==f['delta_sch']);vals=[abs(t+1-f['dep_period']) if t<hi else 0 for t in range(lo,hi+1)];v=m.new_int_var(0,max(vals),'displacement'+str(i));m.add_element(SD[i]-lo,vals,v);cost.append(round(scale*rho)*v)
 for conn in d['aircraft_connections']+d['passenger_connections']:m.add(SD[conn['flight_j']]-SA[conn['flight_i']]>=conn['min_connection_time'])
 for sc in d['scenarios']:
  sid=sc['id'];devents={};aevents={}
  for f in flights:
   i=f['id'];OD[sid,i]=m.new_int_var(f['dep_period']-delta,f['dep_period']+delta+f['max_dep_delay'],f'od{sid}_{i}');OA[sid,i]=m.new_int_var(f['dep_period']-delta+f['delta_min'],f['arr_period']+delta+f['max_arr_delay'],f'oa{sid}_{i}');VD[sid,i]=m.new_int_var(0,f['max_dep_delay'],f'vd{sid}_{i}');VA[sid,i]=m.new_int_var(0,f['max_arr_delay'],f'va{sid}_{i}');m.add(OD[sid,i]-SD[i]==VD[sid,i]);m.add(OA[sid,i]-SA[i]<=VA[sid,i]);m.add(OA[sid,i]>=SA[i]-(f['delta_sch']-f['delta_min']));m.add(OA[sid,i]-OD[sid,i]>=f['delta_min']);m.add(OA[sid,i]-OD[sid,i]<=f['delta_max']);cost.extend([round(scale*(1-rho)*sc['probability']*f['cost_dep_delay'])*VD[sid,i],round(scale*(1-rho)*sc['probability']*f['cost_arr_delay'])*VA[sid,i]])
   for t in range(1,T+1):
    dep=m.new_bool_var(f'd{sid}_{i}_{t}');arr=m.new_bool_var(f'a{sid}_{i}_{t}');m.add(OD[sid,i]==t-1).only_enforce_if(dep);m.add(OD[sid,i]!=t-1).only_enforce_if(dep.Not());m.add(OA[sid,i]==t-1).only_enforce_if(arr);m.add(OA[sid,i]!=t-1).only_enforce_if(arr.Not());devents.setdefault((f['dep_airport'],t),[]).append(dep);aevents.setdefault((f['arr_airport'],t),[]).append(arr)
  for conn in d['aircraft_connections']:m.add(OD[sid,conn['flight_j']]-OA[sid,conn['flight_i']]>=conn['min_connection_time'])
  for airport in d['airports']:
   for t in range(1,T+1):
    cond=sc['operating_conditions'][airport][t-1]
    for seg in d['capacity_envelopes'][airport][cond]:m.add(seg['a']*sum(devents.get((airport,t),[]))+seg['b']*sum(aevents.get((airport,t),[]))<=seg['Q'])
 m.minimize(sum(cost));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,limit-1);st=s.solve(m)
 if st not in (cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError(s.status_name(st))
 schedule={str(f['id']):{'scheduled_dep':s.value(SD[f['id']]),'scheduled_arr':s.value(SA[f['id']])} for f in flights};out=[];obj=0
 for f in flights:
  sd=s.value(SD[f['id']]);obj+=rho*(abs(sd+1-f['dep_period']) if sd<f['dep_period']+delta else 0)
 for sc in d['scenarios']:
  sid=sc['id'];xd={};xa={};vd={};va={}
  for f in flights:
   i=f['id'];dep=s.value(OD[sid,i]);arr=s.value(OA[sid,i]);vd[str(i)]=s.value(VD[sid,i]);va[str(i)]=s.value(VA[sid,i]);obj+=(1-rho)*sc['probability']*(f['cost_dep_delay']*vd[str(i)]+f['cost_arr_delay']*va[str(i)])
   for t in range(f['dep_period']-delta+1,f['dep_period']+delta+f['max_dep_delay']+1):xd[f'{i}_{t}']=int(t<=dep)
   for t in range(f['dep_period']-delta+f['delta_min']+1,f['arr_period']+delta+f['max_arr_delay']+1):xa[f'{i}_{t}']=int(t<=arr)
  out.append({'scenario_id':sid,'x_dep':xd,'x_arr':xa,'v_dep':vd,'v_arr':va})
 return {'objective_value':obj,'schedule':schedule,'scenario_solutions':out}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)

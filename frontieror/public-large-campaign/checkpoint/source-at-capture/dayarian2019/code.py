import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,itertools,math,random
from ortools.sat.python import cp_model

def main():
 parser=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):parser.add_argument('--'+k,required=k!='log_path')
 a=parser.parse_args();started=time.monotonic();deadline=started+max(.1,float(a.time_limit)-2);d=json.load(open(a.instance_path));orders={v['order_id']:v for v in d['orders']};products={v['product_id']:v for v in d['products']};g=d['global_parameters'];D=d['travel_time_matrix'];periods={v['period_id']:v for v in d['periods']};subperiods={v['subperiod_id']:v for v in d['subperiods']};windows=d['production_windows'];N=len(orders);trips={}
 def trip(row,t):
  begin=periods[t]['start_time'];clock=begin+g['preparation_time_minutes'];last=0;travel=0;end=begin
  for i in row:
   loc=i+1;travel+=D[last][loc];clock=max(clock+D[last][loc],orders[i]['time_window_start'])
   if clock>orders[i]['time_window_end']+1e-7:return None
   for pid in orders[i]['demands']:
    valid=windows[str(i)][pid]['valid_subperiod_ids']
    if not any(subperiods[u]['end_time']<=begin for u in valid):return None
   end=max(end,clock+orders[i]['service_time']+D[loc][0]);clock+=orders[i]['service_time'];last=loc
  travel+=D[last][0]
  if end-begin>g['max_shift_route_length_minutes']:return None
  return {'customers':list(row),'start_period':t,'start':begin,'end':end,'travel':travel}
 ids=list(orders)
 if N<=12:
  for size in range(1,min(N,5)+1):
   if time.monotonic()>deadline-15:break
   for subset in itertools.combinations(ids,size):
    if time.monotonic()>deadline-15:break
    if sum(orders[i]['space_requirement']for i in subset)>d['vehicle_capacity']:continue
    for t in periods:
     if time.monotonic()>deadline-15:break
     best=None
     for row in itertools.permutations(subset):
      candidate=trip(row,t)
      if candidate is not None and(best is None or candidate['travel']<best['travel']):best=candidate
     if best is not None:trips[tuple(subset),t]=best
 else:
  rng=random.Random(2019);jobs=[];trip_end=started+min(12,float(a.time_limit)*.25)
  for t in periods:
   for i in ids:
    candidate=trip([i],t)
    if candidate is not None:trips[(i,),t]=candidate;jobs.append((i,t))
  for attempt in range(12):
   rng.shuffle(jobs)
   for seed,t in jobs:
    if len(trips)>=2400 or time.monotonic()>trip_end:break
    row=[seed];load=orders[seed]['space_requirement']
    while len(row)<min(N,5):
     choices=[]
     for i in ids:
      if i in row or load+orders[i]['space_requirement']>d['vehicle_capacity']:continue
      for position in range(len(row)+1):
       proposed=row[:position]+[i]+row[position:];candidate=trip(proposed,t)
       if candidate is not None:choices.append((candidate['travel']*(1+rng.random()*.3),i,candidate))
     if not choices:break
     score,i,candidate=min(choices,key=lambda choice:choice[0]);row=candidate['customers'];load+=orders[i]['space_requirement'];key=tuple(sorted(row)),t
     if key not in trips or candidate['travel']<trips[key]['travel']:trips[key]=candidate
     if len(trips)>=2400 or time.monotonic()>trip_end:break
   if len(trips)>=2400 or time.monotonic()>trip_end:break
 candidates={}
 def put(rr):
  departure={i:r['start_period']for r in rr for i in r['customers']};mask=tuple(sorted(departure));start=min(r['start']for r in rr);end=max(r['end']for r in rr);duration=end-start
  if duration>g['max_shift_route_length_minutes']:return
  cost=g['driver_salary_per_minute']*max(duration,g['min_paid_time_minutes'])+g['vehicle_operating_cost_per_minute']*sum(r['travel']for r in rr);key=(mask,tuple(departure[i]for i in mask))
  if key not in candidates or cost<candidates[key][0]:candidates[key]=(cost,rr,departure)
 alltrips=list(trips.values())
 for tr in alltrips:put([tr])
 if g['max_trips_per_route']>=2:
  if N<=12:
   for i,tr in enumerate(alltrips):
    if time.monotonic()>deadline-12:break
    for j in range(i+1,len(alltrips)):
     if time.monotonic()>deadline-12:break
     nxt=alltrips[j]
     if tr['start_period']==nxt['start_period']or set(tr['customers'])&set(nxt['customers']):continue
     put(sorted([tr,nxt],key=lambda v:v['start_period']))
  else:
   masks=[sum(1<<ids.index(i)for i in tr['customers'])for tr in alltrips];indices=list(range(len(alltrips)));rng.shuffle(indices);pair_end=started+min(20,float(a.time_limit)*.4)
   for index in indices:
    if len(candidates)>=6000 or time.monotonic()>pair_end:break
    tr=alltrips[index];compatible=[]
    for j,nxt in enumerate(alltrips):
     if masks[index]&masks[j]or tr['start_period']==nxt['start_period']:continue
     duration=max(tr['end'],nxt['end'])-min(tr['start'],nxt['start'])
     if duration>g['max_shift_route_length_minutes']:continue
     cost=g['driver_salary_per_minute']*max(duration,g['min_paid_time_minutes'])+g['vehicle_operating_cost_per_minute']*(tr['travel']+nxt['travel']);compatible.append((cost/(len(tr['customers'])+len(nxt['customers'])),j))
    for score,j in sorted(compatible)[:12]:
     put(sorted([tr,alltrips[j]],key=lambda v:v['start_period']))
     if len(candidates)>=6000:break
 routecols=list(candidates.values());m=cp_model.CpModel();theta=[m.NewBoolVar('route'+str(i))for i in range(len(routecols))];obj=[int(round(10000*col[0]))*v for col,v in zip(routecols,theta)];maxvehicles=min(d['num_vehicles'],d['num_drivers']);m.Add(sum(theta)<=maxvehicles)
 incidence={i:[]for i in orders};departures={}
 for col,v in zip(routecols,theta):
  for i,t in col[2].items():incidence[i].append(v);departures.setdefault((i,t),[]).append(v)
 for i in orders:m.Add(sum(incidence[i])==1)
 departure_vars={}
 if N>12:
  for(i,t),variables in departures.items():
   departure_vars[i,t]=m.NewBoolVar('departure%d_%d'%(i,t));m.Add(departure_vars[i,t]==sum(variables))
 q={};valid_by_product={p:set()for p in products}
 for i,order in orders.items():
  for pp,dem in order['demands'].items():
   p=int(pp);valid=windows[str(i)][pp]['valid_subperiod_ids'];valid_by_product[p].update(valid)
   for u in valid:q[i,p,u]=m.NewIntVar(0,int(dem*1000),'q%d_%d_%d'%(i,p,u))
   m.Add(sum(q[i,p,u]for u in valid)==int(dem*1000))
   for u in valid:
    allowed=sum(v for(ii,t),v in departure_vars.items()if ii==i and periods[t]['start_time']>=subperiods[u]['end_time'])if N>12 else sum(v for col,v in zip(routecols,theta)if i in col[2]and periods[col[2][i]]['start_time']>=subperiods[u]['end_time']);m.Add(q[i,p,u]<=int(dem*1000)*allowed)
 E=d['num_production_employees'];W=d['num_workstations'];f={};beta={};sigma=g['setup_length_subperiods'];w=[m.NewIntVar(0,W,'w'+str(t))for t in range(g['num_periods'])];alpha=[m.NewIntVar(0,E,'alpha'+str(t))for t in range(g['num_periods'])]
 for p,valid in valid_by_product.items():
  for u in sorted(valid):f[p,u]=m.NewIntVar(0,E,'f%d_%d'%(p,u));beta[p,u]=m.NewIntVar(0,E,'b%d_%d'%(p,u));obj.append(int(round(10000*g['setup_cost']))*beta[p,u])
  prev=0
  for u in sorted(valid):
   m.Add(f[p,u]<=prev+beta.get((p,u-sigma),0));prev=f[p,u];m.Add(sum(v for(i,pp,uu),v in q.items()if pp==p and uu==u)<=int(round(products[p]['production_rate_per_subperiod']*1000))*f[p,u])
 sp=g['subperiods_per_period'];minpaid=g['min_paid_time_subperiods']//sp
 for u in range(g['num_subperiods']):
  active=sum(f.get((p,u),0)+sum(beta.get((p,v),0)for v in range(max(0,u-sigma+1),u+1))for p in products);m.Add(active<=E);m.Add(active<=w[u//sp])
 for t in range(g['num_periods']):
  m.Add(alpha[t]>=w[t]-(w[t-1]if t else 0));m.Add(w[t]>=sum(alpha[t-j]for j in range(1,minpaid+1)if t-j>=0));obj.append(int(round(10000*g['employee_salary_per_subperiod']*sp))*w[t])
 m.Add(sum(alpha)<=E);m.Minimize(sum(obj));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());status=s.Solve(m)
 if status not in(cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No incumbent found in the bounded caterer route-column search within the time limit')
 chosen=[col for col,v in zip(routecols,theta)if s.Value(v)];routes=[{'vehicle':i,'trips':[{'trip':j,'customers':tr['customers'],'start_period':tr['start_period']}for j,tr in enumerate(col[1])]}for i,col in enumerate(chosen)];schedule=[{'order':i,'product':p,'subperiod':u,'quantity':s.Value(v)/1000}for(i,p,u),v in q.items()if s.Value(v)];staff={'employees_by_product_subperiod':[{'product':p,'subperiod':u,'employees':s.Value(v)}for(p,u),v in f.items()],'setups_by_product_subperiod':[{'product':p,'subperiod':u,'setups':s.Value(v)}for(p,u),v in beta.items()],'workstations_by_period':[s.Value(v)for v in w],'shift_starts_by_period':[s.Value(v)for v in alpha]};value=sum(col[0]for col in chosen)+g['employee_salary_per_subperiod']*sp*sum(s.Value(v)for v in w)+g['setup_cost']*sum(s.Value(v)for v in beta.values());json.dump({'objective_value':value,'routes':routes,'production_schedule':schedule,'production_staffing':staff},open(a.solution_path,'w'))
if __name__=='__main__':main()

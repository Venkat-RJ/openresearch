import argparse,json,time,random,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-3);z=json.load(open(a.instance_path));T=z['planning_horizon']['num_periods'];cars=z['carousels'];flights=z['flights'];target=z['objective']['target_utilization'];threshold=z['objective']['threshold_values'];penalty=z['objective']['penalties'];peaks=z['peak_intervals'];C=len(cars);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;best=float('inf');rng=random.Random(0);options=[]
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 def flags(work):
  val=0;viol=[]
  for c,car in enumerate(cars):
   for k,peak in enumerate(peaks):
    excess=float(work[c,peak['start_period']:peak['end_period']+1].max()/car['Kcb']-target)
    if excess>1e-9:
     h=next((i for i,q in enumerate(threshold) if q>=excess-1e-9),None)
     if h is None:return float('inf'),[]
     val+=penalty[h];viol.append({'carousel_id':car['carousel_id'],'peak_interval':k+1,'threshold_level':h+1,'threshold_value':threshold[h]})
  return val,viol
 for flight in flights:
  opts=[];es,se=flight['earliest_start_period'],flight['end_period']
  for c,car in enumerate(cars):
   bounds=flight['working_station_bounds_by_carousel_type'].get(str(car['type_r']))
   if not bounds:continue
   for sh in range(es,min(flight['latest_start_period'],se-z['central_storage']['depletion_offset_periods']-1)+1):
    for w in range(bounds['Wmin'],min(bounds['Wmax'],car['Kws'])+1):
     gam=np.zeros(T,dtype=int);phi=np.zeros(T,dtype=int);stored=belt=0
     for t in range(es,se):
      k=t-es;arr=flight['arrival_profile'][k] if k<len(flight['arrival_profile']) else 0
      if t<sh:stored+=arr
      else:
       belt+=arr;depleted=min(stored,z['central_storage']['depletion_rate_bags_per_period']);stored-=depleted;belt+=depleted;belt=max(0,belt-w*z['working_rate_bags_per_period'])
      gam[t]=belt;phi[t]=stored
     opts.append((c,w,sh,gam,phi))
  options.append(opts)
 iteration=0
 while time.monotonic()<end:
  ws=np.zeros((C,T),dtype=int);pp=ws.copy();work=ws.copy();storage=np.zeros(T,dtype=int);assign=[];order=sorted(range(len(flights)),key=lambda i:(flights[i]['latest_start_period'],flights[i]['end_period']))
  if iteration:rng.shuffle(order)
  for i in order:
   if time.monotonic()>=end:break
   flight=flights[i];se=flight['end_period'];choice=None;score=None
   for c,w,sh,gam,phi in options[i]:
    if np.any(ws[c,sh:se]+w>cars[c]['Kws']) or np.any(pp[c,sh:se]+flight['num_containers']>cars[c]['Kpp']) or np.any(storage+phi>z['central_storage']['capacity_bags']):continue
    candidate=work.copy();candidate[c]+=gam;val,_=flags(candidate);tie=(val,int(gam.sum()),int(phi.sum()),w,sh)
    if score is None or tie<score:score=tie;choice=(c,w,sh,gam,phi)
   if choice is None:break
   c,w,sh,gam,phi=choice;ws[c,sh:se]+=w;pp[c,sh:se]+=flight['num_containers'];work[c]+=gam;storage+=phi;assign.append({'flight_id':flight['flight_id'],'carousel_id':cars[c]['carousel_id'],'carousel_type':cars[c]['type_r'],'working_stations':w,'handling_start_period':sh,'depletion_start_period':sh,'end_period':se})
  if len(assign)==len(flights):
   val,viol=flags(work)
   if val<best:
    best=val;sol={'objective_value':val,'assignments':assign,'utilization_violations':viol,'runtime':time.monotonic()-start};json.dump(sol,open(a.solution_path,'w'))
    if logger:logger.log_solution(val,sol)
   if best==0:break
  iteration+=1
if __name__=='__main__':main()

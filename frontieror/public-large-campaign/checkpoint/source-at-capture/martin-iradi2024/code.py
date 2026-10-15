import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));ports={p['id']:p['quay_length_meters'] for p in d['ports']};cp=d['cost_parameters'];beta=d['handling_time_deviation_factor_beta_per_meter'];deadline=time.monotonic()+max(.1,a.time_limit-2);rng=random.Random(0);best=float('inf');logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
for it in range(100000):
 if time.monotonic()>deadline:break
 occupied={p:[] for p in ports}
 for ext in d['external_ships']:
  for s in ext['ships']:occupied[ext['port_id']].append((s['berthing_position'],s['berthing_position']+s['length_meters'],s['start_time_hours'],s['end_time_hours']))
 ships=d['ships'].copy()
 if it:rng.shuffle(ships)
 results=[];obj=0;valid=True
 for ship in ships:
  visits=[];speeds=[];arrival=ship['port_visits'][0]['earliest_start_time_hours'];L=ship['length_meters']
  for z,visit in enumerate(ship['port_visits']):
   port=visit['port_id'];ideal=visit['ideal_berthing_position'];candidate={max(0,min(ports[port]-L,ideal)),0,ports[port]-L}
   for x1,x2,t1,t2 in occupied[port]:
    if 0<=x1-L<=ports[port]-L:candidate.add(x1-L)
    if 0<=x2<=ports[port]-L:candidate.add(x2)
   choices=[]
   for x in candidate:
    dev=abs(x-ideal);h=visit['min_handling_time_hours']*(1+beta*dev);start=max(arrival,visit['earliest_start_time_hours'])
    while True:
     conflict=[t2 for x1,x2,t1,t2 in occupied[port] if x<x2-1e-7 and x+L>x1+1e-7 and start<t2-1e-7 and start+h>t1+1e-7]
     if not conflict:break
     start=max(conflict)
    delay=max(0,start+h-visit['expected_finish_time_hours']);lft=max(0,start+h-visit['latest_finish_time_hours']);cost=cp['waiting_cost_usd_per_hour']*(start-arrival)+cp['handling_cost_usd_per_hour']*h+cp['delay_cost_usd_per_hour']*delay+cp['lft_penalty_usd']*lft;choices.append((cost,x,start,h,dev,delay,lft))
   cost,x,start,h,dev,delay,lft=min(choices);obj+=cost
   if start+h>d['planning_horizon_hours']+1e-7:valid=False
   occupied[port].append((x,x+L,start,start+h));visits.append({'port_call_index':visit['port_call_index'],'port_id':port,'berthing_position':x,'start_time':start,'handling_time':h,'arrival_time':arrival,'delay':delay,'lft_violation':lft,'position_deviation':dev})
   if z+1<len(ship['port_visits']):
    nextv=ship['port_visits'][z+1];dist=d['distances_nautical_miles'][port+'_'+nextv['port_id']];options=[]
    for speed in d['speeds_knots']:
     key=str(speed);travel=d['travel_time_hours_per_nm_by_speed'][key]*dist;fuel=ship['fuel_consumption_per_nm_by_speed'][key]*dist*cp['fuel_price_usd_per_tonne'];late=max(0,start+h+travel-nextv['earliest_start_time_hours']);score=fuel+late*(cp['delay_cost_usd_per_hour']+cp['lft_penalty_usd']);options.append((score*(1 if not it else rng.uniform(.7,1.3)),speed,travel,fuel))
    _,speed,travel,fuel=min(options);arrival=start+h+travel;obj+=fuel;speeds.append({'from_call':visit['port_call_index'],'to_call':nextv['port_call_index'],'speed_knots':speed})
  results.append({'ship_id':ship['id'],'port_visits':visits,'speed_selections':speeds})
 if valid and obj<best:
  best=obj;s={'objective_value':obj,'ships':results};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)

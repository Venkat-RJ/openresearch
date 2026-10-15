import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def profiles(f,w,h,b,wr,dr):
 stored=0;belt=0;g={};p={}
 for t in range(f['earliest_start_period'],f['end_period']):
  idx=t-f['earliest_start_period'];a=f['arrival_profile'][idx] if idx<len(f['arrival_profile']) else 0
  if t<h:stored+=a;g[t]=0;p[t]=stored
  else:
   belt+=a
   if t>=b:de=min(stored,dr);belt+=de;stored-=de
   belt-=min(belt,w*wr);g[t]=belt;p[t]=stored
 return g,p

def solve(d,t):
 H=d['planning_horizon']['num_periods'];cars=[{**c,**d['carousel_types'][str(c['type_r'])]} for c in d['carousels']];flights=d['flights'];obj=d['objective'];opts=[];fg=[];deadline=time.monotonic()+max(.1,t-8);restricted=len(flights)>50;cache={}
 for f in flights:
  inds=[]
  for car in cars:
   if str(car['type_r']) not in f['working_station_bounds_by_carousel_type']:continue
   bounds=f['working_station_bounds_by_carousel_type'][str(car['type_r'])]
   ws=range(bounds['Wmin'],min(bounds['Wmax'],car['Kws'])+1)
   if restricted:ws=sorted({ws.start,ws.stop-1}) if ws else []
   hs=range(f['earliest_start_period'],f['latest_start_period']+1)
   if restricted:hs=sorted({hs.start,hs.stop-1})
   for w in ws:
    for h in hs:
     bs=range(h,f['end_period']-d['central_storage']['depletion_offset_periods'])
     if restricted:bs=sorted({bs.start,(bs.start+bs.stop-1)//2,bs.stop-1}) if bs else []
     for b in bs:
      key=(f['flight_id'],w,h,b)
      if key not in cache:cache[key]=profiles(f,w,h,b,d['working_rate_bags_per_period'],d['central_storage']['depletion_rate_bags_per_period'])
      g,p=cache[key];inds.append(len(opts));opts.append((f,car,w,h,b,g,p))
  fg.append(inds)
 n=len(opts);thresholds=[(car,peak,level) for car in cars for peak in d['peak_intervals'] for level in range(len(obj['threshold_values']))];N=n+len(thresholds);cost=np.zeros(N);cost[n:]=[obj['penalties'][l] for c,p,l in thresholds];rows=[];ub=[];lb=[]
 def row(rr,l,u):rows.append(rr);lb.append(l);ub.append(u)
 for inds in fg:row({i:1 for i in inds},1,1)
 for car in cars:
  inds=[i for i,o in enumerate(opts) if o[1]['carousel_id']==car['carousel_id']]
  for h in range(H):
   active=[i for i in inds if opts[i][3]<=h<opts[i][0]['end_period']]
   row({i:opts[i][2] for i in active},-np.inf,car['Kws']);row({i:opts[i][0]['num_containers'] for i in active},-np.inf,car['Kpp'])
  for peak in d['peak_intervals']:
   zs=[n+j for j,(c,p,l) in enumerate(thresholds) if c['carousel_id']==car['carousel_id'] and p['peak_id']==peak['peak_id']];row({z:1 for z in zs},-np.inf,1)
   for h in range(peak['start_period'],peak['end_period']+1):
    rr={i:opts[i][5].get(h,0)/car['Kcb'] for i in inds if opts[i][5].get(h,0)}
    for z in zs:rr[z]=-obj['threshold_values'][thresholds[z-n][2]]
    row(rr,-np.inf,obj['target_utilization'])
 for h in range(H):row({i:o[6][h] for i,o in enumerate(opts) if o[6].get(h,0)},-np.inf,d['central_storage']['capacity_bags'])
 A=lil_matrix((len(rows),N))
 for h,rr in enumerate(rows):
  for i,v in rr.items():A[h,i]=v
 z=milp(cost,integrality=np.ones(N),bounds=Bounds(np.zeros(N),np.ones(N)),constraints=LinearConstraint(A.tocsr(),lb,ub),options={'threads':1,'time_limit':max(.1,deadline-time.monotonic())})
 if z.x is None:raise RuntimeError('No carousel schedule incumbent')
 assignments=[{'flight_id':f['flight_id'],'carousel_id':c['carousel_id'],'carousel_type':c['type_r'],'working_stations':w,'handling_start_period':h,'depletion_start_period':b,'end_period':f['end_period']} for i,(f,c,w,h,b,g,p) in enumerate(opts) if z.x[i]>.5];violations=[{'carousel_id':c['carousel_id'],'peak_interval':p['peak_id'],'threshold_level':l+1,'threshold_value':obj['threshold_values'][l]} for j,(c,p,l) in enumerate(thresholds) if z.x[n+j]>.5]
 return {'objective_value':sum(obj['penalties'][v['threshold_level']-1] for v in violations),'assignments':assignments,'utilization_violations':violations}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

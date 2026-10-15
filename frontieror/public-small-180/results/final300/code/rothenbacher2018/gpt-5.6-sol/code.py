import time
_wall_start=time.time()
import argparse,json,os,itertools
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
P=d['parameters'];base=d['depot']['id'];NL={q['id'] for q in d['truck_customers']};NH={q['id'] for q in d['trailer_customers']};customers={q['id']:q for q in d['truck_customers']+d['trailer_customers']};J=sorted(customers);parking=sorted(NH|{q['id'] for q in d['transshipment_locations']});D=d['distance_matrix'];TT=d['travel_time_matrix'];orig=len(D);dc={j:orig+i for i,j in enumerate(parking)};cp={j:orig+len(parking)+i for i,j in enumerate(parking)};classes=d['fleet']['vehicle_classes'];pool={};enumend=start+min(25,a.time_limit*.5)
def make(vc,order,park=None):
 has=vc['trailer_type_id']!=0
 if park is None:events=[('depot',base)]+[('NL' if i in NL else 'NH',i) for i in order]+[('depot',base)]
 else:
  left=next(k for k,i in enumerate(order) if i in NL);right=max(k for k,i in enumerate(order) if i in NL)
  events=[('depot',base)]+[('NH',i) for i in order[:left]]+[('DC',park)]+[('NL' if i in NL else 'NH',i) for i in order[left:right+1]]+[('CP',park)]+[('NH',i) for i in order[right+1:]]+[('depot',base)]
 truck=trailer=clock=0;attached=has;cost=vc['truck_fixed_cost'];arr={};seq=[];phys=[];prev=None;ptype=None
 for typ,i in events:
  nid=dc[i] if typ=='DC' else cp[i] if typ=='CP' else i
  if prev is not None:
   svc=customers[prev].get('service_time',P['service_time']) if ptype in ('NL','NH') else P['decoupling_time'] if ptype=='DC' else P['coupling_time'] if ptype=='CP' else 0
   clock+=TT[prev][i]+svc
   towing=has and (seq[-1]==base or seq[-1] in cp.values() or seq[-1] in NH) and (nid==base or nid in NH or nid in dc.values());cost+=D[prev][i]*(vc['truck_cost_factor']+(vc['trailer_towing_cost_multiplier'] if towing else 0))
  if typ in ('NL','NH'):
   c=customers[i];tw=c.get('time_window')
   if tw:
    clock=max(clock,tw[0])
    if clock>tw[1]+1e-8:return
   supply=c['supply']
   if typ=='NH' and attached:
    transferred=min(supply,vc['trailer_capacity']-trailer);trailer+=transferred;truck+=supply-transferred
   else:truck+=supply
  elif typ=='DC':attached=False;cost+=P['decoupling_cost']
  elif typ=='CP':attached=True;transferred=min(truck,vc['trailer_capacity']-trailer);truck-=transferred;trailer+=transferred;cost+=P['coupling_cost']
  if truck>vc['truck_capacity']+1e-8 or trailer>vc['trailer_capacity']+1e-8:return
  if clock>P['planning_horizon'][1] or clock>P['max_route_duration']:return
  if str(nid) not in arr:arr[str(nid)]=clock
  seq.append(nid);phys.append(i);prev=i;ptype=typ
 return {'vehicle_class':vc['class_id'],'customers_served':list(order),'route_sequence':seq,'route_sequence_physical':phys,'decouple_locations':[] if park is None else [park],'couple_locations':[] if park is None else [park],'arrival_times':arr,'route_cost':cost}
for size in range(1,len(J)+1):
 for subset in itertools.combinations(J,size):
  for vc in classes:
   key=(vc['class_id'],subset)
   for order in itertools.permutations(subset):
    if time.monotonic()>enumend:break
    parks=parking if vc['trailer_type_id'] and any(i in NL for i in order) else [None]
    for park in parks:
     r=make(vc,order,park)
     if r and (key not in pool or r['route_cost']<pool[key]['route_cost']):pool[key]=r
routes=list(pool.values());R=range(len(routes));m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
x=m.addVars(R,vtype=GRB.BINARY)
for i in J:m.addConstr(gp.quicksum(x[r] for r in R if i in routes[r]['customers_served'])==1)
vcmap={c['class_id']:c for c in classes}
for q in d['fleet']['truck_types']:m.addConstr(gp.quicksum(x[r] for r in R if vcmap[routes[r]['vehicle_class']]['truck_type_id']==q['id'])<=q['num_available'])
for q in d['fleet']['trailer_types']:m.addConstr(gp.quicksum(x[r] for r in R if vcmap[routes[r]['vehicle_class']]['trailer_type_id']==q['id'])<=q['num_available'])
m.setObjective(gp.quicksum(routes[r]['route_cost']*x[r] for r in R))
def pack(v):
 rs=[dict(routes[r]) for r in R if v(x[r])>.5];counts={}
 for r in rs:
  k=r['vehicle_class'];r['vehicle_index']=counts.get(k,0);counts[k]=counts.get(k,0)+1
 return {'objective_value':sum(r['route_cost'] for r in rs),'routes':rs}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible truck/trailer pattern cover')
s=pack(lambda q:q.X)
if log:log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

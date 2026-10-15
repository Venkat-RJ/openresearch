import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,math
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
def small_routes(d,limit):
 import itertools,time,numpy as np
 from scipy.optimize import milp,Bounds,LinearConstraint
 from scipy.sparse import coo_matrix
 start=time.monotonic();customers=d['truck_customers']+d['trailer_customers'];n=len(customers);ids=[c['id'] for c in customers];detail={c['id']:c for c in customers};nl={c['id'] for c in d['truck_customers']};nh=set(ids)-nl;D=d['distance_matrix'];TT=d['travel_time_matrix'];par=d['parameters'];depot=d['depot']['id'];parking=sorted(nh|{c['id'] for c in d['transshipment_locations']});N=len(D);P=len(parking);candidates=[]
 for vc in d['fleet']['vehicle_classes']:
  attached=vc['trailer_type_id']!=0
  for size in range(1,n+1):
   for subset in itertools.combinations(ids,size):
    if sum(detail[v]['supply'] for v in subset)>vc['total_capacity']:continue
    best=None
    for order in itertools.permutations(subset):
     if time.monotonic()-start>limit*.65:break
     truckpos=[i for i,v in enumerate(order) if v in nl]
     if attached and truckpos and truckpos!=list(range(min(truckpos),max(truckpos)+1)):continue
     parks=parking if attached and truckpos else [None]
     for park in parks:
      seq=[depot]+list(order)+[depot]
      if park is not None:
       j=parking.index(park);seq.insert(max(truckpos)+2,N+P+j);seq.insert(min(truckpos)+1,N+j)
      phys=[v if v<N else parking[(v-N)%P] for v in seq];truck=trailer=0.;on=attached;t=par['planning_horizon'][0];cost=vc['truck_fixed_cost'];times={str(depot):t};ok=True
      for k,(u,v) in enumerate(zip(seq,seq[1:])):
       pu,pv=phys[k:k+2];t+=TT[pu][pv];towing=attached and (u==depot or u in nh or N+P<=u<N+2*P) and (v==depot or v in nh or N<=v<N+P);cost+=D[pu][pv]*(vc['truck_cost_factor']+(vc['trailer_towing_cost_multiplier'] if towing else 0))
       if v in detail:
        tw=detail[v].get('time_window',par['planning_horizon']);t=max(t,tw[0]);ok &= t<=tw[1]+1e-8;times[str(v)]=t;q=detail[v]['supply']
        if on:
         qh=min(q,vc['trailer_capacity']-trailer);trailer+=qh;truck+=q-qh
        else:truck+=q
        t+=detail[v]['service_time']
       elif N<=v<N+P:
        times[str(v)]=t;on=False;t+=par['decoupling_time'];cost+=par['decoupling_cost']
       elif N+P<=v<N+2*P:
        times[str(v)]=t;on=True;transfer=min(truck,vc['trailer_capacity']-trailer);truck-=transfer;trailer+=transfer;t+=par['coupling_time'];cost+=par['coupling_cost']
       ok &= truck<=vc['truck_capacity']+1e-8 and trailer<=vc['trailer_capacity']+1e-8
      ok &= t-par['planning_horizon'][0]<=par['max_route_duration']+1e-8 and t<=par['planning_horizon'][1]+1e-8
      if ok and (best is None or cost<best[0]):best=(cost,{'vehicle_class':vc['class_id'],'vehicle_index':0,'customers_served':list(subset),'route_sequence':seq,'route_sequence_physical':phys,'decouple_locations':[] if park is None else [park],'couple_locations':[] if park is None else [park],'arrival_times':times})
    if best:candidates.append((vc,subset,*best))
 trucktypes=d['fleet']['truck_types'];trailertypes=d['fleet']['trailer_types'];rows=[];cols=[];vals=[];lo=[1.]*n+[0.]*(len(trucktypes)+len(trailertypes));hi=[1.]*n+[z['num_available'] for z in trucktypes+trailertypes]
 for j,(vc,subset,cost,route) in enumerate(candidates):
  rr=[ids.index(v) for v in subset]+[n+i for i,z in enumerate(trucktypes) if z['id']==vc['truck_type_id']]+[n+len(trucktypes)+i for i,z in enumerate(trailertypes) if z['id']==vc['trailer_type_id']]
  for i in rr:rows.append(i);cols.append(j);vals.append(1.)
 A=coo_matrix((vals,(rows,cols)),shape=(len(lo),len(candidates))).tocsc();res=milp(np.array([z[2] for z in candidates]),integrality=np.ones(len(candidates)),bounds=Bounds(0,1),constraints=LinearConstraint(A,lo,hi),options={'time_limit':max(.1,limit-(time.monotonic()-start)),'threads':1})
 if res.x is None:return None
 chosen=[z for z,x in zip(candidates,res.x) if x>.5]
 for i,z in enumerate(chosen):z[3]['vehicle_index']=i
 return {'objective_value':sum(z[2] for z in chosen),'routes':[z[3] for z in chosen]}
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path))
 if len(d['truck_customers'])+len(d['trailer_customers'])<=8:
  out=small_routes(d,max(1.,float(a.time_limit)-3))
  if out is not None:
   json.dump(out,open(a.solution_path,'w'))
   if a.log_path:
    from solution_logger import SolutionLogger
    SolutionLogger(a.log_path,sense='minimize').log(out['objective_value'])
   return
 cust=d['truck_customers']+d['trailer_customers'];depot=d['depot']['id'];ids=[depot]+[z['id'] for z in cust];details={z['id']:z for z in cust};classes=[z for z in d['fleet']['vehicle_classes'] if z['trailer_type_id']==0];vehicles=[z for z in classes for _ in range(z['num_available'])];K=len(vehicles);scale=10000;D=d['distance_matrix'];TT=d['travel_time_matrix'];par=d['parameters'];manager=pywrapcp.RoutingIndexManager(len(ids),K,0);r=pywrapcp.RoutingModel(manager)
 for k,vc in enumerate(vehicles):
  cb=r.RegisterTransitCallback(lambda i,j,vc=vc:round(D[ids[manager.IndexToNode(i)]][ids[manager.IndexToNode(j)]]*vc['truck_cost_factor']*scale));r.SetArcCostEvaluatorOfVehicle(cb,k);r.SetFixedCostOfVehicle(round(vc['truck_fixed_cost']*scale),k)
 dc=r.RegisterUnaryTransitCallback(lambda i:details[ids[manager.IndexToNode(i)]]['supply'] if manager.IndexToNode(i) else 0);r.AddDimensionWithVehicleCapacity(dc,0,[z['truck_capacity'] for z in vehicles],True,'Capacity')
 def travel(i,j):
  u=ids[manager.IndexToNode(i)];v=ids[manager.IndexToNode(j)];return math.ceil(scale*(TT[u][v]+details[u]['service_time'] if u in details else TT[u][v])-1e-9)
 tc=r.RegisterTransitCallback(travel);H=round(par['planning_horizon'][1]*scale);r.AddDimension(tc,H,H,False,'Time');td=r.GetDimensionOrDie('Time')
 for i in range(1,len(ids)):
  tw=details[ids[i]].get('time_window',par['planning_horizon']);td.CumulVar(manager.NodeToIndex(i)).SetRange(math.ceil(tw[0]*scale),math.floor(tw[1]*scale))
 for k in range(K):td.CumulVar(r.Start(k)).SetRange(round(par['planning_horizon'][0]*scale),H);r.solver().Add(td.CumulVar(r.End(k))-td.CumulVar(r.Start(k))<=round(par['max_route_duration']*scale))
 params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.seconds=max(1,int(float(a.time_limit))-3);sol=r.SolveWithParameters(params)
 if not sol:raise RuntimeError('Solo-truck fleet could not cover customer loads within time limits')
 routes=[];obj=0.
 for k,vc in enumerate(vehicles):
  i=r.Start(k);seq=[depot];times={str(depot):sol.Value(td.CumulVar(i))/scale};cost=vc['truck_fixed_cost']
  while not r.IsEnd(i):j=sol.Value(r.NextVar(i));u=ids[manager.IndexToNode(i)];v=ids[manager.IndexToNode(j)];cost+=vc['truck_cost_factor']*D[u][v];seq.append(v);i=j
  if len(seq)>2:
   for v in seq[1:-1]:times[str(v)]=sol.Value(td.CumulVar(manager.NodeToIndex(ids.index(v))))/scale
   routes.append({'vehicle_class':vc['class_id'],'vehicle_index':k,'customers_served':seq[1:-1],'route_sequence':seq,'route_sequence_physical':seq,'decouple_locations':[],'couple_locations':[],'arrival_times':times,'route_cost':cost});obj+=cost
 json.dump({'objective_value':obj,'routes':routes},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(obj)
if __name__=='__main__':main()

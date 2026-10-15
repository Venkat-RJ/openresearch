import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,math

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();deadline=time.monotonic()+max(.1,float(a.time_limit)-2);d=json.load(open(a.instance_path));N=d['num_customers'];Q=d['vehicle_capacity'];D=d['distance_matrix'];customers={v['id']:v for v in d['customers']};end=N+1
 def cost(row):return sum(D[i if i!=end else 0][j if j!=end else 0]for i,j in zip([0]+row,row+[0]))
 def feasible(row):
  t=d['depot']['time_window'][0];prev=0
  for j in row:
   t+=D[prev][j]+(customers[prev]['service_time']if prev else 0);t=max(t,customers[j]['time_window'][0])
   if t>customers[j]['time_window'][1]+1e-8:return False
   prev=j
  return t+(customers[prev]['service_time']if prev else 0)+D[prev][0]<=d['depot']['time_window'][1]+1e-8
 routes=[];remaining={j:v['demand']for j,v in customers.items()}
 while any(remaining.values()):
  row=[];deliver={};space=Q
  while space and time.monotonic()<deadline:
   candidates=[]
   for j,q in remaining.items():
    if q<=0 or j in row:continue
    for pos in range(len(row)+1):
     candidate=row[:pos]+[j]+row[pos:]
     if feasible(candidate):candidates.append((cost(candidate)-cost(row),j,candidate))
   if not candidates:break
   _,j,row=min(candidates,key=lambda v:v[0]);q=min(space,remaining[j]);remaining[j]-=q;space-=q;deliver[str(j)]=q
  if not row:
   j=next(j for j in remaining if remaining[j]);row=[j];q=min(Q,remaining[j]);remaining[j]-=q;deliver={str(j):q}
  routes.append({'vehicle':len(routes),'route':[0]+row+[end],'deliveries':deliver})
 if N<=40 and time.monotonic()<deadline:
  import numpy as np
  from scipy.optimize import milp,Bounds,LinearConstraint
  from scipy.sparse import coo_matrix
  columns={tuple(v['route'][1:-1]):cost(v['route'][1:-1])for v in routes};generation_end=min(deadline-30,time.monotonic()+15);limit=50000
  def extend(row,clock,load):
   if time.monotonic()>generation_end or len(columns)>=limit:return
   previous=row[-1]if row else 0
   choices=sorted((j for j in customers if j not in row),key=lambda j:customers[j]['time_window'][1])
   for j in choices:
    arrival=max(clock+D[previous][j]+(customers[previous]['service_time']if previous else 0),customers[j]['time_window'][0])
    if arrival>customers[j]['time_window'][1]+1e-8:continue
    if arrival+customers[j]['service_time']+D[j][0]>d['depot']['time_window'][1]+1e-8:continue
    candidate=row+(j,);columns[candidate]=cost(list(candidate))
    if len(candidate)<9 and load+customers[j]['demand']<2*Q:extend(candidate,arrival,load+customers[j]['demand'])
  extend((),d['depot']['time_window'][0],0);cols=list(columns);R=len(cols);c=[columns[row]for row in cols];lb=[0]*R;ub=[1]*R;integrality=[1]*R;ri=[];ci=[];data=[];lower=[];upper=[];quantities=[];customer_vars={j:[]for j in customers}
  def constraint(entries,lo,hi):
   r=len(lower);lower.append(lo);upper.append(hi)
   for k,val in entries:ri.append(r);ci.append(k);data.append(val)
  for r,row in enumerate(cols):
   qrow=[]
   for j in row:
    k=len(c);c.append(0);lb.append(0);ub.append(customers[j]['demand']);integrality.append(0);quantities.append((r,j,k));customer_vars[j].append(k);qrow.append((k,1));constraint([(k,1),(r,-customers[j]['demand'])],-np.inf,0)
   constraint(qrow+[(r,-Q)],-np.inf,0)
  for j,indices in customer_vars.items():constraint([(k,1)for k in indices],customers[j]['demand'],customers[j]['demand'])
  constraint([(r,1)for r in range(R)],math.ceil(sum(v['demand']for v in customers.values())/Q),sum(math.ceil(v['demand']/Q)for v in customers.values()))
  A=coo_matrix((data,(ri,ci)),shape=(len(lower),len(c))).tocsc();result=milp(c,integrality=integrality,bounds=Bounds(lb,ub),constraints=LinearConstraint(A,lower,upper),options={'time_limit':max(.1,deadline-time.monotonic()),'threads':1,'mip_rel_gap':.005})
  if result.x is not None:
   delivery={r:{}for r in range(R)if result.x[r]>.5}
   for r,j,k in quantities:
    if r in delivery and result.x[k]>1e-7:delivery[r][str(j)]=float(result.x[k])
   candidate=[{'vehicle':i,'route':[0]+[j for j in cols[r]if str(j)in delivery[r]]+[end],'deliveries':delivery[r]}for i,r in enumerate(delivery)]
   if sum(cost(v['route'][1:-1])for v in candidate)<sum(cost(v['route'][1:-1])for v in routes):routes=candidate
 if all(v['demand']<=Q for v in customers.values()) and time.monotonic()<deadline:
  from ortools.constraint_solver import pywrapcp,routing_enums_pb2
  K=sum(math.ceil(v['demand']/Q)for v in customers.values());manager=pywrapcp.RoutingIndexManager(N+1,K,0);routing=pywrapcp.RoutingModel(manager);scale=1000
  cb=routing.RegisterTransitCallback(lambda i,j:int(round(scale*D[manager.IndexToNode(i)][manager.IndexToNode(j)])));routing.SetArcCostEvaluatorOfAllVehicles(cb)
  tm=routing.RegisterTransitCallback(lambda i,j:int(round(scale*(D[manager.IndexToNode(i)][manager.IndexToNode(j)]+(customers[manager.IndexToNode(i)]['service_time']if manager.IndexToNode(i)else 0)))));H=int(scale*d['depot']['time_window'][1]);routing.AddDimension(tm,H,H,False,'Time');times=routing.GetDimensionOrDie('Time')
  for j,v in customers.items():times.CumulVar(manager.NodeToIndex(j)).SetRange(int(scale*v['time_window'][0]),int(scale*v['time_window'][1]))
  for k in range(K):times.CumulVar(routing.Start(k)).SetRange(int(scale*d['depot']['time_window'][0]),H);times.CumulVar(routing.End(k)).SetRange(int(scale*d['depot']['time_window'][0]),H)
  loadcb=routing.RegisterUnaryTransitCallback(lambda i:customers[manager.IndexToNode(i)]['demand']if manager.IndexToNode(i)else 0);routing.AddDimensionWithVehicleCapacity(loadcb,0,[Q]*K,True,'Capacity');params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.FromMilliseconds(max(1,int((deadline-time.monotonic())*1000)));sol=routing.SolveWithParameters(params)
  if sol is not None:
   candidate=[]
   for k in range(K):
    index=routing.Start(k);row=[]
    while not routing.IsEnd(index):
     j=manager.IndexToNode(index)
     if j:row.append(j)
     index=sol.Value(routing.NextVar(index))
    if row:candidate.append({'vehicle':k,'route':[0]+row+[end],'deliveries':{str(j):customers[j]['demand']for j in row}})
   if sum(cost(v['route'][1:-1])for v in candidate)<sum(cost(v['route'][1:-1])for v in routes):routes=candidate
 objective=sum(cost(v['route'][1:-1])for v in routes);json.dump({'objective_value':objective,'num_vehicles_used':len(routes),'routes':routes,'deliveries':{str(j):{'demand':v['demand'],'total_delivered':v['demand']}for j,v in customers.items()}},open(a.solution_path,'w'))
if __name__=='__main__':main()

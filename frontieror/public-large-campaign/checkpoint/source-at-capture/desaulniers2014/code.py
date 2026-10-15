import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model
from ortools.constraint_solver import pywrapcp,routing_enums_pb2

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));old='depot'in d;n=d['num_customers'];T=d['num_periods'];K=d['num_vehicles'];Q=d['vehicle_capacity'];D=d['distance_matrix']
 if old:
  depot=d['depot'];customers=sorted(d['customers'],key=lambda v:v['id']);initial=[depot['initial_inventory']]+[v['initial_inventory']for v in customers];caps=[depot['max_inventory']]+[v['max_inventory']for v in customers];holding=[depot['holding_cost']]+[v['holding_cost']for v in customers];demand=[[v['demand_per_period']]*T for v in customers];production=[depot['production_per_period']]*T;minimum=[0]+[v.get('min_inventory',0)for v in customers]
 else:initial=d['initial_inventories'];caps=d['inventory_capacities'];holding=d['holding_costs'];demand=d['customer_demands'];production=d['supplier_production'];minimum=[0]*(n+1)
 if n>12:
  deadline=start+max(.1,float(a.time_limit)-2);plans=[];current=initial[:];jit=[];jit_ok=True
  for t in range(T):
   quantities={i:max(0,demand[i-1][t]+minimum[i]-current[i])for i in range(1,n+1)}
   if any(current[i]+quantities[i]>caps[i]or quantities[i]>Q for i in quantities):jit_ok=False
   current[0]+=production[t]-sum(quantities.values())
   if not minimum[0]<=current[0]<=caps[0]:jit_ok=False
   for i in quantities:current[i]+=quantities[i]-demand[i-1][t]
   jit.append(quantities)
  if jit_ok:plans.append(jit)
  for target in range(2,K+1):
   for geographic,co_delivery in((False,False),(True,False),(False,True),(True,True)):
    current=initial[:];plan=[];valid=True
    for t in range(T):
     quantities={i:max(0,demand[i-1][t]+minimum[i]-current[i])for i in range(1,n+1)};required=sum(quantities.values());available=current[0]+production[t];budget=min(available,max(required,target*Q))-required;later=[i for i in quantities if quantities[i]==0];offers=[]
     for i in quantities:
      if not quantities[i]:continue
      desired=min(Q,caps[i]-current[i],max(0,sum(demand[i-1][t:])+minimum[i]-current[i]));extra=desired-quantities[i]
      if extra>0:
       saving=min((D[i][j]for j in later),default=D[0][i])if geographic else D[0][i];offers.append((saving/max(1,extra),i,extra))
     for score,i,extra in sorted(offers,reverse=True):
      if extra<=budget:quantities[i]+=extra;budget-=extra
     if required and co_delivery:
      visited=[i for i in quantities if quantities[i]];optional=[]
      for i in later:
       needed=max(0,sum(demand[i-1][t:])+minimum[i]-current[i])
       if needed<=0 or needed>min(Q,caps[i]-current[i]):continue
       insertion=min((D[j][i]+D[i][0]-D[j][0]for j in visited),default=2*D[0][i]);saving=2*D[0][i]-max(0,insertion);priority=saving/max(1,needed)if geographic else -insertion/max(1,needed);optional.append((priority,i,needed))
      for score,i,needed in sorted(optional,reverse=True):
       if needed<=budget:quantities[i]=needed;budget-=needed
     if sum(quantities.values())>K*Q or sum(quantities.values())>available or any(current[i]+quantities[i]>caps[i]or quantities[i]>Q for i in quantities):valid=False;break
     current[0]=available-sum(quantities.values())
     if not minimum[0]<=current[0]<=caps[0]:valid=False;break
     for i in quantities:current[i]+=quantities[i]-demand[i-1][t]
     plan.append(quantities)
    if valid:plans.append(plan)
  for factor in(.8,):
   if deadline-time.monotonic()<5:break
   schedule=cp_model.CpModel();stock={};quantity={};active={};cost=[]
   for t in range(T):
    for i in range(n+1):stock[i,t]=schedule.NewIntVar(minimum[i],caps[i],'I%d_%d'%(i,t));cost.append(int(round(holding[i]*10000))*stock[i,t])
    for i in range(1,n+1):
     quantity[i,t]=schedule.NewIntVar(0,min(Q,caps[i]),'q%d_%d'%(i,t));active[i,t]=schedule.NewBoolVar('v%d_%d'%(i,t));schedule.Add(quantity[i,t]<=min(Q,caps[i])*active[i,t]);schedule.Add(quantity[i,t]>=active[i,t]);previous=initial[i]if t==0 else stock[i,t-1];schedule.Add(stock[i,t]==previous+quantity[i,t]-demand[i-1][t]);schedule.Add(previous+quantity[i,t]<=caps[i]);cost.append(int(round(factor*2*D[0][i]*10000))*active[i,t])
    previous=initial[0]if t==0 else stock[0,t-1];schedule.Add(stock[0,t]==previous+production[t]-sum(quantity[i,t]for i in range(1,n+1)));schedule.Add(sum(quantity[i,t]for i in range(1,n+1))<=K*Q)
   if jit_ok:
    current=initial[:]
    for t in range(T):
     current[0]+=production[t]-sum(jit[t].values())
     for i in range(1,n+1):current[i]+=jit[t][i]-demand[i-1][t];schedule.AddHint(quantity[i,t],jit[t][i]);schedule.AddHint(active[i,t],int(jit[t][i]>0))
     for i in range(n+1):schedule.AddHint(stock[i,t],current[i])
   schedule.Minimize(sum(cost));ss=cp_model.CpSolver();ss.parameters.num_search_workers=1;ss.parameters.max_time_in_seconds=min(3,max(.1,deadline-time.monotonic()-2));status=ss.Solve(schedule)
   if status in(cp_model.FEASIBLE,cp_model.OPTIMAL):plans.append([{i:ss.Value(quantity[i,t])for i in range(1,n+1)}for t in range(T)])
  unique={}
  for plan in plans:unique[tuple(plan[t][i]for t in range(T)for i in range(1,n+1))]=plan
  plans=list(unique.values());best=None;route_pool={}
  def add_route(row):
   key=frozenset(row[1:-1]);cost=sum(D[i][j]for i,j in zip(row,row[1:]))
   if key and(key not in route_pool or cost<route_pool[key][1]):route_pool[key]=(row[:],cost)
  for plan_index,plan in enumerate(plans):
   current=initial[:];period_data={};all_routes={};all_inventory={str(i):{}for i in range(n+1)};value=0;valid=True
   for t,quantities in enumerate(plan):
    customers=[i for i in range(1,n+1)if quantities[i]>0];bins=[];loads=[]
    for i in sorted(customers,key=lambda i:quantities[i],reverse=True):
     choices=[j for j in range(len(bins))if loads[j]+quantities[i]<=Q]
     if choices:j=max(choices,key=lambda j:loads[j]);bins[j].append(i);loads[j]+=quantities[i]
     else:bins.append([i]);loads.append(quantities[i])
    rows=[]
    if len(bins)<=K:
     for group in bins:
      row=[0];remaining=set(group)
      while remaining:i=min(remaining,key=lambda i:D[row[-1]][i]);remaining.remove(i);row.append(i)
      row.append(0);improved=True
      while improved:
       improved=False
       for left in range(1,len(row)-2):
        for right in range(left+1,len(row)-1):
         proposed=row[:left]+list(reversed(row[left:right+1]))+row[right+1:]
         if sum(D[i][j]for i,j in zip(proposed,proposed[1:]))<sum(D[i][j]for i,j in zip(row,row[1:])):row=proposed;improved=True
      rows.append(row)
    available=deadline-time.monotonic()
    if customers and available>.2:
     nodes=[0]+customers;manager=pywrapcp.RoutingIndexManager(len(nodes),K,0);routing=pywrapcp.RoutingModel(manager);callback=routing.RegisterTransitCallback(lambda i,j:int(round(10000*D[nodes[manager.IndexToNode(i)]][nodes[manager.IndexToNode(j)]])));routing.SetArcCostEvaluatorOfAllVehicles(callback);capacity_callback=routing.RegisterUnaryTransitCallback(lambda i:quantities.get(nodes[manager.IndexToNode(i)],0));routing.AddDimensionWithVehicleCapacity(capacity_callback,0,[Q]*K,True,'capacity');parameters=pywrapcp.DefaultRoutingSearchParameters();parameters.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;parameters.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;remaining_routes=sum(any(q>0 for q in quantities.values())for pi,future in enumerate(plans)if pi>=plan_index for tt,quantities in enumerate(future)if pi>plan_index or tt>=t);seconds=min(4 if sum(quantities.values())>.85*K*Q else 1.5,available/max(1,remaining_routes));parameters.time_limit.FromMilliseconds(max(1,int(seconds*1000)))
     assignment=None
     if rows:
      node_index={v:i for i,v in enumerate(nodes)};warm=[[node_index[i]for i in row[1:-1]]for row in rows]+[[]for _ in range(K-len(rows))];assignment=routing.ReadAssignmentFromRoutes(warm,True)
     result=routing.SolveFromAssignmentWithParameters(assignment,parameters)if assignment is not None else routing.SolveWithParameters(parameters)
     if result is not None:
      routed=[]
      for k in range(K):
       index=routing.Start(k);row=[]
       while not routing.IsEnd(index):row.append(nodes[manager.IndexToNode(index)]);index=result.Value(routing.NextVar(index))
       row.append(0)
       if len(row)>2:routed.append(row)
      if not rows or sum(D[i][j]for row in routed for i,j in zip(row,row[1:]))<sum(D[i][j]for row in rows for i,j in zip(row,row[1:])):rows=routed
    if customers and not rows:valid=False;break
    for row in rows:add_route(row)
    current[0]+=production[t]-sum(quantities.values());deliveries={str(i):q for i,q in quantities.items()if q};route_map={str(k):{'route':row,'deliveries':{str(i):quantities[i]for i in row[1:-1]}}for k,row in enumerate(rows)}
    for i in range(1,n+1):current[i]+=quantities[i]-demand[i-1][t]
    value+=sum(D[i][j]for row in rows for i,j in zip(row,row[1:]))+sum(holding[i]*current[i]for i in range(n+1));period_data[str(t+1)]={'routes':rows,'deliveries':deliveries,'inventories':{('depot'if i==0 else str(i)):current[i]for i in range(n+1)}};all_routes[str(t+1)]=route_map
    for i in range(n+1):all_inventory[str(i)][str(t+1)]=current[i]
   if valid and(best is None or value<best[0]):best=(value,period_data,all_routes,all_inventory)
  if best is None:raise RuntimeError('No incumbent found in decomposed inventory and capacitated route construction within the time limit')
  if deadline-time.monotonic()>8:
   original=list(route_pool.values())
   for row,cost in original:
    for position in range(1,len(row)-1):add_route(row[:position]+row[position+1:])
    for length in range(2,len(row)-1):add_route([0]+row[1:length+1]+[0]);add_route([0]+row[-length-1:-1]+[0])
   for seed in range(1,n+1):
    neighbors=sorted(range(1,n+1),key=lambda i:D[seed][i])
    for size in(1,3,5,8,12,16):
     remaining=set(neighbors[:min(size,n)]);row=[0]
     while remaining:i=min(remaining,key=lambda i:D[row[-1]][i]);remaining.remove(i);row.append(i)
     row.append(0);add_route(row)
   columns=list(route_pool.values());joint=cp_model.CpModel();select={};deliver={};stock={};objective=[];incidence={i:[]for i in range(1,n+1)}
   for z,(row,cost)in enumerate(columns):
    for i in row[1:-1]:incidence[i].append(z)
   for t in range(T):
    for i in range(n+1):stock[i,t]=joint.NewIntVar(minimum[i],caps[i],'I%d_%d'%(i,t));objective.append(int(round(holding[i]*10000))*stock[i,t])
    for z,(row,cost)in enumerate(columns):
     select[t,z]=joint.NewBoolVar('route%d_%d'%(t,z));objective.append(int(round(cost*10000))*select[t,z]);load=[]
     for i in row[1:-1]:
      deliver[t,z,i]=joint.NewIntVar(0,min(Q,caps[i]),'q%d_%d_%d'%(t,z,i));joint.Add(deliver[t,z,i]<=min(Q,caps[i])*select[t,z]);load.append(deliver[t,z,i])
     joint.Add(sum(load)<=Q*select[t,z])
    joint.Add(sum(select[t,z]for z in range(len(columns)))<=K)
    for i in range(1,n+1):
     received=sum(deliver[t,z,i]for z in incidence[i]);previous=initial[i]if t==0 else stock[i,t-1];joint.Add(stock[i,t]==previous+received-demand[i-1][t]);joint.Add(previous+received<=caps[i]);joint.Add(sum(select[t,z]for z in incidence[i])<=1)
    previous=initial[0]if t==0 else stock[0,t-1];joint.Add(stock[0,t]==previous+production[t]-sum(v for(tt,z,i),v in deliver.items()if tt==t))
   keys={frozenset(row[1:-1]):z for z,(row,cost)in enumerate(columns)};hint_routes={};hint_quantity={}
   for t in range(T):
    for record in best[2][str(t+1)].values():
     z=keys[frozenset(record['route'][1:-1])];hint_routes[t,z]=1
     for i,q in record['deliveries'].items():hint_quantity[t,z,int(i)]=q
    for i in range(n+1):joint.AddHint(stock[i,t],best[3][str(i)][str(t+1)])
   for key,var in select.items():joint.AddHint(var,hint_routes.get(key,0))
   for key,var in deliver.items():joint.AddHint(var,hint_quantity.get(key,0))
   joint.Add(sum(objective)<=int(round(best[0]*10000)));joint.Minimize(sum(objective));js=cp_model.CpSolver();js.parameters.num_search_workers=1;js.parameters.max_time_in_seconds=max(.1,deadline-time.monotonic());status=js.Solve(joint)
   if status in(cp_model.FEASIBLE,cp_model.OPTIMAL):
    period_data={};all_routes={};all_inventory={str(i):{}for i in range(n+1)};value=0
    for t in range(T):
     rows=[];quantities={};route_map={}
     for z,(row,cost)in enumerate(columns):
      if not js.Value(select[t,z]):continue
      amounts={str(i):js.Value(deliver[t,z,i])for i in row[1:-1]if js.Value(deliver[t,z,i])};reduced=[0]+[i for i in row[1:-1]if str(i)in amounts]+[0]
      if not amounts:continue
      route_map[str(len(rows))]={'route':reduced,'deliveries':amounts};rows.append(reduced);quantities.update(amounts);value+=sum(D[i][j]for i,j in zip(reduced,reduced[1:]))
     inventory={('depot'if i==0 else str(i)):js.Value(stock[i,t])for i in range(n+1)};period_data[str(t+1)]={'routes':rows,'deliveries':quantities,'inventories':inventory};all_routes[str(t+1)]=route_map
     for i in range(n+1):all_inventory[str(i)][str(t+1)]=js.Value(stock[i,t]);value+=holding[i]*js.Value(stock[i,t])
    if value<best[0]:best=value,period_data,all_routes,all_inventory
  value,period_data,all_routes,all_inventory=best;out={'objective_value':value,'solution_details':{'periods':period_data}}if old else {'objective_value':value,'routes':all_routes,'inventories':all_inventory};json.dump(out,open(a.solution_path,'w'));return
 m=cp_model.CpModel();inventory=[[m.NewIntVar(minimum[i],caps[i],'I%d_%d'%(i,t))for t in range(T)]for i in range(n+1)];delivery={};visits={};arcs={};obj=[]
 for t in range(T):
  for i in range(n+1):obj.append(int(round(holding[i]*10000))*inventory[i][t])
  for k in range(K):
   use=m.NewBoolVar('use%d_%d'%(t,k));vv={i:m.NewBoolVar('v%d_%d_%d'%(t,k,i))for i in range(1,n+1)};ff={};aa={}
   for i in range(n+1):
    for j in range(n+1):
     if i==j:continue
     x=m.NewBoolVar('x%d_%d_%d_%d'%(t,k,i,j));f=m.NewIntVar(0,Q,'f%d_%d_%d_%d'%(t,k,i,j));m.Add(f<=Q*x);aa[i,j]=x;ff[i,j]=f;obj.append(D[i][j]*10000*x)
   m.Add(sum(aa[0,j]for j in range(1,n+1))==use);m.Add(sum(aa[j,0]for j in range(1,n+1))==use)
   for i in range(1,n+1):
    q=m.NewIntVar(0,min(Q,caps[i]),'q%d_%d_%d'%(t,k,i));delivery[t,k,i]=q;visits[t,k,i]=vv[i];m.Add(q<=Q*vv[i]);m.Add(q>=vv[i]);m.Add(sum(aa[i,j]for j in range(n+1)if j!=i)==vv[i]);m.Add(sum(aa[j,i]for j in range(n+1)if j!=i)==vv[i]);m.Add(sum(ff[j,i]for j in range(n+1)if j!=i)-sum(ff[i,j]for j in range(n+1)if j!=i)==q)
   m.Add(sum(delivery[t,k,i]for i in range(1,n+1))<=Q);m.Add(sum(ff[0,j]for j in range(1,n+1))-sum(ff[j,0]for j in range(1,n+1))==sum(delivery[t,k,i]for i in range(1,n+1)));arcs[t,k]=aa
   if k:m.Add(use<=sum(arcs[t,k-1][0,j]for j in range(1,n+1)))
  for i in range(1,n+1):
   q=sum(delivery[t,k,i]for k in range(K));prev=initial[i]if t==0 else inventory[i][t-1];m.Add(inventory[i][t]==prev+q-demand[i-1][t]);m.Add(prev+q<=caps[i]);m.Add(sum(visits[t,k,i]for k in range(K))<=1)
  prev=initial[0]if t==0 else inventory[0][t-1];m.Add(inventory[0][t]==prev+production[t]-sum(delivery[t,k,i]for k in range(K)for i in range(1,n+1)))
 m.Minimize(sum(obj));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,float(a.time_limit)-2-(time.monotonic()-start));status=s.Solve(m)
 if status not in(cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible inventory routing plan found')
 periods={};rr={};iv={str(i):{}for i in range(n+1)};value=0
 for t in range(T):
  routes=[];qd={};rmap={}
  for k in range(K):
   aa=arcs[t,k];nexts={i:j for(i,j),v in aa.items()if s.Value(v)}
   if 0 not in nexts:continue
   row=[0];j=nexts[0]
   while j!=0:row.append(j);j=nexts[j]
   row.append(0);quant={str(i):s.Value(delivery[t,k,i])for i in row[1:-1]};qd.update(quant);routes.append(row);rmap[str(k)]={'route':row,'deliveries':quant};value+=sum(D[i][j]for i,j in zip(row,row[1:]))
  inv={('depot'if i==0 else str(i)):s.Value(inventory[i][t])for i in range(n+1)}
  for i in range(n+1):v=s.Value(inventory[i][t]);iv[str(i)][str(t+1)]=v;value+=holding[i]*v
  periods[str(t+1)]={'routes':routes,'deliveries':qd,'inventories':inv};rr[str(t+1)]=rmap
 out={'objective_value':value,'solution_details':{'periods':periods}}if old else {'objective_value':value,'routes':rr,'inventories':iv};json.dump(out,open(a.solution_path,'w'))
if __name__=='__main__':main()

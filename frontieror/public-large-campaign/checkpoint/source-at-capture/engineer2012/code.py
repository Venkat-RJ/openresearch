import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();started=time.monotonic();d=json.load(open(a.instance_path));T=d['planning_horizon'];ports={v['port_id']:v for v in d['supply_ports']+d['demand_ports']};supply=set(v['port_id']for v in d['supply_ports']);m=cp_model.CpModel();obj=[];all_arcs={};operations={};inventories={};scale=10000
 for vessel in d['vessels']:
  vid=vessel['id'];Q=vessel['capacity'];start=vessel['availability_start'];end=vessel['availability_end'];src=('src',vid);snk=('snk',vid);network=[(p,t)for p in ports for t in range(start,end+1)];edges=[]
  for p in ports:edges.append((src,(p,start),'source',0))
  for p,t in network:
   edges.append(((p,t),snk,'sink',0))
   if t<end:edges.append(((p,t),(p,t+1),'demurrage',vessel['demurrage_cost_per_day']))
   for q in ports:
    if p==q:continue
    dt=d['travel_times'].get(p+'->'+q)
    if dt is not None and t+dt<=end:edges.append(((p,t),(q,t+dt),'travel',dt*vessel['transport_cost_per_day']))
  aa=[];incoming={node:[]for node in network+[src,snk]};outgoing={node:[]for node in network+[src,snk]}
  for z,(tail,head,typ,cost)in enumerate(edges):
   x=m.NewBoolVar('x%s_%d'%(vid,z));lim=Q
   for node in (tail,head):
    if node[0]in ports:lim=min(lim,d['draft_limits'].get(vid+'->'+node[0],Q))
   f=m.NewIntVar(0,lim,'f%s_%d'%(vid,z));m.Add(f<=lim*x)
   if typ=='source':m.Add(f==vessel['initial_load']*x)
   if typ=='sink':m.Add(f==0)
   aa.append((tail,head,typ,cost,x,f));incoming[head].append((x,f));outgoing[tail].append((x,f));obj.append(int(round(scale*cost))*x)
  m.Add(sum(x for x,f in outgoing[src])==1);m.Add(sum(x for x,f in incoming[snk])==1)
  for p,t in network:
   node=(p,t);m.Add(sum(x for x,f in incoming[node])==sum(x for x,f in outgoing[node]));visit=sum(x for x,f in outgoing[node]);m.Add(visit<=1);port=ports[p];z=m.NewBoolVar('op%s_%s_%d'%(vid,p,t));q=m.NewIntVar(0,port['F_max'],'q%s_%s_%d'%(vid,p,t));m.Add(q<=port['F_max']*z);m.Add(q>=port['F_min']*z);m.Add(z<=visit);sign=1 if p in supply else -1;m.Add(sum(f for x,f in outgoing[node])-sum(f for x,f in incoming[node])==sign*q);operations[vid,p,t]=q;unit=port['unit_procurement_cost']if p in supply else -port['unit_supply_revenue'];obj.append(int(round(scale*unit))*q)
  all_arcs[vid]=aa
 for p,port in ports.items():
  previous=port['initial_inventory']
  for t in range(1,T+1):
   inv=m.NewIntVar(0,port['storage_capacity'][t-1],'I%s_%d'%(p,t));q=sum(v for(vid,pp,tt),v in operations.items()if pp==p and tt==t);delta=port['production_rate'][t-1]if p in supply else -port['consumption_rate'][t-1];m.Add(inv==previous+delta-(q if p in supply else -q));inventories[p,t]=inv;previous=inv
 m.Minimize(sum(obj));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,float(a.time_limit)-2-(time.monotonic()-started));status=s.Solve(m)
 if status not in(cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible maritime inventory voyage plan found')
 routes={};opout={};transport=0.;demurrage=0.;procurement=0.;revenue=0.
 for vessel in d['vessels']:
  vid=vessel['id'];chosen={tail:(tail,head,typ,cost,f)for tail,head,typ,cost,x,f in all_arcs[vid]if s.Value(x)};row=[];node=('src',vid)
  while node!=('snk',vid):
   tail,head,typ,cost,f=chosen[node];row.append({'tail':str(tail),'head':str(head),'type':typ,'cost':cost,'product_flow':s.Value(f)});node=head
   if typ=='travel':transport+=cost
   if typ=='demurrage':demurrage+=cost
  routes[vid]=row;oo=[]
  for(v,p,t),q in operations.items():
   if v!=vid or not s.Value(q):continue
   amount=s.Value(q);sign=1 if p in supply else -1;oo.append({'port':p,'time':t,'quantity':sign*amount,'type':'load'if sign==1 else 'discharge'})
   if sign==1:procurement+=amount*ports[p]['unit_procurement_cost']
   else:revenue+=amount*ports[p]['unit_supply_revenue']
  opout[vid]=sorted(oo,key=lambda v:v['time'])
 total=transport+demurrage+procurement-revenue;json.dump({'objective_value':total,'vessel_routes':routes,'vessel_operations':opout,'port_inventories':{p:[{'time':t,'inventory':s.Value(inventories[p,t])}for t in range(1,T+1)]for p in ports},'cost_breakdown':{'transport_cost':transport,'demurrage_cost':demurrage,'procurement_cost':procurement,'supply_revenue':revenue,'total_cost':total}},open(a.solution_path,'w'))
if __name__=='__main__':main()

import argparse,json,os,time,random
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();begin=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));T=d['planning_horizon'];ports={p['port_id']:p for p in d['supply_ports']+d['demand_ports']};sup=[p['port_id'] for p in d['supply_ports']];dem=[p['port_id'] for p in d['demand_ports']];vessels={v['id']:v for v in d['vessels']};travel={tuple(k.split('->')):v for k,v in d['travel_times'].items()};rng=random.Random(0);paths={};visits={};ceiling={};costs={};keys=set()
 for vid,v in vessels.items():
  lo=v['availability_start'];hi=v['availability_end'];pool={};limit={j:min(v['capacity'],d['draft_limits'].get(vid+'->'+j,v['capacity'])) for j in ports}
  def add(seq):
   seq=tuple(seq)
   if seq in pool:return
   arcs=[(('src',vid),seq[0],'source',0.)];total=0
   for x,y in zip(seq,seq[1:]):
    typ='demurrage' if x[0]==y[0] else 'travel';cost=(v['demurrage_cost_per_day'] if typ=='demurrage' else v['transport_cost_per_day']*(y[1]-x[1]));arcs.append((x,y,typ,cost));total+=cost
   arcs.append((seq[-1],('snk',vid),'sink',0.));pool[seq]=(arcs,total)
  for j in ports:add([(j,lo)]);add([(j,t) for t in range(lo,hi+1)])
  for j in sup:
   for k in dem:
    tt=travel[j,k]
    for dep in range(max(lo,1),hi-tt+1):add([(j,t) for t in range(lo,dep+1)]+[(k,t) for t in range(dep+tt,hi+1)])
  attempts=0
  while len(pool)<220 and attempts<5000:
   attempts+=1;j=rng.choice(sup);seq=[(j,lo)];t=lo;cross=False
   while t<hi:
    if not cross and rng.random()<.20:dest=rng.choice(dem);cross=True
    elif rng.random()<.10:dest=rng.choice(dem if cross else sup)
    else:dest=j
    dt=1 if dest==j else travel.get((j,dest),hi+1)
    if t+dt>hi:dest=j;dt=1
    t+=dt;j=dest;seq.append((j,t))
   add(seq)
  for j in sup:
   for s in sup:
    if s==j:continue
    for k in dem:
     for h in dem:
      if h==k:continue
      for wait in (0,1,2,4):
       shift=max(lo,1)+wait;arr=shift+travel[j,s]
       for wait2 in (0,1,3):
        dep=arr+wait2;at=dep+travel[s,k];last=at+travel[k,h]
        if last<=hi:add([(j,t) for t in range(lo,shift+1)]+[(s,t) for t in range(arr,dep+1)]+[(k,at),(h,last)])
  originals=list(pool)
  for seq in originals:
   first=next((idx for idx,node in enumerate(seq) if node[0] in dem),None)
   if first is not None:
    for end in range(first,len(seq)):add(seq[:end+1])
  mandatory=[seq for seq in pool if len(seq)==1]+[seq for seq in originals if len({j for j,t in seq})==2 and seq[-1][1]==hi][:48];remaining=[seq for seq in pool if seq not in mandatory];rng.shuffle(remaining);selected=mandatory+remaining[:190-len(mandatory)]
  for r,seq in enumerate(selected):
   arcs,cost=pool[seq]
   paths[vid,r]=arcs;costs[vid,r]=cost;seen=set(seq);visits[vid,r]=seen
   for j,t in seen:
    if 1<=t<=T:keys.add((vid,j,t))
   bound={t:v['capacity'] for t in range(lo,hi+1)}
   for x,y,typ,_ in arcs:
    if typ in ('demurrage','travel'):
     for t in range(x[1],y[1]):bound[t]=min(bound[t],limit[x[0]],limit[y[0]])
    elif typ=='sink':bound[x[1]]=min(bound[x[1]],limit[x[0]])
   ceiling[vid,r]=bound
 model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.NumericFocus=0;model.Params.MIPGap=1e-4;y=model.addVars(list(paths),vtype=GRB.BINARY);f=model.addVars(sorted(keys),lb=0);z=model.addVars(sorted(keys),vtype=GRB.BINARY);cargo=model.addVars([(vid,t) for vid,v in vessels.items() for t in range(v['availability_start'],v['availability_end']+1)],lb=0);I={(j,t):model.addVar(lb=0,ub=ports[j]['storage_capacity'][t-1]) for j in ports for t in range(1,T+1)}
 for vid,v in vessels.items():
  rs=[r for vv,r in paths if vv==vid];model.addConstr(gp.quicksum(y[vid,r] for r in rs)==1)
  for t in range(v['availability_start'],v['availability_end']+1):
   prev=cargo[vid,t-1] if t>v['availability_start'] else v['initial_load'];net=gp.quicksum((1 if j in sup else -1)*f[vid,j,t] for j in ports if (vid,j,t) in f);model.addConstr(cargo[vid,t]==prev+net);model.addConstr(cargo[vid,t]<=gp.quicksum(ceiling[vid,r][t]*y[vid,r] for r in rs))
 for vid,j,t in keys:
  p=ports[j];model.addConstr(f[vid,j,t]<=p['F_max']*z[vid,j,t]);model.addConstr(f[vid,j,t]>=p['F_min']*z[vid,j,t]);model.addConstr(z[vid,j,t]<=gp.quicksum(y[vv,r] for vv,r in paths if vv==vid and (j,t) in visits[vv,r]))
 for j,p in ports.items():
  for t in range(1,T+1):
   prev=I[j,t-1] if t>1 else p['initial_inventory'];rate=p['production_rate'][t-1] if j in sup else -p['consumption_rate'][t-1];net=gp.quicksum((1 if j in sup else -1)*f[vid,j,t] for vid in vessels if (vid,j,t) in f);model.addConstr(I[j,t]==prev+rate-net)
 model.setObjective(gp.quicksum(costs[key]*y[key] for key in paths)+gp.quicksum((ports[j]['unit_procurement_cost'] if j in sup else -ports[j]['unit_supply_revenue'])*f[vid,j,t] for vid,j,t in keys));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  routes={};operations={};transport_cost=demurrage_cost=procurement_cost=revenue=0.
  for vid,v in vessels.items():
   r=next(r for vv,r in paths if vv==vid and get(y[vv,r])>.5);result=[]
   for x,h,typ,cost in paths[vid,r]:
    load=v['initial_load'] if typ=='source' else float(get(cargo[vid,x[1]]));result.append({'tail':str(x),'head':str(h),'type':typ,'cost':cost,'product_flow':max(0.,load)})
    if typ=='travel':transport_cost+=cost
    elif typ=='demurrage':demurrage_cost+=cost
   routes[vid]=result;ops=[]
   for vv,j,t in sorted(keys):
    if vv!=vid or get(z[vv,j,t])<.5:continue
    qty=float(get(f[vv,j,t]));signed=qty if j in sup else -qty;ops.append({'port':j,'time':t,'quantity':signed,'type':'load' if j in sup else 'discharge'})
    if j in sup:procurement_cost+=qty*ports[j]['unit_procurement_cost']
    else:revenue+=qty*ports[j]['unit_supply_revenue']
   operations[vid]=ops
  obj=transport_cost+demurrage_cost+procurement_cost-revenue;return {'objective_value':obj,'vessel_routes':routes,'vessel_operations':operations,'port_inventories':{j:[{'time':t,'inventory':float(get(I[j,t]))} for t in range(1,T+1)] for j in ports},'cost_breakdown':{'transport_cost':transport_cost,'demurrage_cost':demurrage_cost,'procurement_cost':procurement_cost,'supply_revenue':revenue,'total_cost':obj}}
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   s=extract(md.cbGetSolution);json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(s['objective_value'],s)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-begin)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No restricted maritime voyage incumbent')
if __name__=='__main__':main()

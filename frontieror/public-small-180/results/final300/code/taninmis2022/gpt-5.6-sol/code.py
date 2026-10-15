import argparse,json,os,time,itertools,math
wall_started=time.time()
started=time.monotonic()
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']): del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True)
a=p.parse_args(); end=started+a.time_limit-3; d=json.load(open(a.instance_path)); sys=d['system']; gens=sys['generators']; lines=sys['transmission_lines']; storage=sys['storage']; buses=sys['buses']; tree=d['scenario_tree']; days=d['representative_days']; sp=d['scenario_parameters']; glob=d['global_parameters']; H=sp['hours_per_day']; ns=sp['num_lower_level_scenarios_per_day']; existing=[g for g in gens if g['existing']]; candidates=[l for l in lines if not l['existing']]; loads={b:sum(v['annual_peak_load_MW'] for v in sys['loads'] if v['bus']==b) for b in buses}; logger=configured_logger(a.log_path,sense='minimize'); best=math.inf; growth={tree[0]['id']:1.}
for node in tree:
 for i,ch in enumerate(node['children']): growth[ch]=growth[node['id']]*(1+sp['demand_growth_high_percent'] if i==0 else 1+sp['demand_growth_low_percent'])
contexts=[(node,day,si,sc) for node in tree for day in days for si,sc in enumerate(d['lower_level_scenarios'][str(day['id'])])]
def blank(build):
 out={f:{} for f in ['alpha','gamma','p_gen','s_res','f_fwd','f_bwd','u_stor','v_stor','r_stor']}; root=tree[0]['id']; out['investments']={str(node['id']):{'generators':{g['id']:0 for g in gens if not g['existing']},'lines':{l['id']:int(l['id'] in build and node['id']==root) for l in candidates},'storage':{s['id']:0 for s in storage if not s['existing']}} for node in tree}; return out
def emit(out,z):
 global best
 if z>=best-1e-6: return
 best=z; out['objective_value']=z; json.dump(out,open(a.solution_path,'w')); logger.log_solution(z,out)
# Local emergency generators furnish an immediate complete feasible witness.
fallback=blank(set()); z=0.
for node,day,si,sc in contexts:
 nid=node['id']; k=day['id']; weight=node['probability']/(len(days)*ns)
 for h in range(H):
  demand={b:loads[b]*growth[nid]*day['load_profile'][h]*sc['demand_multipliers'][h] for b in buses}
  for g in gens:
   gid=g['id']; key=f'{nid},{k},{si},{h},{gid}'; local=g['existing'] and g['Pmin_MW']==0 and g['Pmax_MW']>=loads[g['bus']]*growth[nid]*2 and g['b_cost']>=10000; on=int(local); power=demand[g['bus']] if local else 0.; reserve=glob['spinning_reserve_percent']*power; fallback['alpha'][key]=on; fallback['gamma'][key]=on if h==0 else 0; fallback['p_gen'][key]=power; fallback['s_res'][key]=reserve; z+=weight*(g['a_cost']*on+g['startup_cost']*(on if h==0 else 0)+(g['b_cost']+g['c_cost']*g['Pmax_MW'])*power)
  for l in lines:
   key=f'{nid},{k},{si},{h},{l["id"]}'; fallback['f_fwd'][key]=0.; fallback['f_bwd'][key]=0.
  for s in storage:
   key=f'{nid},{k},{si},{h},{s["id"]}'
   for field in ['u_stor','v_stor','r_stor']: fallback[field][key]=0.
emit(fallback,z)
plans=[set(l['id'] for l in candidates if l['to_bus'] in loads and loads[l['to_bus']]>0)]
plans+=[set(candidates[i]['id'] for i in range(len(candidates)) if mask>>i&1) for mask in sorted(range(1<<len(candidates)),key=lambda mask:sum(l['capital_cost_per_kW']*l['flow_limit_MW']*1000 for i,l in enumerate(candidates) if mask>>i&1))]
seen=set()
for build in plans:
 token=tuple(sorted(build))
 if token in seen or time.monotonic()>=end: continue
 seen.add(token); capital=sum(l['capital_cost_per_kW']*l['flow_limit_MW']*1000 for l in candidates if l['id'] in build)
 if capital>=best: continue
 active_lines=[l for l in lines if l['existing'] or l['id'] in build]; out=blank(build); z=capital; complete=True
 for ci,(node,day,si,sc) in enumerate(contexts):
  if time.monotonic()>=end: complete=False; break
  nid=node['id']; k=day['id']; weight=node['probability']/(len(days)*ns); model=gp.Model(); model.Params.OutputFlag=0; model.Params.Threads=1; model.Params.Seed=0; model.Params.MIPGap=1e-4; model.Params.NumericFocus=0
  alpha={(g['id'],h):model.addVar(vtype=GRB.BINARY,obj=g['a_cost']) for g in existing for h in range(H)}; gamma={(g['id'],h):model.addVar(vtype=GRB.BINARY,obj=g['startup_cost']) for g in existing for h in range(H)}; power={(g['id'],h):model.addVar(obj=g['b_cost']+g['c_cost']*g['Pmax_MW']) for g in existing for h in range(H)}; reserve={(g['id'],h):model.addVar() for g in existing for h in range(H)}; ff={(l['id'],h):model.addVar(ub=l['flow_limit_MW']) for l in active_lines for h in range(H)}; fb={(l['id'],h):model.addVar(ub=l['flow_limit_MW']) for l in active_lines for h in range(H)}
  for g in existing:
   gid=g['id']
   for h in range(H):
    key=gid,h; eff=g['Pmax_MW']
    if g['type']=='solar': eff*=day['solar_profile'][h]*sc['solar_multipliers'][h]
    elif g['type']=='wind': eff*=sc['wind_power_factors'][h]
    model.addConstr(power[key]+reserve[key]<=eff*alpha[key]); model.addConstr(power[key]>=g['Pmin_MW']*alpha[key]); model.addConstr(gamma[key]>=alpha[key]-(alpha[gid,h-1] if h else 0))
    if h:
     model.addConstr(power[key]-power[gid,h-1]<=g['ramp_MW_per_h']); model.addConstr(power[gid,h-1]-power[key]<=g['ramp_MW_per_h'])
     for tau in range(h,min(H,h+g['min_on_h'])): model.addConstr(alpha[gid,tau]>=alpha[key]-alpha[gid,h-1])
     for tau in range(h,min(H,h+g['min_off_h'])): model.addConstr(1-alpha[gid,tau]>=alpha[gid,h-1]-alpha[key])
  for h in range(H):
   for b in buses:
    demand=loads[b]*growth[nid]*day['load_profile'][h]*sc['demand_multipliers'][h]; expr=gp.quicksum(power[g['id'],h] for g in existing if g['bus']==b)
    for l in active_lines:
     lid=l['id']; loss=l['loss_factor']
     if l['to_bus']==b: expr+=(1-loss)*ff[lid,h]-fb[lid,h]
     if l['from_bus']==b: expr+=(1-loss)*fb[lid,h]-ff[lid,h]
    model.addConstr(expr==demand); model.addConstr(gp.quicksum(reserve[g['id'],h] for g in existing if g['bus']==b)>=glob['spinning_reserve_percent']*demand)
  model.update(); assert model.NumVars<=2000 and model.NumConstrs<=2000
  variables=list(alpha.values())+list(gamma.values())+list(power.values())+list(reserve.values())+list(ff.values())+list(fb.values())
  def finish_context(values):
   candidate={field:vals.copy() for field,vals in out.items()}; groups=[alpha,gamma,power,reserve,ff,fb]; offset=0; maps=[]
   for group in groups:
    maps.append(dict(zip(group,values[offset:offset+len(group)]))); offset+=len(group)
   av,gv,pv,rv,fv,bv=maps; total=z
   for h in range(H):
    for g in gens:
     gid=g['id']; key=f'{nid},{k},{si},{h},{gid}'; on=int(av[gid,h]>.5) if g['existing'] else 0; gam=int(gv[gid,h]>.5) if g['existing'] else 0; val=max(0,pv[gid,h]) if g['existing'] else 0.; res=max(0,rv[gid,h]) if g['existing'] else 0.; candidate['alpha'][key]=on; candidate['gamma'][key]=gam; candidate['p_gen'][key]=val; candidate['s_res'][key]=res; total+=weight*(g['a_cost']*on+g['startup_cost']*gam+(g['b_cost']+g['c_cost']*g['Pmax_MW'])*val)
    for l in lines:
     lid=l['id']; key=f'{nid},{k},{si},{h},{lid}'; candidate['f_fwd'][key]=max(0,fv[lid,h]) if l in active_lines else 0.; candidate['f_bwd'][key]=max(0,bv[lid,h]) if l in active_lines else 0.
    for s in storage:
     key=f'{nid},{k},{si},{h},{s["id"]}'
     for field in ['u_stor','v_stor','r_stor']: candidate[field][key]=0.
   emit(candidate,total)
  def cb(m,where):
   if where==GRB.Callback.MIPSOL and ci==len(contexts)-1: finish_context(m.cbGetSolution(variables))
  model.Params.TimeLimit=max(.05,(end-time.monotonic())/(len(contexts)-ci)); model.optimize(cb)
  if not model.SolCount: complete=False; model.dispose(); break
  for h in range(H):
   for g in gens:
    gid=g['id']; key=f'{nid},{k},{si},{h},{gid}'; on=int(alpha[gid,h].X>.5) if g['existing'] else 0; gam=int(gamma[gid,h].X>.5) if g['existing'] else 0; val=max(0,power[gid,h].X) if g['existing'] else 0.; res=max(0,reserve[gid,h].X) if g['existing'] else 0.; out['alpha'][key]=on; out['gamma'][key]=gam; out['p_gen'][key]=val; out['s_res'][key]=res; z+=weight*(g['a_cost']*on+g['startup_cost']*gam+(g['b_cost']+g['c_cost']*g['Pmax_MW'])*val)
   for l in lines:
    lid=l['id']; key=f'{nid},{k},{si},{h},{lid}'; out['f_fwd'][key]=max(0,ff[lid,h].X) if l in active_lines else 0.; out['f_bwd'][key]=max(0,fb[lid,h].X) if l in active_lines else 0.
   for s in storage:
    key=f'{nid},{k},{si},{h},{s["id"]}'
    for field in ['u_stor','v_stor','r_stor']: out[field][key]=0.
  model.dispose()
 if complete: emit(out,z)

import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 T=data['time_periods'];periods=range(1,T+1);gens={g['id']:g for g in data['generators']['generators']};buses={b['id']:b for b in data['electricity_network']['buses']};lines={l['id']:l for l in data['electricity_network']['lines']};gas=data['gas_network'];junctions={j['id']:j for j in gas['junctions']};conns={c['id']:c for c in gas['connections']};beta=data['objective_weight_beta'];m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;m.Params.BarQCPConvTol=1e-8;witness={}
 def var(name,lb=0,ub=GRB.INFINITY,binary=False):
  v=m.addVar(lb=lb,ub=ub,vtype=GRB.BINARY if binary else GRB.CONTINUOUS,name=name);witness[name]=v;return v
 o={};vp={};vm={};r={};w={};se={};p={};phi={}
 for u,g in gens.items():
  o[u,0]=g['initial_status'];p[u,0]=g['initial_gen'];witness[f'o_{u}_0']=o[u,0];witness[f'p_{u}_0']=p[u,0]
  for t in periods:
   o[u,t]=var(f'o_{u}_{t}',binary=True);vp[u,t]=var(f'vp_{u}_{t}',binary=True);vm[u,t]=var(f'vm_{u}_{t}',binary=True);r[u,t]=var(f'r_{u}_{t}')
   for b in g['bids']:
    bid=b['id'];w[u,bid,t]=var(f'w_{u}_{bid}_{t}',binary=True);se[u,bid,t]=var(f'se_{u}_{bid}_{t}',ub=b['max_amount'])
   p[u,t]=gp.quicksum(se[u,b['id'],t] for b in g['bids']);witness[f'p_{u}_{t}']=p[u,t]
   if g['is_gfpp']:phi[u,t]=g['max_gas_price_fraction']*gas['max_gas_price_mmBtu']*o[u,t];witness[f'phimax_{u}_{t}']=phi[u,t]
 theta={(i,t):var(f'theta_{i}_{t}',b['voltage_angle_lb'],b['voltage_angle_ub']) for i,b in buses.items() for t in periods};f={}
 for lid,l in lines.items():
  for t in periods:
   f[lid,t]=l['susceptance']*(theta[l['from_bus'],t]-theta[l['to_bus'],t]);witness[f'f_{lid}_{t}']=f[lid,t];m.addConstr(f[lid,t]<=l['thermal_limit']);m.addConstr(f[lid,t]>=-l['thermal_limit']);m.addConstr(theta[l['from_bus'],t]-theta[l['to_bus'],t]<=l['angle_diff_limit']);m.addConstr(theta[l['from_bus'],t]-theta[l['to_bus'],t]>=-l['angle_diff_limit'])
 sg={};qg={};lg={};gamma={};pisq={};sgs={};phig={};psi={};zone={j:z['id'] for z in gas['pricing_zones'] for j in z['junctions']}
 for z in gas['pricing_zones']:
  for t in periods:psi[z['id'],t]=gas['min_gas_price_mmBtu'];witness[f"psi_{z['id']}_{t}"]=psi[z['id'],t]
 for j,junc in junctions.items():
  for t in periods:
   qg[j,t]=var(f'qg_{j}_{t}',ub=junc['gas_demand_profile'][t-1]);lg[j,t]=junc['gas_demand_profile'][t-1]-qg[j,t];witness[f'lg_{j}_{t}']=lg[j,t];pisq[j,t]=var(f'pisq_{j}_{t}',junc['pressure_lb_squared'],junc['pressure_ub_squared']);gfpps=[u for u,g in gens.items() if g['is_gfpp'] and g['gas_junction']==j]
   if gfpps:
    gamma[j,t]=var(f'gamma_{j}_{t}');m.addQConstr(gamma[j,t]>=gp.quicksum(gens[u]['heat_rate_coefficients']['H_u2']*p[u,t]*p[u,t]+gens[u]['heat_rate_coefficients']['H_u1']*p[u,t]+gens[u]['heat_rate_coefficients']['H_u0']*o[u,t] for u in gfpps))
   else:gamma[j,t]=0.;witness[f'gamma_{j}_{t}']=0.
   if junc['is_source']:
    for s in junc['supply_intervals']:sgs[j,s['id'],t]=var(f"sgs_{j}_{s['id']}_{t}",ub=s['interval_ub']-s['interval_lb'])
    sg[j,t]=gp.quicksum(sgs[j,s['id'],t] for s in junc['supply_intervals'])
   else:sg[j,t]=0.
   witness[f'sg_{j}_{t}']=sg[j,t]
 for cid,c in conns.items():
  a,b=c['from_junction'],c['to_junction']
  for t in periods:
   phig[cid,t]=var(f'phig_{cid}_{t}')
   if c['type']=='pipeline':m.addQConstr(c['weymouth_factor']*phig[cid,t]*phig[cid,t]<=pisq[a,t]-pisq[b,t])
   else:
    prefix='compression' if c['type']=='compressor' else 'control';m.addConstr(pisq[b,t]>=c[prefix+'_ratio_lb']**2*pisq[a,t]);m.addConstr(pisq[b,t]<=c[prefix+'_ratio_ub']**2*pisq[a,t])
 for j in junctions:
  for t in periods:m.addConstr(sg[j,t]-lg[j,t]-gamma[j,t]==gp.quicksum(phig[cid,t] for cid,c in conns.items() if c['from_junction']==j)-gp.quicksum(phig[cid,t] for cid,c in conns.items() if c['to_junction']==j))
 for u,g in gens.items():
  remaining=g['initial_active_periods']+g['initial_inactive_periods']
  for t in periods:
   if t<=remaining:m.addConstr(o[u,t]==g['initial_status'])
   m.addConstr(vp[u,t]-vm[u,t]==o[u,t]-o[u,t-1]);m.addConstr(vp[u,t]+vm[u,t]<=1);m.addConstr(gp.quicksum(vp[u,k] for k in range(max(1,t-g['min_up_time']+1),t+1))<=o[u,t]);m.addConstr(gp.quicksum(vm[u,k] for k in range(max(1,t-g['min_down_time']+1),t+1))<=1-o[u,t]);m.addConstr(p[u,t]>=g['min_power']*o[u,t]);m.addConstr(p[u,t]<=g['max_power']*o[u,t]);m.addConstr(p[u,t]-p[u,t-1]<=g['ramp_up']*o[u,t-1]+g['max_power']*vp[u,t]);m.addConstr(p[u,t-1]-p[u,t]<=g['ramp_down']*o[u,t]+g['min_power']*vm[u,t])
   for h,cost in g['startup_cost_params']:m.addConstr(r[u,t]>=cost*(o[u,t]-gp.quicksum(o[u,t-k] if t-k>=0 else g['initial_status'] for k in range(1,h+1))))
   for k,b in enumerate(g['bids']):
    bid=b['id'];m.addConstr(w[u,bid,t]<=o[u,t]);m.addConstr(se[u,bid,t]<=b['max_amount']*w[u,bid,t])
    if k+1<len(g['bids']):m.addConstr(se[u,bid,t]>=b['max_amount']*w[u,g['bids'][k+1]['id'],t])
   if g['is_gfpp']:m.addConstr(phi[u,t]>=psi[zone[g['gas_junction']],t]*o[u,t])
 for i,b in buses.items():
  for t in periods:m.addConstr(gp.quicksum(p[u,t] for u,g in gens.items() if g['bus']==i)-b['demand_profile'][t-1]==gp.quicksum(f[lid,t] for lid,l in lines.items() if l['from_bus']==i)-gp.quicksum(f[lid,t] for lid,l in lines.items() if l['to_bus']==i))
 electricity=gp.quicksum(g['no_load_cost']*o[u,t]+r[u,t]+gp.quicksum(b['price']*se[u,b['id'],t] for b in g['bids']) for u,g in gens.items() for t in periods);gascost=gp.quicksum(j['demand_shedding_cost']*qg[jid,t]+(gp.quicksum(s['slope']*sgs[jid,s['id'],t] for s in j['supply_intervals']) if j['is_source'] else 0) for jid,j in junctions.items() for t in periods);m.setObjective(beta*electricity+(1-beta)*gascost)
 def extract(get):
  def value(v):
   if isinstance(v,gp.Var):return get(v)
   if isinstance(v,gp.LinExpr):return v.getConstant()+sum(v.getCoeff(i)*get(v.getVar(i)) for i in range(v.size()))
   return float(v)
  primary={key:value(v) for key,v in witness.items()};obj=beta*value(electricity)+(1-beta)*value(gascost);return {'objective_value':obj,'released_problem_variant':data['released_problem_variant'],'objective_weight_beta':beta,'primary_variables':primary}
 def cb(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 if not m.SolCount:raise RuntimeError('no integrated UCGNA feasible incumbent')
 return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

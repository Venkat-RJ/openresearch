import time
_wall_start=time.time()
import argparse,json,os,math
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();deadline=start+a.time_limit-.5;d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
old='problem_dimensions' in d;raw=d['generators'];G=range(len(raw));policy=d['policy'];generators=[]
if old:
 H=d['subperiods']['hours_per_subperiod'];W=range(d['subperiods']['num_subperiods']);weights=d['subperiods']['subperiod_weights'];total=d['problem_dimensions']['total_timesteps'];Z=range(1,d['problem_dimensions']['num_zones']+1);segments=[(s,z,q['max_frac'],q['cost_per_MWh']) for s,q in enumerate(d['nse_segments']) for z in Z]
 for i,g in enumerate(raw):
  av=d['availability_profiles'].get(str(g['id']),{'type':'constant','value':1});sigma=av['values'] if av['type']=='timeseries' else [av['value']]*total;generators.append(dict(g,sigma=sigma,increment=g['capacity_size_MW'],energy_increment=g.get('storage_capacity_size_MWh',1),physical=g['capacity_size_MW']))
else:
 H=d['problem_parameters']['hours_per_subperiod'];W=range(d['problem_parameters']['num_weeks']);weights=d['subperiod_weights'];total=d['problem_parameters']['total_hours'];Z=range(d['problem_parameters']['num_zones']);segments=[(s,q['zone'],q['max_nse_fraction'],q['cost_per_MWh']) for s,q in enumerate(d['consumer_segments'])]
 for i,g in enumerate(raw):
  q=d['generator_parameters'][g['resource_type']];existing=d['existing_capacity'].get(str(i),0);inc=g['capacity_size_MW'] if g['is_UC'] else d['released_investment_convention']['non_uc_power_increment_MW'];einc=d['released_investment_convention']['non_uc_storage_energy_increment_MWh'];bp=d['generator_parameters']['battery'];generators.append(dict(g,id=i,increment=inc,energy_increment=einc,physical=g['capacity_size_MW'],existing_capacity_MW=existing,max_capacity_MW=q['max_capacity_MW'],inv_cost_per_MW_yr=q['investment_cost_per_MW_yr'],fom_cost_per_MW_yr=q['fixed_OM_cost_per_MW_yr'],var_cost_per_MWh=q['variable_cost_per_MWh'],min_output_frac=q['min_output_frac'],ramp_up_frac_per_hr=q['ramp_up_rate_pct_per_hr'],ramp_dn_frac_per_hr=q['ramp_down_rate_pct_per_hr'],co2_tons_per_MWh=q['CO2_emission_tons_per_MWh'],sigma=d['availability'].get(str(i),[1]*total),existing_storage_capacity_MWh=existing/bp['capacity_size_MW']*bp['energy_capacity_size_MWh'],max_storage_capacity_MWh=bp['max_energy_capacity_MWh'],inv_cost_energy_per_MWh_yr=q.get('energy_investment_cost_per_MWh_yr',0),fom_cost_energy_per_MWh_yr=q.get('energy_fixed_OM_cost_per_MWh_yr',0),min_duration_MWh_per_MW=bp['min_duration_MWh_per_MW'],max_duration_MWh_per_MW=bp['max_duration_MWh_per_MW'],duration_MWh_per_MW=q.get('duration_MWh_per_MW',0)))
T=range(total);alpha=[weights[t//H]/H for t in T];demand={z:d['demand'][str(z)] for z in Z};ST=[i for i,g in enumerate(generators) if g['is_storage']];UC=[i for i,g in enumerate(generators) if g['is_UC']];HY=[i for i,g in enumerate(generators) if g['is_hydro']];lines=d['transmission_lines'];L=range(len(lines))
UG={};owner={}
for i in UC:
 g=generators[i];key=(g['zone'],g['resource_type'],g['physical'],g['min_output_frac'],g['ramp_up_frac_per_hr'],g['ramp_dn_frac_per_hr'],g['var_cost_per_MWh'],g['co2_tons_per_MWh'],g['is_RPS_qualifying'],tuple(g['sigma']));rep=next((k for k,vs in UG.items() if vs[0][0]==key),i);UG.setdefault(rep,[]).append((key,i));owner[i]=rep
UG={rep:[i for _,i in vs] for rep,vs in UG.items()};bal={z:max([rep for rep in UG if generators[rep]['zone']==z and ('gas' in generators[rep]['resource_type'] or generators[rep]['resource_type'].startswith('ng_'))],key=lambda rep:sum(generators[i]['max_capacity_MW']+generators[i]['existing_capacity_MW'] for i in UG[rep])) for z in Z};B=range((total+5)//6)
m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
new=m.addVars(G,vtype=GRB.INTEGER,lb=0,ub={i:generators[i]['max_capacity_MW']/generators[i]['increment'] for i in G});ret=m.addVars(G,vtype=GRB.INTEGER,lb=0,ub={i:generators[i]['existing_capacity_MW']/generators[i]['increment'] if generators[i]['can_retire'] else 0 for i in G});cap={i:generators[i]['existing_capacity_MW']+generators[i]['increment']*(new[i]-ret[i]) for i in G};en=m.addVars(ST,vtype=GRB.INTEGER,lb=0,ub={i:generators[i]['max_storage_capacity_MWh']/generators[i]['energy_increment'] for i in ST});er=m.addVars(ST,vtype=GRB.INTEGER,lb=0,ub={i:generators[i]['existing_storage_capacity_MWh']/generators[i]['energy_increment'] if generators[i]['can_retire'] else 0 for i in ST});ecap={i:generators[i]['existing_storage_capacity_MWh']+generators[i]['energy_increment']*(en[i]-er[i]) for i in ST};commit=m.addVars(list(UG),lb=0);output=m.addVars([i for i in G if i not in ST and i not in HY and i not in UC],lb=0);block=m.addVars([(rep,b) for rep in UG if rep not in bal.values() for b in B],lb=0)
for i in ST:m.addConstr(ecap[i]>=generators[i]['min_duration_MWh_per_MW']*cap[i]);m.addConstr(ecap[i]<=generators[i]['max_duration_MWh_per_MW']*cap[i])
for rep in UG:m.addConstr(commit[rep]<=gp.quicksum(cap[i] for i in UG[rep]))
fn=m.addVars(L,vtype=GRB.INTEGER,lb=0,ub={l:lines[l]['max_new_capacity_MW'] if old else lines[l]['max_capacity_MW']-lines[l]['existing_capacity_MW'] for l in L});flow=m.addVars(L,lb=-GRB.INFINITY)
for l in L:m.addConstr(flow[l]<=lines[l]['existing_capacity_MW']+fn[l]);m.addConstr(-flow[l]<=lines[l]['existing_capacity_MW']+fn[l])
nse=m.addVars(range(len(segments)),lb=0,ub={s:seg[2] for s,seg in enumerate(segments)});rps=m.addVars(W,lb=0);co2=m.addVars(W,lb=0);inj={};level={};spill={};rows=[]
for i in G:
 g=generators[i]
 if i in ST:
  for t in T:inj[i,t]=gp.LinExpr(0)
 elif i in HY:
  if old:
   for t in T:inj[i,t]=g['sigma'][t]*cap[i];level[i,t]=0;spill[i,t]=0
  else:
   inflow=d['hydro_inflow'].get(str(i),[0]*total)
   for w in W:
    hours=list(range(w*H,(w+1)*H));avg=sum(inflow[t] for t in hours)/H;cum=[];v=0
    for t in hours:v+=inflow[t]-avg;cum.append(v)
    low=min(cum)
    m.addConstr((max(cum)-low)*cap[i]<=g['duration_MWh_per_MW']*cap[i]);m.addConstr(avg*cap[i]>=g['min_output_frac']*cap[i])
    for k,t in enumerate(hours):q=min(avg,g['sigma'][t]);inj[i,t]=q*cap[i];spill[i,t]=(avg-q)*cap[i];level[i,t]=(cum[k]-low)*cap[i]
 elif i in UC:
  for t in T:inj[i,t]=gp.LinExpr(0)
 else:
  m.addConstr(output[i]<=cap[i]);m.addConstr(output[i]>=g['min_output_frac']*cap[i] if old else min(g['sigma'])*output[i]>=g['min_output_frac']*cap[i])
  for t in T:
   inj[i,t]=g['sigma'][t]*output[i];prev=(t//H)*H+(t-1)%H
   if g['ramp_up_frac_per_hr']<1:rows.append(inj[i,t]-g['sigma'][prev]*output[i]-g['ramp_up_frac_per_hr']*cap[i])
   if g['ramp_dn_frac_per_hr']<1:rows.append(g['sigma'][prev]*output[i]-inj[i,t]-g['ramp_dn_frac_per_hr']*cap[i])
for rep in UG:
 if rep in bal.values():continue
 g=generators[rep]
 for b in B:m.addConstr(block[rep,b]>=g['min_output_frac']*commit[rep]);m.addConstr(block[rep,b]<=min(g['sigma'][b*6:min(total,(b+1)*6)])*commit[rep])
 for t in T:
  inj[rep,t]=1*block[rep,t//6];prev=(t//H)*H+(t-1)%H
  rows.append(block[rep,t//6]-block[rep,prev//6]-g['ramp_up_frac_per_hr']*commit[rep]);rows.append(block[rep,prev//6]-block[rep,t//6]-g['ramp_dn_frac_per_hr']*commit[rep])
for z in Z:
 i=bal[z];g=generators[i]
 for t in T:
  inj[i,t]=gp.LinExpr(demand[z][t])-gp.quicksum(inj[j,t] for j in G if generators[j]['zone']==z and j!=i)-gp.quicksum(demand[z][t]*nse[s] for s,seg in enumerate(segments) if seg[1]==z)+gp.quicksum(flow[l] for l in L if lines[l]['from_zone']==z)-gp.quicksum(flow[l] for l in L if lines[l]['to_zone']==z)
 for t in T:
  prev=(t//H)*H+(t-1)%H;rows.extend([inj[i,t]-g['sigma'][t]*commit[i],g['min_output_frac']*commit[i]-inj[i,t],inj[i,t]-inj[i,prev]-g['ramp_up_frac_per_hr']*commit[i],inj[i,prev]-inj[i,t]-g['ramp_dn_frac_per_hr']*commit[i]])
fixed=gp.quicksum(generators[i]['inv_cost_per_MW_yr']*generators[i]['increment']*new[i]+generators[i]['fom_cost_per_MW_yr']*cap[i] for i in G)+gp.quicksum(generators[i].get('inv_cost_energy_per_MWh_yr',0)*generators[i]['energy_increment']*en[i]+generators[i].get('fom_cost_energy_per_MWh_yr',0)*ecap[i] for i in ST)+gp.quicksum(lines[l]['inv_cost_per_MW_yr']*fn[l] if old else lines[l]['investment_cost_per_MW_yr']*fn[l] for l in L)
for i in HY:fixed+=generators[i].get('inv_cost_energy_per_MWh_yr',0)*generators[i]['duration_MWh_per_MW']*(generators[i]['increment'] if old else 1)*new[i]+generators[i].get('fom_cost_energy_per_MWh_yr',0)*generators[i]['duration_MWh_per_MW']*cap[i]
variable=gp.quicksum(alpha[t]*generators[i]['var_cost_per_MWh']*inj[i,t] for i in G for t in T)+gp.quicksum(alpha[t]*seg[3]*demand[seg[1]][t]*nse[s] for s,seg in enumerate(segments) for t in T)
rc=policy['RPS_noncompliance_cost_per_MWh'];cc=policy['CO2_noncompliance_cost_per_ton'];cost=fixed+variable+gp.quicksum(rc*rps[w]+cc*co2[w] for w in W)
for w in W:
 hours=range(w*H,(w+1)*H);dem=sum(alpha[t]*demand[z][t] for z in Z for t in hours);renew=gp.quicksum(alpha[t]*inj[i,t] for i in G if generators[i]['is_RPS_qualifying'] for t in hours);emit=gp.quicksum(alpha[t]*generators[i]['co2_tons_per_MWh']*inj[i,t] for i in G for t in hours)
 if old:
  if policy['RPS_enabled']:m.addConstr(renew+rps[w]>=policy['RPS_share']*dem)
  if policy['CO2_cap_enabled']:m.addConstr(emit-co2[w]<=policy['CO2_cap_tons_per_MWh']*dem)
if not old:
 dem=sum(alpha[t]*demand[z][t] for z in Z for t in T);m.addConstr(gp.quicksum(alpha[t]*inj[i,t] for i in G if generators[i]['is_RPS_qualifying'] for t in T)+rps.sum()>=policy['RPS_share']*dem);m.addConstr(gp.quicksum(alpha[t]*generators[i]['co2_tons_per_MWh']*inj[i,t] for i in G for t in T)-co2.sum()<=policy['CO2_cap_tons_per_MWh']*dem)
m.setObjective(cost);m.update();variables=m.getVars();best=float('inf');solution=None

def evaluate(e,v):
 if isinstance(e,(int,float)):return e
 if isinstance(e,gp.Var):return v[e.index]
 return e.getConstant()+sum(e.getCoeff(k)*v[e.getVar(k).index] for k in range(e.size()))
def pack(v):
 ev=lambda e:evaluate(e,v);witness={}
 def put(name,ix,value):
  if abs(value)>1e-9:witness[(name+'_'+'_'.join(map(str,ix))) if old else (name+'['+','.join(map(str,ix))+']')]=value
 inv=[] if old else {};storage={};trans=[] if old else {}
 for i,g in enumerate(generators):
  pn=ev(new[i]);pr=ev(ret[i]);pc=ev(cap[i]);put('yPnew' if old else 'y_P_new',(i,),pn);put('yPret' if old else 'y_P_ret',(i,),pr);put('yP' if old else 'y_P',(i,),pc)
  if old:entry={'generator_id':g['id'],'resource_type':g['resource_type'],'zone':g['zone'],'y_P_new':pn,'y_P_ret':pr,'y_P_total_MW':pc};inv.append(entry)
  else:inv[str(i)]={'resource_type':g['resource_type'],'zone':g['zone'],'new_units':pn,'retired_units':pr,'new_MW':pn*g['increment'],'retired_MW':pr*g['increment'],'total_capacity_MW':pc}
  if i in ST:
   enew=ev(en[i]);eret=ev(er[i]);energy=ev(ecap[i]);put('yEnew' if old else 'y_E_new',(i,),enew);put('yEret' if old else 'y_E_ret',(i,),eret);put('yE' if old else 'y_E',(i,),energy)
   if old:entry.update(y_E_new=enew,y_E_ret=eret,y_E_total_MWh=energy)
   else:storage[str(i)]={'new_units':enew,'new_MWh':enew*g['energy_increment'],'total_energy_MWh':energy}
  share=pc/sum(ev(cap[j]) for j in UG[owner[i]]) if i in UC and sum(ev(cap[j]) for j in UG[owner[i]])>1e-9 else 0
  for t in T:
   put('xinj' if old else 'x_inj',(i,t),share*ev(inj[owner[i],t]) if i in UC else ev(inj[i,t]))
   if i in UC:put('xcommit' if old else 'x_commit',(i,t),share*ev(commit[owner[i]])/g['physical'])
   if i in HY:
    put('xhydro' if old else 'x_lvl',(i,t),ev(level[i,t]))
    if not old:put('x_spill',(i,t),ev(spill[i,t]))
 for l,line in enumerate(lines):
  fnew=ev(fn[l]);totalcap=line['existing_capacity_MW']+fnew;put('yFnew' if old else 'y_F_new',(l,),fnew)
  if old:trans.append({'line_id':line['id'],'y_F_new_MW':fnew,'total_capacity_MW':totalcap})
  else:put('y_F',(l,),totalcap);trans[str(l)]={'new_MW':fnew,'total_capacity_MW':totalcap}
  for t in T:put('xflow' if old else 'x_flow',(l,t),ev(flow[l]))
 for s,seg in enumerate(segments):
  for t in T:put('xnse' if old else 'x_nse',(seg[0],seg[1],t) if old else (s,t),ev(nse[s])*demand[seg[1]][t])
 for w in W:put('xRPS' if old else 'x_RPS',(w,),ev(rps[w]));put('xCO2' if old else 'x_CO2',(w,),ev(co2[w]))
 sol={'objective_value':ev(cost),'model_variables':witness}
 if old:sol.update(investment_decisions=inv,transmission_decisions=trans)
 else:sol.update(investments=inv,storage_energy_investments=storage,transmission_investments=trans)
 return sol

def publish(v):
 global best,solution
 if any(evaluate(row,v)>1e-6 for row in rows):return
 s=pack(v)
 if s['objective_value']<best:
  best=s['objective_value'];solution=s
  if log:log.log_solution(best,s)
def cb(mm,where):
 if where==GRB.Callback.MIPSOL:publish(mm.cbGetSolution(variables))
active=set()
while time.monotonic()<deadline:
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 if not m.SolCount:break
 v=[z.X for z in variables];violated=sorted([(evaluate(row,v),k) for k,row in enumerate(rows) if k not in active and evaluate(row,v)>1e-7],reverse=True)
 if not violated:publish(v);break
 budget=1850-m.NumConstrs
 if budget<=0:break
 for _,k in violated[:min(100,budget)]:m.addConstr(rows[k]<=0);active.add(k)
if solution is None:raise RuntimeError('No complete feasible restricted-profile energy witness')
solution['runtime']=time.monotonic()-start;json.dump(solution,open(a.solution_path,'w'))

import time
_wall_start=time.time()
import argparse,json,os
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();deadline=start+a.time_limit-.5;d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
raw=d['generators'];G=range(len(raw));params=d['problem_parameters'];H=params['hours_per_subperiod'];W=range(params['num_weeks']);T=range(params['total_hours']);Z=range(params['num_zones']);ST=[i for i,g in enumerate(raw) if g['is_storage']];UC=[i for i,g in enumerate(raw) if g['is_UC']];HY=[i for i,g in enumerate(raw) if g['is_hydro']];Q={i:d['generator_parameters'][g['resource_type']] for i,g in enumerate(raw)};existing={i:d['existing_capacity'].get(str(i),0) for i in G};inc={i:raw[i]['capacity_size_MW'] if i in UC else d['released_investment_convention']['non_uc_power_increment_MW'] for i in G};einc=d['released_investment_convention']['non_uc_storage_energy_increment_MWh'];sigma={i:d['availability'].get(str(i),[1]*len(T)) for i in G};alpha=[d['subperiod_weights'][t//H]/H for t in T];dem={z:d['demand'][str(z)] for z in Z};segments=d['consumer_segments'];lines=d['transmission_lines'];L=range(len(lines))
UG={};owner={}
for i in UC:
 q=Q[i];key=(raw[i]['zone'],raw[i]['resource_type'],raw[i]['capacity_size_MW'],q['min_output_frac'],q['ramp_up_rate_pct_per_hr'],q['ramp_down_rate_pct_per_hr'],q['variable_cost_per_MWh'],q['CO2_emission_tons_per_MWh'],raw[i]['is_RPS_qualifying'],tuple(sigma[i]));rep=next((k for k,vs in UG.items() if vs[0][0]==key),i);UG.setdefault(rep,[]).append((key,i));owner[i]=rep
UG={rep:[i for _,i in vs] for rep,vs in UG.items()};bal={z:max([rep for rep in UG if raw[rep]['zone']==z and 'gas' in raw[rep]['resource_type']],key=lambda rep:sum(existing[i]+Q[i]['max_capacity_MW'] for i in UG[rep])) for z in Z};B=range((len(T)+5)//6)
m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
new=m.addVars(G,lb=0,vtype=GRB.INTEGER,ub={i:Q[i]['max_capacity_MW']/inc[i] for i in G});ret=m.addVars(G,lb=0,vtype=GRB.INTEGER,ub={i:existing[i]/inc[i] if raw[i]['can_retire'] else 0 for i in G});cap={i:existing[i]+inc[i]*(new[i]-ret[i]) for i in G};bp=d['generator_parameters']['battery'];e0={i:existing[i]/bp['capacity_size_MW']*bp['energy_capacity_size_MWh'] for i in ST};en=m.addVars(ST,lb=0,vtype=GRB.INTEGER,ub=bp['max_energy_capacity_MWh']/einc);er=m.addVars(ST,lb=0,vtype=GRB.INTEGER,ub={i:e0[i]/einc if raw[i]['can_retire'] else 0 for i in ST});ecap={i:e0[i]+einc*(en[i]-er[i]) for i in ST};commit=m.addVars(list(UG),lb=0);output=m.addVars([i for i in G if i not in ST and i not in HY and i not in UC],lb=0);block=m.addVars([(rep,b) for rep in UG if rep not in bal.values() for b in B],lb=0)
for i in ST:m.addConstr(ecap[i]>=bp['min_duration_MWh_per_MW']*cap[i]);m.addConstr(ecap[i]<=bp['max_duration_MWh_per_MW']*cap[i])
for rep in UG:m.addConstr(commit[rep]<=gp.quicksum(cap[i] for i in UG[rep]))
fn=m.addVars(L,lb=0,vtype=GRB.INTEGER,ub={l:max(0,lines[l]['max_capacity_MW']-lines[l]['existing_capacity_MW']) for l in L});flow=m.addVars(L,lb=-GRB.INFINITY)
for l in L:m.addConstr(flow[l]<=lines[l]['existing_capacity_MW']+fn[l]);m.addConstr(-flow[l]<=lines[l]['existing_capacity_MW']+fn[l])
nse=m.addVars(range(len(segments)),lb=0,ub={s:seg['max_nse_fraction'] for s,seg in enumerate(segments)});rps=m.addVars(W,lb=0);co2=m.addVars(W,lb=0);inj={};lvl={};spill={};rows=[]
for i in G:
 q=Q[i];theta=q.get('min_output_frac',0);physical=raw[i]['capacity_size_MW']
 if i in ST:
  for t in T:inj[i,t]=gp.LinExpr(0)
 elif i in HY:
  inflow=d['hydro_inflow'].get(str(i),[0]*len(T))
  for w in W:
   hours=list(range(w*H,(w+1)*H));avg=sum(inflow[t] for t in hours)/H;v=0;cum=[]
   for t in hours:v+=inflow[t]-avg;cum.append(v)
   low=min(cum);m.addConstr((max(cum)-low)*cap[i]<=q['duration_MWh_per_MW']*cap[i]);m.addConstr(avg*cap[i]>=theta*cap[i])
   for k,t in enumerate(hours):f=min(avg,sigma[i][t]);inj[i,t]=f*cap[i];spill[i,t]=(avg-f)*cap[i];lvl[i,t]=(cum[k]-low)*cap[i]
 elif i in UC:
  for t in T:inj[i,t]=gp.LinExpr(0)
 else:
  m.addConstr(output[i]<=cap[i]);m.addConstr(min(sigma[i])*output[i]>=theta*cap[i])
  for t in T:inj[i,t]=sigma[i][t]*output[i]
  for t in T:
   prev=(t//H)*H+(t-1)%H;up=q.get('ramp_up_rate_pct_per_hr',0);dn=q.get('ramp_down_rate_pct_per_hr',0)
   if 0<up<1:rows.append(inj[i,t]-inj[i,prev]-up*cap[i])
   if 0<dn<1:rows.append(inj[i,prev]-inj[i,t]-dn*cap[i])
for rep in UG:
 if rep in bal.values():continue
 q=Q[rep]
 for b in B:m.addConstr(block[rep,b]>=q['min_output_frac']*commit[rep]);m.addConstr(block[rep,b]<=min(sigma[rep][b*6:min(len(T),(b+1)*6)])*commit[rep])
 for t in T:
  inj[rep,t]=1*block[rep,t//6];prev=(t//H)*H+(t-1)%H
  rows.append(block[rep,t//6]-block[rep,prev//6]-q['ramp_up_rate_pct_per_hr']*commit[rep]);rows.append(block[rep,prev//6]-block[rep,t//6]-q['ramp_down_rate_pct_per_hr']*commit[rep])
for z in Z:
 i=bal[z];q=Q[i];physical=raw[i]['capacity_size_MW']
 for t in T:inj[i,t]=gp.LinExpr(dem[z][t])-gp.quicksum(inj[j,t] for j in G if raw[j]['zone']==z and j!=i)-gp.quicksum(dem[z][t]*nse[s] for s,seg in enumerate(segments) if seg['zone']==z)+gp.quicksum(flow[l] for l in L if lines[l]['from_zone']==z)-gp.quicksum(flow[l] for l in L if lines[l]['to_zone']==z)
 for t in T:
  prev=(t//H)*H+(t-1)%H;rows.extend([inj[i,t]-sigma[i][t]*commit[i],q['min_output_frac']*commit[i]-inj[i,t],inj[i,t]-inj[i,prev]-q['ramp_up_rate_pct_per_hr']*commit[i],inj[i,prev]-inj[i,t]-q['ramp_down_rate_pct_per_hr']*commit[i]])
fixed=gp.quicksum(Q[i]['investment_cost_per_MW_yr']*inc[i]*new[i]+Q[i]['fixed_OM_cost_per_MW_yr']*cap[i] for i in G)+gp.quicksum(bp['energy_investment_cost_per_MWh_yr']*einc*en[i]+bp['energy_fixed_OM_cost_per_MWh_yr']*ecap[i] for i in ST)+gp.quicksum(lines[l]['investment_cost_per_MW_yr']*fn[l] for l in L)
for i in HY:fixed+=Q[i]['energy_investment_cost_per_MWh_yr']*Q[i]['duration_MWh_per_MW']*new[i]+Q[i]['energy_fixed_OM_cost_per_MWh_yr']*Q[i]['duration_MWh_per_MW']*cap[i]
variable=gp.quicksum(alpha[t]*Q[i]['variable_cost_per_MWh']*inj[i,t] for i in G for t in T)+gp.quicksum(alpha[t]*seg['cost_per_MWh']*dem[seg['zone']][t]*nse[s] for s,seg in enumerate(segments) for t in T);policy=d['policy'];cost=fixed+variable+policy['RPS_noncompliance_cost_per_MWh']*rps.sum()+policy['CO2_noncompliance_cost_per_ton']*co2.sum();weighted=sum(alpha[t]*dem[z][t] for z in Z for t in T)
m.addConstr(gp.quicksum(alpha[t]*inj[i,t] for i in G if raw[i]['is_RPS_qualifying'] for t in T)+rps.sum()>=policy['RPS_share']*weighted);m.addConstr(gp.quicksum(alpha[t]*Q[i]['CO2_emission_tons_per_MWh']*inj[i,t] for i in G for t in T)-co2.sum()<=policy['CO2_cap_tons_per_MWh']*weighted);m.setObjective(cost);m.update();variables=m.getVars();best=float('inf');solution=None

def evaluate(e,v):
 if isinstance(e,(int,float)):return e
 if isinstance(e,gp.Var):return v[e.index]
 return e.getConstant()+sum(e.getCoeff(k)*v[e.getVar(k).index] for k in range(e.size()))
def pack(v):
 ev=lambda e:evaluate(e,v);values={};invest={};storage={};trans={}
 def put(name,ix,value):
  if abs(value)>1e-9:values[name+'['+','.join(map(str,ix))+']']=value
 for i in G:
  pn=ev(new[i]);pr=ev(ret[i]);pc=ev(cap[i]);put('y_P_new',(i,),pn);put('y_P_ret',(i,),pr);put('y_P',(i,),pc);invest[str(i)]={'resource_type':raw[i]['resource_type'],'zone':raw[i]['zone'],'new_units':pn,'new_MW':inc[i]*pn,'retired_units':pr,'retired_MW':inc[i]*pr,'total_capacity_MW':pc}
  if i in ST:
   energy=ev(ecap[i]);put('y_E_new',(i,),ev(en[i]));put('y_E_ret',(i,),ev(er[i]));put('y_E',(i,),energy);storage[str(i)]={'new_units':ev(en[i]),'new_MWh':einc*ev(en[i]),'total_energy_MWh':energy}
  share=pc/sum(ev(cap[j]) for j in UG[owner[i]]) if i in UC and sum(ev(cap[j]) for j in UG[owner[i]])>1e-9 else 0
  for t in T:
   put('x_inj',(i,t),share*ev(inj[owner[i],t]) if i in UC else ev(inj[i,t]))
   if i in UC:put('x_commit',(i,t),share*ev(commit[owner[i]])/raw[i]['capacity_size_MW'])
   if i in HY:put('x_lvl',(i,t),ev(lvl[i,t]));put('x_spill',(i,t),ev(spill[i,t]))
 for l in L:
  fnew=ev(fn[l]);f=lines[l]['existing_capacity_MW']+fnew;put('y_F_new',(l,),fnew);put('y_F',(l,),f);trans[str(l)]={'new_MW':fnew,'total_capacity_MW':f}
  for t in T:put('x_flow',(l,t),ev(flow[l]))
 for s,seg in enumerate(segments):
  for t in T:put('x_nse',(s,t),ev(nse[s])*dem[seg['zone']][t])
 for w in W:put('x_RPS',(w,),ev(rps[w]));put('x_CO2',(w,),ev(co2[w]))
 return {'objective_value':ev(cost),'model_variables':values,'investments':invest,'storage_energy_investments':storage,'transmission_investments':trans}
def publish(v):
 global best,solution
 if any(evaluate(row,v)>1e-6 for row in rows):return
 sol=pack(v)
 if sol['objective_value']<best:
  best=sol['objective_value'];solution=sol
  if log:log.log_solution(best,sol)
def cb(mm,where):
 if where==GRB.Callback.MIPSOL:publish(mm.cbGetSolution(variables))
active=set()
while time.monotonic()<deadline:
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 if not m.SolCount:break
 v=[z.X for z in variables];bad=sorted([(evaluate(row,v),k) for k,row in enumerate(rows) if k not in active and evaluate(row,v)>1e-7],reverse=True)
 if not bad:publish(v);break
 budget=1850-m.NumConstrs
 if budget<=0:break
 for _,k in bad[:min(100,budget)]:m.addConstr(rows[k]<=0);active.add(k)
if solution is None:raise RuntimeError('No complete feasible restricted-profile energy witness')
solution['runtime']=time.monotonic()-start;json.dump(solution,open(a.solution_path,'w'))

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import argparse,json,time,math,warnings
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from solution_logger import SolutionLogger

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--instance_path',required=True); ap.add_argument('--solution_path',required=True); ap.add_argument('--time_limit',type=int,required=True); ap.add_argument('--log_path'); a=ap.parse_args(); end=time.monotonic()+max(1,a.time_limit-3); d=json.load(open(a.instance_path)); T=d['time_periods']; G=d['generators']['generators']; E=d['electricity_network']; gas=d['gas_network']; beta=d['objective_weight_beta']; B=E['buses']; L=E['lines']; J=gas['junctions']; C=gas['connections']; js={j['id']:j for j in J}
 names=[]; ix={}; lbs=[]; ubs=[]; ints=[]; obj=[]; rows=[]; lo=[]; hi=[]; cones=[]
 def var(name,lb=0,ub=np.inf,integer=False,cost=0):
  ix[name]=len(names); names.append(name); lbs.append(lb); ubs.append(ub); ints.append(int(integer)); obj.append(cost); return name
 def row(terms,lb=-np.inf,ub=np.inf):
  z={}
  for name,c in terms: z[ix[name]]=z.get(ix[name],0)+c
  rows.append(z); lo.append(lb); hi.append(ub)
 def v(pre,i,t): return pre+'_'+str(i)+'_'+str(t)
 for g in G:
  u=g['id']; var(v('o',u,0),g['initial_status'],g['initial_status']); var(v('p',u,0),g['initial_gen'],g['initial_gen'])
  for t in range(1,T+1):
   var(v('o',u,t),0,1,True,beta*g['no_load_cost']); var(v('vp',u,t),0,1,True); var(v('vm',u,t),0,1,True); var(v('r',u,t),cost=beta); var(v('p',u,t),0,g['max_power'])
   if g['is_gfpp']: var(v('phimax',u,t),0,gas['max_gas_price_mmBtu'])
   for b in g['bids']: var('w_'+str(u)+'_'+str(b['id'])+'_'+str(t),0,1,True); var('se_'+str(u)+'_'+str(b['id'])+'_'+str(t),0,b['max_amount'],False,beta*b['price'])
 for t in range(1,T+1):
  for b in B: var(v('theta',b['id'],t),b['voltage_angle_lb'],b['voltage_angle_ub'])
  for l in L: var(v('f',l['id'],t),-l['thermal_limit'],l['thermal_limit'])
  for j in J:
   u=j['id']; demand=j['gas_demand_profile'][t-1]; var(v('sg',u,t),0,np.inf if j['is_source'] else 0); var(v('pisq',u,t),j['pressure_lb_squared'],j['pressure_ub_squared']); var(v('lg',u,t),0,demand); var(v('qg',u,t),0,demand,False,(1-beta)*j['demand_shedding_cost']); var(v('gamma',u,t))
   if j['is_source']:
    for si in j['supply_intervals']: var('sgs_'+str(u)+'_'+str(si['id'])+'_'+str(t),0,si['interval_ub']-si['interval_lb'],False,(1-beta)*si['slope'])
  for c in C:
   ub=np.inf
   if c['type']=='pipeline' and c['weymouth_factor']>0: ub=math.sqrt(max(0,js[c['from_junction']]['pressure_ub_squared']-js[c['to_junction']]['pressure_lb_squared'])/c['weymouth_factor'])
   var(v('phig',c['id'],t),0,ub)
  for z in gas['pricing_zones']: var(v('psi',z['id'],t),gas['min_gas_price_mmBtu'],gas['max_gas_price_mmBtu'])
 for g in G:
  u=g['id']
  for t in range(1,T+1):
   o=v('o',u,t); prev=v('o',u,t-1); pp=v('p',u,t-1); p=v('p',u,t); vp=v('vp',u,t); vm=v('vm',u,t)
   row([(vp,1),(vm,-1),(o,-1),(prev,1)],0,0); row([(vp,1),(vm,1)],ub=1)
   row([(v('vp',u,k),1) for k in range(max(1,t-g['min_up_time']+1),t+1)]+[(o,-1)],ub=0); row([(v('vm',u,k),1) for k in range(max(1,t-g['min_down_time']+1),t+1)]+[(o,1)],ub=1)
   for h,cost in g['startup_cost_params']:
    terms=[(v('r',u,t),1),(o,-cost)]; const=0
    for k in range(1,int(h)+1):
     if t-k>=0: terms.append((v('o',u,t-k),cost))
     else: const+=cost*g['initial_status']
    row(terms,lb=-const)
   row([(p,1),(o,-g['min_power'])],lb=0); row([(p,1),(o,-g['max_power'])],ub=0); row([(p,1),(pp,-1),(prev,-g['ramp_up']),(vp,-g['max_power'])],ub=0); row([(pp,1),(p,-1),(o,-g['ramp_down']),(vm,-g['min_power'])],ub=0)
   bids=g['bids']; se=[]
   for q,b in enumerate(bids):
    w='w_'+str(u)+'_'+str(b['id'])+'_'+str(t); z='se_'+str(u)+'_'+str(b['id'])+'_'+str(t); se.append((z,-1)); row([(w,1),(o,-1)],ub=0); row([(z,1),(w,-b['max_amount'])],ub=0)
    if q+1<len(bids): row([(z,1),('w_'+str(u)+'_'+str(bids[q+1]['id'])+'_'+str(t),-b['max_amount'])],lb=0)
   row([(p,1)]+se,0,0)
   if g['is_gfpp']:
    ph=v('phimax',u,t); M=gas['max_gas_price_mmBtu']; frac=g.get('max_gas_price_fraction')
    if frac is not None: row([(ph,1)],ub=frac*M)
    for z in gas['pricing_zones']:
     if g['gas_junction'] in z['junctions']: row([(v('psi',z['id'],t),1),(ph,-1),(o,M)],ub=M)
 for t in range(1,T+1):
  for l in L:
   tf=v('theta',l['from_bus'],t); tt=v('theta',l['to_bus'],t); row([(v('f',l['id'],t),1),(tf,-l['susceptance']),(tt,l['susceptance'])],0,0); row([(tf,1),(tt,-1)],-l['angle_diff_limit'],l['angle_diff_limit'])
  for b in B:
   u=b['id']; terms=[(v('p',g['id'],t),1) for g in G if g['bus']==u]+[(v('f',l['id'],t),1) for l in L if l['to_bus']==u]+[(v('f',l['id'],t),-1) for l in L if l['from_bus']==u]; row(terms,b['demand_profile'][t-1],b['demand_profile'][t-1])
  for j in J:
   u=j['id']; row([(v('lg',u,t),1),(v('qg',u,t),1)],j['gas_demand_profile'][t-1],j['gas_demand_profile'][t-1]); row([(v('sg',u,t),1),(v('lg',u,t),-1),(v('gamma',u,t),-1)]+[(v('phig',c['id'],t),1) for c in C if c['to_junction']==u]+[(v('phig',c['id'],t),-1) for c in C if c['from_junction']==u],0,0)
   if j['is_source']: row([(v('sg',u,t),1)]+[('sgs_'+str(u)+'_'+str(si['id'])+'_'+str(t),-1) for si in j['supply_intervals']],0,0)
   gf=[g for g in G if g['is_gfpp'] and g['gas_junction']==u]; qu=[]; lin=[(v('gamma',u,t),-1)]
   for g in gf:
    h=g['heat_rate_coefficients']; qu.append((v('p',g['id'],t),h['H_u2'])); lin.extend([(v('p',g['id'],t),h['H_u1']),(v('o',g['id'],t),h['H_u0'])])
   if gf: cones.append((qu,lin)); row(lin,ub=0)
   else: row([(v('gamma',u,t),1)],0,0)
  for c in C:
   pf=v('pisq',c['from_junction'],t); pt=v('pisq',c['to_junction'],t)
   if c['type']=='compressor': row([(pt,1),(pf,-c['compression_ratio_lb']**2)],lb=0); row([(pt,1),(pf,-c['compression_ratio_ub']**2)],ub=0)
   else:
    lin=[(pt,1),(pf,-1)]; qu=[(v('phig',c['id'],t),c['weymouth_factor'])]; cones.append((qu,lin)); row(lin,ub=0)
 def tangent(qu,lin,x):
  terms=lin[:]; rhs=0
  for name,c in qu:
   z=x[ix[name]]; terms.append((name,2*c*z)); rhs+=c*z*z
  row(terms,ub=rhs)
 for fraction in (.125,.25,.375,.5,.625,.75,.875,1.):
  seed=np.zeros(len(names))
  for qu,lin in cones:
   for name,c in qu:
    upper=ubs[ix[name]]; seed[ix[name]]=fraction*upper if np.isfinite(upper) else 0
   tangent(qu,lin,seed)
 # A secant envelope supplies a genuinely feasible conservative incumbent.
 base_count=len(rows)
 for qu,lin in cones:
  terms=lin[:]
  for name,c in qu: terms.append((name,c*ubs[ix[name]]))
  row(terms,ub=0)
 conservative=True
 best=None; log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 while time.monotonic()<end:
  rr=[]; cc=[]; dd=[]
  for r,z in enumerate(rows):
   for c,value in z.items(): rr.append(r); cc.append(c); dd.append(value)
  A=coo_matrix((dd,(rr,cc)),shape=(len(rows),len(names))).tocsr()
  with warnings.catch_warnings():
   warnings.simplefilter('ignore'); res=milp(np.array(obj),integrality=np.array(ints),bounds=Bounds(lbs,ubs),constraints=LinearConstraint(A,lo,hi),options={'time_limit':min(10.,max(.1,end-time.monotonic())),'threads':1,'mip_rel_gap':.005})
  if res.x is None:
   if conservative:
    rows=rows[:base_count]; lo=lo[:base_count]; hi=hi[:base_count]; conservative=False; continue
   break
  bad=[]
  for qu,lin in cones:
   nonlinear=sum(c*res.x[ix[name]]**2 for name,c in qu); linear=sum(c*res.x[ix[name]] for name,c in lin)
   if nonlinear+linear>1e-7*max(1,nonlinear,abs(linear)): bad.append((qu,lin))
  if not bad:
   primary={name:float(res.x[i]) for i,name in enumerate(names)}; best={'objective_value':float(np.dot(obj,res.x)),'released_problem_variant':d['released_problem_variant'],'objective_weight_beta':beta,'primary_variables':primary}
   if log: log.log_solution(best['objective_value'],best)
   if conservative:
    rows=rows[:base_count]; lo=lo[:base_count]; hi=hi[:base_count]; conservative=False; continue
   break
  if conservative:
   rows=rows[:base_count]; lo=lo[:base_count]; hi=hi[:base_count]; conservative=False
  for qu,lin in bad: tangent(qu,lin,res.x)
 if best is None: raise RuntimeError('Integrated-primal UCGNA outer approximation did not find a conic-feasible incumbent within time limit')
 json.dump(best,open(a.solution_path,'w'))
if __name__=='__main__': main()

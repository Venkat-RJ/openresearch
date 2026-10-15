import argparse,json,os,time
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

def exactsolve(d,limit):
 deadline=time.monotonic()+max(.1,limit-3)
 system=d['system'];sp=d['scenario_parameters'];gp=d['global_parameters'];tree=d['scenario_tree'];nodes={x['id']:x for x in tree};gens=system['generators'];lines=system['transmission_lines'];stores=system['storage'];H=sp['hours_per_day'];cost=[];hi=[];integer=[];rr=[];cc=[];vv=[];lb=[];ub=[];op={name:{} for name in ['alpha','gamma','p_gen','s_res','f_fwd','f_bwd','u_stor','v_stor','r_stor']};inv={};paths={};growth={tree[0]['id']:1.};cm={tree[0]['id']:1.};loads={b:sum(l['annual_peak_load_MW'] for l in system['loads'] if l['bus']==b) for b in system['buses']}
 def var(c=0,u=np.inf,it=0):j=len(cost);cost.append(c);hi.append(u);integer.append(it);return j
 def row(co,l=-np.inf,u=np.inf):
  z=len(lb);lb.append(l);ub.append(u)
  for j,v in co.items():
   if v:rr.append(z);cc.append(j);vv.append(v)
 def plus(co,j,v):co[j]=co.get(j,0)+v
 families={'generators':gens,'lines':lines,'storage':stores}
 def capacity(a):return a.get('Pmax_MW',a.get('flow_limit_MW',a.get('capacity_MW',0)))
 for node in sorted(tree,key=lambda x:x['stage']):
  nid=node['id'];parent=node['parent_id'];paths[nid]=([nid] if parent is None else paths[parent]+[nid])
  for k,ch in enumerate(node['children']):growth[ch]=growth[nid]*(1+sp['demand_growth_high_percent' if k==0 else 'demand_growth_low_percent']);cm[ch]=cm[nid]*(1+sp['cost_variation_percent']*(1 if k==0 else -1))
  inv[nid]={}
  for family,assets in families.items():
   inv[nid][family]={}
   for a in assets:
    if not a['existing']:inv[nid][family][a['id']]=var(node['probability']*cm[nid]*a['capital_cost_per_kW']*capacity(a)*1000,1,1)
 def avail(nid,family,a):return {} if a['existing'] else {inv[n][family][a['id']]:1 for n in paths[nid]}
 for node in tree:
  nid=node['id']
  for family,assets in families.items():
   for a in assets:
    if not a['existing']:row(avail(nid,family,a),u=1)
  existing=sum(gp['derating_factors'].get(g['type'],1)*g['Pmax_MW'] for g in gens if g['existing']);co={}
  for g in gens:
   if not g['existing']:
    for v in avail(nid,'generators',g):plus(co,v,gp['derating_factors'].get(g['type'],1)*g['Pmax_MW'])
  row(co,l=(1+gp['reserve_margin_requirement'])*sum(loads.values())*growth[nid]-existing)
  for day in d['representative_days']:
   did=day['id'];scenarios=d['lower_level_scenarios'][str(did)];weight=node['probability']/(len(d['representative_days'])*len(scenarios))
   for q,sc in enumerate(scenarios):
    def key(h,a):return f'{nid},{did},{q},{h},{a["id"]}'
    for h in range(H):
     for g in gens:
      kk=key(h,g);alpha=var(weight*g['a_cost'],1,1);gamma=var(weight*g['startup_cost'],1,1);p=var(weight*(g['b_cost']+g['c_cost']*g['Pmax_MW']));reserve=var();op['alpha'][kk]=alpha;op['gamma'][kk]=gamma;op['p_gen'][kk]=p;op['s_res'][kk]=reserve
      if not g['existing']:row({alpha:1,**{v:-1 for v in avail(nid,'generators',g)}},u=0)
      eff=g['Pmax_MW']
      if g['type']=='solar':eff*=day['solar_profile'][h]*sc['solar_multipliers'][h]
      if g['type']=='wind':eff*=sc['wind_power_factors'][h]
      row({p:1,reserve:1,alpha:-eff},u=0);row({alpha:g['Pmin_MW'],p:-1},u=0)
      co={alpha:1,gamma:-1}
      if h:co[op['alpha'][key(h-1,g)]]=-1
      row(co,u=0)
      if h:row({p:1,op['p_gen'][key(h-1,g)]:-1},-g['ramp_MW_per_h'],g['ramp_MW_per_h'])
     for a in lines:
      kk=key(h,a)
      for name in ['f_fwd','f_bwd']:
       v=var(0,a['flow_limit_MW']);op[name][kk]=v
       if not a['existing']:row({v:1,**{j:-a['flow_limit_MW'] for j in avail(nid,'lines',a)}},u=0)
     for a in stores:
      kk=key(h,a);u=var(0,a['capacity_MW']);v=var(0,a['capacity_MW']);r=var(0,a['capacity_MW']);op['u_stor'][kk]=u;op['v_stor'][kk]=v;op['r_stor'][kk]=r;row({u:1,r:-1},u=0)
      if not a['existing']:
       for j in [u,v,r]:row({j:1,**{z:-a['capacity_MW'] for z in avail(nid,'storage',a)}},u=0)
      if not h:row({r:1},0,0)
      else:row({r:1,op['r_stor'][key(h-1,a)]:-1,op['v_stor'][key(h-1,a)]:-a['efficiency'],op['u_stor'][key(h-1,a)]:1},0,0)
     for b in system['buses']:
      demand=loads[b]*growth[nid]*day['load_profile'][h]*sc['demand_multipliers'][h];balance={};reserveco={}
      for g in gens:
       if g['bus']==b:plus(balance,op['p_gen'][key(h,g)],1);plus(reserveco,op['s_res'][key(h,g)],1)
      for a in stores:
       if a['bus']==b:plus(balance,op['u_stor'][key(h,a)],1);plus(balance,op['v_stor'][key(h,a)],-1)
      for a in lines:
       kk=key(h,a);loss=a['loss_factor']
       if a['from_bus']==b:plus(balance,op['f_fwd'][kk],-1);plus(balance,op['f_bwd'][kk],1-loss)
       if a['to_bus']==b:plus(balance,op['f_fwd'][kk],1-loss);plus(balance,op['f_bwd'][kk],-1)
      row(balance,demand,demand);row(reserveco,l=gp['spinning_reserve_percent']*demand)
    for g in gens:
     for h in range(1,H):
      cur=op['alpha'][key(h,g)];prev=op['alpha'][key(h-1,g)]
      for tau in range(h,min(H,h+g['min_on_h'])):row({cur:1,prev:-1,op['alpha'][key(tau,g)]:-1} if tau!=h else {prev:-1},u=0)
      for tau in range(h,min(H,h+g['min_off_h'])):
       co={prev:1,cur:-1};plus(co,op['alpha'][key(tau,g)],1);row(co,u=1)
 A=coo_matrix((vv,(rr,cc)),shape=(len(lb),len(cost))).tocsc();res=milp(cost,integrality=integer,bounds=Bounds(0,hi),constraints=LinearConstraint(A,lb,ub),options={'time_limit':max(.05,deadline-time.monotonic()),'threads':1,'mip_rel_gap':.001})
 if res.x is None:raise RuntimeError('No feasible capacity expansion: '+res.message)
 x=res.x;out={name:{k:(round(x[j]) if name in ('alpha','gamma') else float(x[j])) for k,j in vals.items()} for name,vals in op.items()};out['investments']={str(n):{family:{str(a):round(x[j]) for a,j in values.items()} for family,values in fam.items()} for n,fam in inv.items()};out['objective_value']=float(np.dot(cost,x));return out

def solve(d,limit,solution_path=None):
 if len(d['system']['buses'])<30:return exactsolve(d,limit)
 import copy
 import highspy
 from scipy.sparse import block_diag,hstack,vstack,csr_matrix
 deadline=time.monotonic()+max(.1,limit-4)
 system=d['system'];sp=d['scenario_parameters'];gp=d['global_parameters'];tree=d['scenario_tree'];H=sp['hours_per_day'];gens=system['generators'];lines=system['transmission_lines'];stores=system['storage'];buses=system['buses'];nb=len(buses);busindex={b:j for j,b in enumerate(buses)};loads=np.array([sum(l['annual_peak_load_MW'] for l in system['loads'] if l['bus']==b) for b in buses]);families={'generators':gens,'lines':lines,'storage':stores};growth={tree[0]['id']:1.}
 for node in tree:
  for k,ch in enumerate(node['children']):growth[ch]=growth[node['id']]*(1+sp['demand_growth_high_percent' if k==0 else 'demand_growth_low_percent'])
 emergency={}
 for i,g in enumerate(gens):
  if g['existing'] and g['Pmin_MW']==0 and g['min_on_h']==0 and g['min_off_h']==0 and g['type'] not in ('solar','wind'):
   b=g['bus']
   if b not in emergency or g['Pmax_MW']>gens[emergency[b]]['Pmax_MW']:emergency[b]=i
 blocks=[]
 for node in tree:
  for day in d['representative_days']:
   ss=d['lower_level_scenarios'][str(day['id'])]
   for q,sc in enumerate(ss):
    demand=growth[node['id']]*np.array(day['load_profile'])[:,None]*np.array(sc['demand_multipliers'])[:,None]*loads[None,:]
    blocks.append((node,day,q,sc,demand,node['probability']/len(d['representative_days'])/len(ss)))
 for b,j in busindex.items():
  if loads[j] and b not in emergency:return exactsolve(d,max(.1,deadline-time.monotonic()))
  if b in emergency:
   g=gens[emergency[b]]
   if any(np.max(dm[:,j])*(1+gp['spinning_reserve_percent'])>g['Pmax_MW'] or np.max(np.abs(np.diff(dm[:,j])))>g['ramp_MW_per_h'] for node,day,q,sc,dm,w in blocks):return exactsolve(d,max(.1,deadline-time.monotonic()))
 existing_derated=sum(gp['derating_factors'].get(g['type'],1)*g['Pmax_MW'] for g in gens if g['existing'])
 if any(existing_derated<(1+gp['reserve_margin_requirement'])*loads.sum()*growth[node['id']] for node in tree):return exactsolve(d,max(.1,deadline-time.monotonic()))
 names=['alpha','gamma','p_gen','s_res','f_fwd','f_bwd','u_stor','v_stor','r_stor'];out={name:{} for name in names};out['investments']={str(node['id']):{family:{str(a['id']):0 for a in assets if not a['existing']} for family,assets in families.items()} for node in tree};out['objective_value']=0.;emset=set(emergency.values());marg=np.array([g['b_cost']+g['c_cost']*g['Pmax_MW'] for g in gens]);basecost=[];savedobj=float('inf')
 def emit():
  nonlocal savedobj
  if solution_path and out['objective_value']<savedobj:
   with open(solution_path,'w') as f:json.dump(out,f,separators=(',',':'))
   savedobj=out['objective_value']
 for node,day,q,sc,dm,w in blocks:
  value=0.
  for h in range(H):
   prefix=f'{node["id"]},{day["id"]},{q},{h},'
   for i,g in enumerate(gens):
    k=prefix+str(g['id']);a=int(i in emset);p=float(dm[h,busindex[g['bus']]]) if a else 0.;r=gp['spinning_reserve_percent']*p;out['alpha'][k]=a;out['gamma'][k]=a if h==0 else 0;out['p_gen'][k]=p;out['s_res'][k]=r;value+=g['a_cost']*a+g['startup_cost']*(a if h==0 else 0)+marg[i]*p
   for line in lines:
    k=prefix+str(line['id']);out['f_fwd'][k]=0.;out['f_bwd'][k]=0.
   for a in stores:
    k=prefix+str(a['id'])
    for name in ['u_stor','v_stor','r_stor']:out[name][k]=0.
  basecost.append(value);out['objective_value']+=w*value
 emit()
 adjacency={b:set() for b in buses}
 for line in lines:
  if line['existing']:adjacency[line['from_bus']].add(line['to_bus']);adjacency[line['to_bus']].add(line['from_bus'])
 live={b for b,j in busindex.items() if loads[j]>0};todo=list(live)
 while todo:
  b=todo.pop()
  for neighbor in adjacency[b]-live:live.add(neighbor);todo.append(neighbor)
 active=[i for i,g in enumerate(gens) if g['existing'] and (g['bus'] in live or i in emset)];ng=len(active);gi={i:j for j,i in enumerate(active)};nl=len(lines);nv=H*(ng+nl);rr=[];cc=[];vv=[];rlo=[];rhi=[];cost=np.zeros(nv)
 def row(co,lo=-np.inf,hi=np.inf):
  z=len(rlo);rlo.append(lo);rhi.append(hi)
  for j,v in co.items():rr.append(z);cc.append(j);vv.append(v)
 for h in range(H):
  for b in buses:
   co={h*ng+gi[i]:1 for i in active if gens[i]['bus']==b}
   for j,line in enumerate(lines):
    if line['loss_factor']!=0:return exactsolve(d,max(.1,deadline-time.monotonic()))
    if line['from_bus']==b:co[H*ng+h*nl+j]=-1
    if line['to_bus']==b:co[H*ng+h*nl+j]=1
   row(co,0,0)
 for h in range(1,H):
  for i in active:
   g=gens[i]
   if g['ramp_MW_per_h']<g['Pmax_MW']:row({h*ng+gi[i]:1,(h-1)*ng+gi[i]:-1},-g['ramp_MW_per_h'],g['ramp_MW_per_h'])
 A=coo_matrix((vv,(rr,cc)),shape=(len(rlo),nv)).tocsr();rlo=np.array(rlo);rhi=np.array(rhi);low=np.zeros(nv);upper=np.zeros(nv)
 for h in range(H):
  for i in active:cost[h*ng+gi[i]]=marg[i]
 constant=sum(H*g['a_cost']+g['startup_cost'] for i,g in enumerate(gens) if i in active)
 linecaps=np.array([line['flow_limit_MW'] for line in lines]);candidate=[j for j,line in enumerate(lines) if not line['existing']];openlines=np.array([int(line['existing']) for line in lines])
 def parameters(block,opened):
  node,day,q,sc,dm,w=block;lo=low.copy();up=upper.copy();rl=rlo.copy();ru=rhi.copy();rl[:H*nb]=dm.ravel();ru[:H*nb]=dm.ravel()
  for h in range(H):
   for i in active:
    g=gens[i];j=h*ng+gi[i];lo[j]=g['Pmin_MW'];cap=g['Pmax_MW']
    if g['type']=='solar':cap*=day['solar_profile'][h]*sc['solar_multipliers'][h]
    elif g['type']=='wind':cap*=sc['wind_power_factors'][h]
    if i in emset:cap-=gp['spinning_reserve_percent']*dm[h,busindex[g['bus']]]
    up[j]=cap
   lo[H*ng+h*nl:H*ng+(h+1)*nl]=-linecaps*opened;up[H*ng+h*nl:H*ng+(h+1)*nl]=linecaps*opened
  return lo,up,rl,ru
 lo,up,rl,ru=parameters(blocks[0],openlines);lp=highspy.HighsLp();lp.num_col_=nv;lp.num_row_=A.shape[0];lp.col_cost_=cost;lp.col_lower_=lo;lp.col_upper_=up;lp.row_lower_=rl;lp.row_upper_=ru;lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise;lp.a_matrix_.start_=A.indptr;lp.a_matrix_.index_=A.indices;lp.a_matrix_.value_=A.data;solver=highspy.Highs();solver.setOptionValue('threads',1);solver.setOptionValue('output_flag',False);solver.setOptionValue('solver','simplex');solver.passModel(lp);allcols=np.arange(nv,dtype=np.int32);allrows=np.arange(A.shape[0],dtype=np.int32);lastsave=time.monotonic()
 def dispatch(opened):
  nonlocal lastsave
  for z,block in enumerate(blocks):
   remaining=deadline-time.monotonic()
   if remaining<=.15:break
   lo,up,rl,ru=parameters(block,opened);solver.changeColsBounds(nv,allcols,lo,up);solver.changeRowsBounds(A.shape[0],allrows,rl,ru);solver.setOptionValue('time_limit',max(.01,min(1.5,remaining)));solver.run();solution=solver.getSolution()
   if not solution.value_valid or solver.getInfo().primal_solution_status!=highspy.SolutionStatus.kSolutionStatusFeasible:continue
   x=np.array(solution.col_value);value=float(cost@x)+constant
   if value>=basecost[z]-1e-6:continue
   node,day,q,sc,dm,w=block;out['objective_value']+=w*(value-basecost[z]);basecost[z]=value
   for h in range(H):
    prefix=f'{node["id"]},{day["id"]},{q},{h},'
    for i in active:
     g=gens[i];k=prefix+str(g['id']);out['alpha'][k]=1;out['gamma'][k]=int(h==0);out['p_gen'][k]=float(x[h*ng+gi[i]]);out['s_res'][k]=float(gp['spinning_reserve_percent']*dm[h,busindex[g['bus']]]) if i in emset else 0.
    for j,line in enumerate(lines):
     k=prefix+str(line['id']);flow=float(x[H*ng+h*nl+j]);out['f_fwd'][k]=max(0.,flow);out['f_bwd'][k]=max(0.,-flow)
   if time.monotonic()-lastsave>5:emit();lastsave=time.monotonic()
 dispatch(openlines);emit()
 if len(tree)==1 and candidate and deadline-time.monotonic()>8:
  sample=[blocks[int(j*(len(blocks)-1)/3)] for j in range(4)];nc=len(candidate);mats=[];lbs=[];ubs=[];los=[];ups=[]
  for block in sample:
   lo,up,rl,ru=parameters(block,np.ones(nl));mats.append(A);lbs.extend(rl);ubs.extend(ru);los.extend(lo);ups.extend(up)
  master=hstack([block_diag(mats,format='csr'),csr_matrix((len(lbs),nc))],format='csr');cr=[];cv=[];ci=[];r=0
  for s in range(len(sample)):
   for h in range(H):
    for k,j in enumerate(candidate):
     col=s*nv+H*ng+h*nl+j
     for sign in [1,-1]:cr.extend([r,r]);ci.extend([col,len(sample)*nv+k]);cv.extend([sign,-linecaps[j]]);r+=1
  links=coo_matrix((cv,(cr,ci)),shape=(r,len(sample)*nv+nc)).tocsr();master=vstack([master,links],format='csc');capital=np.array([lines[j]['capital_cost_per_kW']*linecaps[j]*1000 for j in candidate]);mcost=np.concatenate([np.tile(cost/len(sample),len(sample)),capital]);integ=np.zeros(len(mcost));integ[-nc:]=1;res=milp(mcost,integrality=integ,bounds=Bounds(np.r_[los,np.zeros(nc)],np.r_[ups,np.ones(nc)]),constraints=LinearConstraint(master,np.r_[lbs,np.full(r,-np.inf)],np.r_[ubs,np.zeros(r)]),options={'threads':1,'time_limit':max(.05,min(7,(deadline-time.monotonic())*.25)),'mip_rel_gap':.01})
  if res.x is not None:
   selected=np.rint(res.x[-nc:]).astype(int)
   if selected.any():
    previous=copy.deepcopy(out);previous_cost=out['objective_value'];openlines[candidate]=selected
    for k,j in enumerate(candidate):out['investments'][str(tree[0]['id'])]['lines'][str(lines[j]['id'])]=int(selected[k])
    out['objective_value']+=float(capital@selected);dispatch(openlines)
    if out['objective_value']>=previous_cost:out=previous
 emit();return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 out=solve(json.load(open(a.instance_path)),a.time_limit,a.solution_path)
 with open(a.solution_path,'w') as f:json.dump(out,f,separators=(',',':'))

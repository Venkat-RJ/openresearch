import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def solve(d,t):
 products={p['id']:p for p in d['products']};fac=d['facilities'];pf={(a['product_id'],a['facility_id']):a for a in d['product_facility_data']};distributions={(a['product_id'],dd['distribution_id']):dd for a in d['products_distributions'] for dd in a['distributions']};keys=list(distributions);var={};cost=[];integ=[];hi=[]
 def add(key,c=0,binary=False,upper=np.inf):var[key]=len(cost);cost.append(c);integ.append(int(binary));hi.append(1 if binary else upper)
 for p,dd in keys:
  dist=distributions[p,dd];add(('d',p,dd),binary=True)
  for f in fac:add(('x',p,dd,f['id']),pf[p,f['id']]['manufacturing_cost'])
  for s in dist['scenarios']:
   add(('w',p,dd,s['scenario_id']),-s['probability']*products[p]['sale_price']);add(('o',p,dd,s['scenario_id']),-s['probability']*products[p]['discounted_price'])
 rows=[];lb=[];ub=[]
 def row(rr,l,u):rows.append({var[k]:v for k,v in rr.items()});lb.append(l);ub.append(u)
 for p in products:row({('d',pp,dd):1 for pp,dd in keys if pp==p},1,1)
 for f in fac:row({('x',p,dd,f['id']):1 for p,dd in keys},-np.inf,f['capacity'])
 for p,dd in keys:
  dist=distributions[p,dd];combo=dist['level_combination']
  for j,f in enumerate(fac):
   levels=pf[p,f['id']]['production_levels'];lev=next(l for l in levels if l['level_id']==combo[j]) if combo is not None else next(l for l in levels if l['lower_bound']==0)
   row({('x',p,dd,f['id']):1,('d',p,dd):-lev['upper_bound']},-np.inf,0);row({('x',p,dd,f['id']):1,('d',p,dd):-lev['lower_bound']},0,np.inf)
  for s in dist['scenarios']:
   sid=s['scenario_id'];row({('w',p,dd,sid):1,('d',p,dd):-s['demand']},-np.inf,0);rr={('w',p,dd,sid):1,('o',p,dd,sid):1}
   for f in fac:rr['x',p,dd,f['id']]=-s.get('yields',{}).get(str(f['id']),0)
   row(rr,0,0)
 A=lil_matrix((len(rows),len(cost)))
 for i,rr in enumerate(rows):
  for j,v in rr.items():A[i,j]=v
 result=milp(cost,integrality=integ,bounds=Bounds(np.zeros(len(cost)),hi),constraints=LinearConstraint(A.tocsr(),lb,ub),options={'threads':1,'time_limit':max(.1,t-1)})
 if result.x is None:raise RuntimeError('No endogenous yield manufacturing incumbent')
 values=result.x;solution={'x':{},'y':{},'delta':{},'z':{},'w':{},'o':{}}
 for p in products:
  chosen=next(dd for pp,dd in keys if pp==p and values[var['d',p,dd]]>.5);combo=distributions[p,chosen]['level_combination']
  for j,f in enumerate(fac):
   fid=f['id'];allocation=max(0,sum(values[var['x',p,dd,fid]] for pp,dd in keys if pp==p));solution['x'][f'{p}_{fid}']=float(allocation);levels=pf[p,fid]['production_levels'];level=combo[j] if combo is not None else next(l['level_id'] for l in levels if l['lower_bound']==0)
   for l in levels:solution['y'][f'{p}_{fid}_{l["level_id"]}']=int(l['level_id']==level)
  for pp,dd in keys:
   if pp!=p:continue
   solution['delta'][f'{p}_{dd}']=int(dd==chosen)
   for s in distributions[p,dd]['scenarios']:
    key=f'{p}_{dd}_{s["scenario_id"]}';zz=sum(s.get('yields',{}).get(str(f['id']),0)*solution['x'][f'{p}_{f["id"]}'] for f in fac);ww=min(zz,s['demand']);solution['z'][key]=zz;solution['w'][key]=ww;solution['o'][key]=zz-ww
 obj=-sum(pf[p,f['id']]['manufacturing_cost']*solution['x'][f'{p}_{f["id"]}'] for p in products for f in fac)
 for p,dd in keys:
  if not solution['delta'][f'{p}_{dd}']:continue
  for s in distributions[p,dd]['scenarios']:
   key=f'{p}_{dd}_{s["scenario_id"]}';obj+=s['probability']*(products[p]['sale_price']*solution['w'][key]+products[p]['discounted_price']*solution['o'][key])
 return {'objective_value':obj,'solution':solution}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));g=d['generator'];tree=d['scenario_tree'];nodes=tree['nodes'];n=len(nodes);lookup={v['id']:i for i,v in enumerate(nodes)};upper=g['C_upper_MW'];lower=g['C_lower_MW'];fuel_a=g['fuel_a_dollars_per_h'];fuel_b=g['fuel_b_dollars_per_MWh'];startup=g['U_bar_startup_cost_dollars'];shutdown=g['U_lower_shutdown_cost_dollars'];c=np.zeros(3*n);rows=[];cols=[];vals=[];lo=[];hi=[]
 def add(terms,l=-np.inf,h=np.inf):
  r=len(lo)
  for j,v in terms.items():rows.append(r);cols.append(j);vals.append(v)
  lo.append(l);hi.append(h)
 for i,v in enumerate(nodes):
  prob=v['probability'];c[i]=-prob*(v['electricity_price_dollars_per_MWh']-fuel_b);c[n+i]+=prob*fuel_a;add({i:1,n+i:-upper},h=0);add({i:1,n+i:-lower},l=0);parent=v['parent_id']
  if parent is None:continue
  j=lookup[parent];c[2*n+i]+=prob*(startup+shutdown);c[n+j]+=prob*shutdown;c[n+i]-=prob*shutdown;add({2*n+i:1,n+i:-1,n+j:1},l=0);add({2*n+i:1,n+i:-1},h=0);add({2*n+i:1,n+j:1},h=1);add({i:1,j:-1,n+j:upper-g['V_plus_MW_per_h']},h=upper);add({j:1,i:-1,n+i:upper-g['V_minus_MW_per_h']},h=upper)
  for length,up in [(tree['L_min_up_time'],True),(tree['ell_min_down_time'],False)]:
   stack=[(i,0)]
   while stack:
    k,depth=stack.pop()
    if depth>=length:continue
    if k!=i:
     if up:add({n+i:1,n+j:-1,n+k:-1},h=0)
     else:add({n+j:1,n+i:-1,n+k:1},h=1)
    stack.extend((lookup[ch],depth+1)for ch in nodes[k]['children_ids'])
 x=np.zeros(3*n)
 if n<=20000:
  A=coo_matrix((vals,(rows,cols)),shape=(len(lo),3*n)).tocsc();bounds=np.concatenate((np.full(n,upper),np.ones(2*n)));integer=np.concatenate((np.zeros(n),np.ones(2*n)));r=milp(c,integrality=integer,bounds=Bounds(0,bounds),constraints=LinearConstraint(A,lo,hi),options={'threads':1,'time_limit':max(.1,float(a.time_limit)-2-(time.monotonic()-start))})
  if r.x is not None:x=r.x
 for level in [lower,upper]:
  candidate=np.concatenate((np.full(n,level),np.ones(n),np.zeros(n)))
  if c@candidate<c@x:x=candidate
 json.dump({'objective_value':float(-c@x),'x':{str(v['id']):float(x[i])for i,v in enumerate(nodes)if x[i]},'y':{str(v['id']):int(round(x[n+i]))for i,v in enumerate(nodes)if x[n+i]>.5},'u':{str(v['id']):int(round(x[2*n+i]))for i,v in enumerate(nodes)if v['parent_id']is not None and x[2*n+i]>.5}},open(a.solution_path,'w'))
if __name__=='__main__':main()

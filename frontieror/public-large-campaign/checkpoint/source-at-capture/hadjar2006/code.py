import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def solve(d,t):
 deps=d['depots'];tasks=d['tasks'];arcs=[];cost=[]
 for dep in deps:
  k=dep['id']
  for a in d['depot_to_task_costs']:
   if a['depot_id']==k:arcs.append((k,None,a['task_id']));cost.append(a['cost'])
  for a in d['compatible_pairs']:arcs.append((k,a['task_i'],a['task_j']));cost.append(a['cost'])
  for a in d['task_to_depot_costs']:
   if a['depot_id']==k:arcs.append((k,a['task_id'],None));cost.append(a['cost'])
 N=len(arcs);rows=[];lb=[];ub=[]
 for task in tasks:rows.append({i:1 for i,(k,u,v) in enumerate(arcs) if v==task['id']});lb.append(1);ub.append(1)
 for dep in deps:
  k=dep['id']
  for task in tasks:
   v=task['id'];rows.append({i:1 if vv==v else -1 for i,(kk,uu,vv) in enumerate(arcs) if kk==k and (vv==v or uu==v)});lb.append(0);ub.append(0)
  rows.append({i:1 for i,(kk,u,v) in enumerate(arcs) if kk==k and u is None});lb.append(-np.inf);ub.append(dep['capacity'])
 A=lil_matrix((len(rows),N))
 for r,rr in enumerate(rows):
  for i,v in rr.items():A[r,i]=v
 z=milp(cost,integrality=np.ones(N),bounds=Bounds(np.zeros(N),np.ones(N)),constraints=LinearConstraint(A.tocsr(),lb,ub),options={'threads':1,'time_limit':max(.1,t-1)})
 if z.x is None:raise RuntimeError('No multidepot vehicle incumbent')
 chosen=[(a,cost[i]) for i,a in enumerate(arcs) if z.x[i]>.5];routes=[]
 for (k,u,v),co in chosen:
  if u is not None:continue
  sequence=[v];total=co
  while True:
   (kk,uu,vv),cc=next((a,c) for a,c in chosen if a[0]==k and a[1]==sequence[-1]);total+=cc
   if vv is None:break
   sequence.append(vv)
  routes.append({'depot':k,'tasks':sequence,'cost':total})
 return {'objective_value':sum(r['cost'] for r in routes),'routes':routes}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

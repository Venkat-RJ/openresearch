import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def solve(d,t):
 start=time.monotonic()
 n=d['num_orders'];k=d['num_slabs'];sw=d['slab_weights'];ow=d['order_weights'];color=d['order_colors']
 totals=[{} for _ in range(k)]
 for i,slabs in enumerate(d['eligible_slabs_per_order']):
  for j in slabs:totals[j][color[i]]=totals[j].get(color[i],0)+ow[i]
 active={j for j in range(k) if 2*sum(sorted(totals[j].values(),reverse=True)[:d['max_colors_per_knapsack']])>sw[j]+1e-8}
 if not active:return {'objective_value':0,'assignments':[],'slabs_used':[]}
 pairs=[(i,j) for i in range(n) for j in d['eligible_slabs_per_order'][i] if j in active];p=len(pairs);cs=sorted(set(color));cp=[(j,c) for j in range(k) for c in cs];ci={v:p+k+i for i,v in enumerate(cp)};N=p+k+len(cp);cost=np.zeros(N);cost[:p]=[-2*ow[i] for i,j in pairs];cost[p:p+k]=sw;rows=[];ub=[]
 by_order=[{} for _ in range(n)];by_slab=[{} for _ in range(k)]
 for h,(i,j) in enumerate(pairs):by_order[i][h]=1;by_slab[j][h]=ow[i]
 def row(v,b):rows.append(v);ub.append(b)
 for i in range(n):row(by_order[i],1)
 for j in range(k):row({**by_slab[j],p+j:-sw[j]},0);row({ci[j,c]:1 for c in cs},d['max_colors_per_knapsack'])
 for h,(i,j) in enumerate(pairs):row({h:1,ci[j,color[i]]:-1},0)
 A=lil_matrix((len(rows),N))
 for r,rr in enumerate(rows):
  for i,v in rr.items():A[r,i]=v
 result=milp(cost,integrality=np.ones(N),bounds=Bounds(np.zeros(N),np.ones(N)),constraints=LinearConstraint(A.tocsr(),-np.inf,np.array(ub)),options={'time_limit':max(.1,t-8-(time.monotonic()-start)),'threads':1})
 if result.x is None:return {'objective_value':0,'assignments':[],'slabs_used':[]}
 assignments=[{'order':i,'slab':j} for h,(i,j) in enumerate(pairs) if result.x[h]>.5];used=sorted(set(a['slab'] for a in assignments));obj=2*sum(ow[a['order']] for a in assignments)-sum(sw[j] for j in used)
 return {'objective_value':obj,'assignments':assignments,'slabs_used':used}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=__import__('time').time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':__import__('time').time()-start,'objective_value':r['objective_value']})+'\n')

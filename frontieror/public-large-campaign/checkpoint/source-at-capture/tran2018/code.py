import argparse,json,os,time
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

def solve(d,limit):
 nodes=[a['id'] for a in d['nodes']];pos={u:i for i,u in enumerate(nodes)};n=len(nodes);od=d['od_pairs'];R=d['parameters']['R'];p=d['parameters']['p'];dist={}
 for a in d['arcs']:dist[a['from'],a['to']]=a['distance'];dist[a['to'],a['from']]=a['distance']
 sets=[]
 for q in od:
  path=q['shortest_path'];rt=path+list(reversed(path[:-1]));cum=[0.]
  for i in range(len(rt)-1):cum.append(cum[-1]+dist[rt[i],rt[i+1]])
  sets.append([{rt[c] for c in range(a+1) if cum[a+1]-cum[c]<=R+1e-9} for a in range(len(rt)-1) if cum[a+1]>R+1e-9])
 rr=[];cc=[];vv=[];lb=[];ub=[]
 def row(co,l,u):
  z=len(lb);lb.append(l);ub.append(u)
  for j,v in co.items():rr.append(z);cc.append(j);vv.append(v)
 row({i:1 for i in range(n)},p,p)
 for q,ss in enumerate(sets):
  for s in ss:row({**{pos[u]:1 for u in s},n+q:-1},0,np.inf)
 c=np.array([0.]*n+[-q['flow'] for q in od]);A=coo_matrix((vv,(rr,cc)),shape=(len(lb),len(c))).tocsc();res=milp(c,integrality=np.ones(len(c)),bounds=Bounds(0,1),constraints=LinearConstraint(A,lb,ub),options={'time_limit':max(.01,limit-1),'threads':1})
 if res.x is None:stations=nodes[:p]
 else:stations=[u for i,u in enumerate(nodes) if res.x[i]>.5]
 chosen=set(stations);served=[q for q,ss in enumerate(sets) if all(s&chosen for s in ss)]
 return {'objective_value':sum(od[q]['flow'] for q in served),'stations':stations,'refueled_od_pairs':served}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)

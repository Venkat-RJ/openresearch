import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def solve(d,t):
 patients=d['patients'];slots=[(h,day,o) for h in d['hospitals'] for day in h['days'] for o in range(d['num_ors_per_hospital'])];suites=[(h,day) for h in d['hospitals'] for day in h['days']];n=len(patients);L=len(slots);U=len(suites);S=d['num_scenarios'];N=U+L+n*L+n+S*n*L;u=lambda h,day:next(i for i,(hh,dd) in enumerate(suites) if hh['hospital_id']==h and dd['day_id']==day);y=lambda l:U+l;x=lambda i,l:U+L+i*L+l;w=lambda i:U+L+n*L+i;z=lambda s,i,l:U+L+n*L+n+s*n*L+i*L+l;c=np.zeros(N);lo=np.zeros(N);hi=np.ones(N);rows=[];lb=[];ub=[]
 def row(rr,l,r):rows.append(rr);lb.append(l);ub.append(r)
 for j,(h,day) in enumerate(suites):c[j]=day['G_hd']
 for l,(h,day,o) in enumerate(slots):
  c[y(l)]=day['F_hd'];row({y(l):1,u(h['hospital_id'],day['day_id']):-1},-np.inf,0)
 for i,p in enumerate(patients):
  if p['is_mandatory']:hi[w(i)]=0
  else:c[w(i)]=p['c_unsched']
  row({**{x(i,l):1 for l in range(L)},w(i):1},1,1)
  for l,(h,day,o) in enumerate(slots):
   c[x(i,l)]=p['c_sched_per_day'][str(day['day_label'])]+p['c_cancel'];row({x(i,l):1,y(l):-1},-np.inf,0)
   for s in range(S):c[z(s,i,l)]=-p['c_cancel']/S;row({z(s,i,l):1,x(i,l):-1},-np.inf,0)
 for s,sc in enumerate(d['scenarios']):
  for l,(h,day,o) in enumerate(slots):row({z(s,i,l):sc['surgery_durations_minutes'][i] for i in range(n)},-np.inf,day['B_hd'])
 A=lil_matrix((len(rows),N))
 for r,rr in enumerate(rows):
  for j,v in rr.items():A[r,j]=v
 result=milp(c,integrality=np.ones(N),bounds=Bounds(lo,hi),constraints=LinearConstraint(A.tocsr(),lb,ub),options={'threads':1,'time_limit':max(.1,t-1)})
 if result.x is None:raise RuntimeError('No surgery scheduling incumbent')
 xx=result.x;r={'objective_value':float(c@xx),'u':{f'{h["hospital_id"]},{day["day_id"]}':int(round(xx[j])) for j,(h,day) in enumerate(suites)},'y':{f'{h["hospital_id"]},{day["day_id"]},{o}':int(round(xx[y(l)])) for l,(h,day,o) in enumerate(slots)},'x':{f'{h["hospital_id"]},{day["day_id"]},{p["patient_id"]},{o}':1 for i,p in enumerate(patients) for l,(h,day,o) in enumerate(slots) if xx[x(i,l)]>.5},'w':{str(p['patient_id']):int(round(xx[w(i)])) for i,p in enumerate(patients) if not p['is_mandatory']}}
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

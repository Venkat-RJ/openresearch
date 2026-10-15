import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,itertools,math
import numpy as np

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();deadline=time.monotonic()+max(.1,float(a.time_limit)-2);d=json.load(open(a.instance_path));n=d['n_nodes'];count=d['p'];D=np.array(d['distance_matrix']);W=np.array(d['demand_matrix']);wait=np.array([v['waiting_time']for v in d['nodes']]);q=d['parameters'];cache={}
 def util(k,l):
  key=(min(k,l),max(k,l))
  if key in cache:return cache[key]
  def one(k,l):
   if k==l:dist=D[:,k,None]+D[k,None,:];t=dist+wait[:,None]+wait[k];fare=q['cost_scalar']*dist;A=q['attractiveness_1hub']
   else:dist=D[:,k,None]+D[k,l]+D[l,None,:];t=dist+wait[:,None]+wait[k]+wait[l];fare=q['cost_scalar']*(D[:,k,None]+q['alpha']*D[k,l]+D[l,None,:]);A=q['attractiveness_2hub']
   den=q['gamma']*t**q['beta']+(1-q['gamma'])*fare**q['lambda'];return np.divide(A,den,out=np.zeros_like(den),where=den>0)
  v=one(k,l)if k==l else np.maximum(one(k,l),one(l,k));cache[key]=v;return v
 comp=sum((util(k,l)for k,l in itertools.combinations_with_replacement(d['competitor_hub_locations'],2)),np.zeros((n,n)))
 def score(hubs):
  own=sum((util(k,l)for k,l in itertools.combinations_with_replacement(hubs,2)),np.zeros((n,n)));return float(np.sum(W*np.divide(own,own+comp,out=np.zeros_like(own),where=own+comp>0)))
 hubs=[]
 while len(hubs)<count:
  j=max((j for j in range(n)if j not in hubs),key=lambda j:score(hubs+[j]));hubs.append(j)
 best=score(hubs)
 if math.comb(n,count)<=200000:
  for hh in itertools.combinations(range(n),count):
   if time.monotonic()>deadline:break
   value=score(hh)
   if value>best:best=value;hubs=list(hh)
 else:
  improved=True
  while improved and time.monotonic()<deadline:
   improved=False
   for i in range(count):
    for j in range(n):
     if j in hubs or time.monotonic()>deadline:continue
     hh=hubs.copy();hh[i]=j;value=score(hh)
     if value>best:best=value;hubs=hh;improved=True
 json.dump({'objective_value':best,'hub_locations':hubs},open(a.solution_path,'w'))
if __name__=='__main__':main()

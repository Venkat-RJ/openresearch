import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import argparse,json,time,random
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-3);z=json.load(open(a.instance_path));d=np.array(z['cost_matrix'],dtype=np.int64);n=z['n'];p=z['p'];rng=random.Random(0);best=float('inf');fac=None;logger=SolutionLogger(a.log_path,sense='minimize')
logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
def save(f):
 global best,fac
 nearest=np.argmin(d[:,f],axis=1);value=int(d[np.arange(n),np.array(f)[nearest]].sum())
 if value>=best:return
 best=value;fac=f.copy();s={'objective_value':value,'opened_facilities':f,'assignments':[{'customer':i,'facility':int(f[nearest[i]]),'cost':int(d[i,f[nearest[i]]])} for i in range(n)],'runtime':time.monotonic()-start};json.dump(s,open(a.solution_path,'w'));logger.log_solution(value,s)
def descend(f):
 save(f)
 while time.monotonic()<end:
  current=int(d[:,f].min(axis=1).sum());move=None;gain=0
  for k in range(p):
   remaining=f[:k]+f[k+1:];base=d[:,remaining].min(axis=1) if remaining else np.full(n,10**15);values=np.minimum(base[:,None],d).sum(axis=0);values[f]=10**15;j=int(values.argmin());improvement=current-int(values[j])
   if improvement>gain:gain=improvement;move=(k,j)
  if move is None:return f
  f[move[0]]=move[1];save(f)
 return f
f=[int(d.sum(axis=0).argmin())]
while len(f)<p:
 values=np.minimum(d[:,f].min(axis=1)[:,None],d).sum(axis=0);values[f]=10**15;f.append(int(values.argmin()))
descend(f)
while time.monotonic()<end:
 f=fac.copy();keep=max(0,p-rng.randint(1,max(1,p//2)));f=rng.sample(f,keep);f+=rng.sample([j for j in range(n) if j not in f],p-keep);descend(f)

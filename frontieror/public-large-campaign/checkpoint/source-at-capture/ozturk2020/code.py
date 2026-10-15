import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));jobs=d['jobs'];M=d['num_machines'];cap=d['batch_capacity'];deadline=time.monotonic()+max(.1,a.time_limit-2);rng=random.Random(0);best=float('inf');logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
for it in range(100000):
 if time.monotonic()>deadline:break
 left=set(range(len(jobs)));free=[0]*M;counts=[0]*M;batches=[];assign={};obj=0
 while left:
  k=min(range(M),key=lambda k:free[k]);i=min(left,key=lambda i:(max(free[k],jobs[i]['release_date'])+jobs[i]['processing_time'])*(1 if not it else rng.uniform(.7,1.3)));selected=[i];left.remove(i);weight=jobs[i]['size'];start=max(free[k],jobs[i]['release_date']);pp=jobs[i]['processing_time']
  candidates=sorted(left,key=lambda j:abs(jobs[j]['processing_time']-pp)+max(0,jobs[j]['release_date']-start))
  for j in candidates:
   jj=jobs[j]
   if weight+jj['size']>cap:continue
   ss=max(start,jj['release_date']);pr=max(pp,jj['processing_time']);old=(start+pp)*len(selected)+max(free[k],jj['release_date'])+jj['processing_time'];new=(ss+pr)*(len(selected)+1)
   if new<=old*(1 if not it else rng.uniform(.95,1.15)):selected.append(j);left.remove(j);weight+=jj['size'];start=ss;pp=pr
  counts[k]+=1;b=counts[k];free[k]=start+pp;ids=[jobs[i]['job_id'] for i in selected];batches.append({'batch_id':b,'machine':k,'start_time':start,'processing_time':pp,'jobs':ids});obj+=free[k]*len(ids)
  for i in ids:assign[str(i)]={'batch':b,'machine':k,'completion_time':free[k]}
 if obj<best:
  best=obj;s={'objective_value':obj,'batches':batches,'job_assignments':assign};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)

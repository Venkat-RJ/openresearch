import time
_wall_start=time.time()
import argparse,json,random,itertools,math
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+a.time_limit-1
with open(a.instance_path) as f:d=json.load(f)
n=d['n_jobs'];m=d['n_machines'];pt=d['processing_times'];ni=d['no_idle_flags'];rng=random.Random(0);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if logger:logger.min_interval=0;logger.start_time=_wall_start
def completion(seq):
 row=[];v=0
 for j in seq:v+=pt[j][0];row.append(v)
 rows=[row]
 for i in range(1,m):
  if ni[i]:
   cum=0;shift=0
   for k,j in enumerate(seq):shift=max(shift,row[k]-cum);cum+=pt[j][i]
   nxt=[];v=shift
   for j in seq:v+=pt[j][i];nxt.append(v)
  else:
   nxt=[];v=0
   for k,j in enumerate(seq):v=max(v,row[k])+pt[j][i];nxt.append(v)
  rows.append(nxt);row=nxt
 return rows
def value(seq):return completion(seq)[-1][-1]
def pack(seq):
 rows=completion(seq);return {'objective_value':rows[-1][-1],'sequence':seq[:],'completion_times':{'machine_'+str(i):rows[i] for i in range(m)}}
order=sorted(range(n),key=lambda j:sum(pt[j][:m]),reverse=True);seq=[]
for j in order:
 seq=min((seq[:k]+[j]+seq[k:] for k in range(len(seq)+1)),key=value)
best=seq[:];bv=value(best)
if logger:logger.log_solution(bv,pack(best))
if n<=8:
 for seq in itertools.permutations(range(n)):
  if time.monotonic()>=end:break
  v=value(seq)
  if v<bv:
   best=list(seq);bv=v
   if logger:logger.log_solution(bv,pack(best))
else:
 current=best[:];cv=bv;temp=max(1,sum(sum(pt[j][:m]) for j in range(n))/(n*m)*.4)
 while time.monotonic()<end:
  removed=rng.sample(current,min(4,n));s=[j for j in current if j not in removed]
  for j in removed:
   s=min((s[:k]+[j]+s[k:] for k in range(len(s)+1)),key=value)
   if time.monotonic()>=end:break
  v=value(s)
  if len(s)!=n:break
  if v<bv:
   best=s[:];bv=v
   if logger:logger.log_solution(bv,pack(best))
  if v<cv or rng.random()<math.exp(min(0,(cv-v)/temp)):current=s;cv=v
sol=pack(best);sol['runtime']=time.monotonic()-start
if logger:logger.log_solution(sol['objective_value'],sol)
with open(a.solution_path,'w') as f:json.dump(sol,f)

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random,itertools
import numpy as np
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));n=d['n_jobs'];m=d['n_machines'];pt=np.asarray(d['processing_times']);flags=d['no_idle_flags'];deadline=time.monotonic()+max(.1,float(a.time_limit)-2)
 def evaluate(seq,full=False):
  z=len(seq);P=pt[seq].T;C=np.zeros((m,z));C[0]=np.cumsum(P[0])
  for i in range(1,m):
   if flags[i]:cum=np.cumsum(P[i]);start=max(C[i-1]-np.r_[0,cum[:-1]]);C[i]=start+cum
   else:
    for k in range(z):C[i,k]=max(C[i-1,k],C[i,k-1] if k else 0)+P[i,k]
  return C if full else C[-1,-1]
 seq=[]
 for j in sorted(range(n),key=lambda j:sum(pt[j]),reverse=True):
  choices=[seq[:k]+[j]+seq[k:] for k in range(len(seq)+1)];seq=min(choices,key=evaluate)
 best=seq[:];value=evaluate(best);rng=random.Random(42)
 if n<=9:
  for seq in itertools.permutations(range(n)):
   if time.monotonic()>deadline:break
   seq=list(seq);v=evaluate(seq)
   if v<value:best=seq;value=v
 else:
  while time.monotonic()<deadline:
   seq=best[:];remove=rng.sample(seq,min(4,n));seq=[v for v in seq if v not in remove]
   for j in remove:seq=min([seq[:k]+[j]+seq[k:] for k in range(len(seq)+1)],key=evaluate)
   v=evaluate(seq)
   if v<value:best=seq;value=v
 C=evaluate(best,True);json.dump({'objective_value':float(value),'sequence':best,'completion_times':{'machine_'+str(i):C[i].tolist() for i in range(m)}},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(float(value))
if __name__=='__main__':main()

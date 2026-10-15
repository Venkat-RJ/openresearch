import argparse,json,os,time,itertools,functools,math
wall_started=time.time()
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True)
a=p.parse_args(); end=time.monotonic()+a.time_limit-1; d=json.load(open(a.instance_path)); n=d['num_tasks']; m=d['num_stations']; t=d['processing_times']; F=d['forward_setup_times']; B=d['backward_setup_times']; pred=[0]*n; full=(1<<n)-1; logger=configured_logger(a.log_path,sense='minimize'); best=math.inf
for i,j in d['precedence_relations']: pred[j-1]|=1<<(i-1)
def stime(s): return sum(t[i] for i in s)+sum(F[i][j] for i,j in zip(s,s[1:]))+B[s[-1]][s[0]]
def emit(seqs):
 global best
 z=max(stime(s) for s in seqs)
 if z>=best: return
 best=z; sol={'objective_value':z,'cycle_time':z,'assignment':{str(i+1):k+1 for k,s in enumerate(seqs) for i in s},'station_sequences':{str(k+1):[i+1 for i in s] for k,s in enumerate(seqs)}}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(z,sol)
order=[]; used=0
while len(order)<n:
 i=next(i for i in range(n) if not (used>>i)&1 and not pred[i]&~used); order.append(i); used|=1<<i
for cuts in itertools.combinations(range(1,n),m-1):
 if time.monotonic()>=end: break
 cs=(0,)+cuts+(n,); emit([order[cs[k]:cs[k+1]] for k in range(m)])
class Expired(Exception): pass
def check():
 if time.monotonic()>=end: raise Expired
@functools.lru_cache(None)
def station(S):
 check(); ids=[i for i in range(n) if (S>>i)&1]; processing=sum(t[i] for i in ids)
 if processing>=best: return math.inf,()
 opt=math.inf; path=()
 for first in ids:
  if pred[first]&S: continue
  @functools.lru_cache(None)
  def extend(mask,last):
   if mask==S: return B[last][first],()
   val=math.inf; seq=()
   for j in ids:
    if (mask>>j)&1 or pred[j]&S&~mask: continue
    z,tail=extend(mask|(1<<j),j); z+=F[last][j]
    if z<val: val=z; seq=(j,)+tail
   return val,seq
  z,tail=extend(1<<first,first)
  if processing+z<opt: opt=processing+z; path=(first,)+tail
 return opt,path
@functools.lru_cache(None)
def partition(used,k):
 check(); remaining=full^used
 if k==1:
  v,s=station(remaining); return v,(s,)
 if remaining.bit_count()<k: return math.inf,()
 opt=math.inf; seqs=(); S=remaining
 while S:
  if remaining.bit_count()-S.bit_count()>=k-1:
   nxt=used|S
   if all(not (nxt>>i)&1 or not pred[i]&~nxt for i in range(n)):
    v,s=station(S)
    if v<min(best,opt):
     z,tail=partition(nxt,k-1); z=max(v,z)
     if z<opt:
      opt=z; seqs=(s,)+tail
      if used==0 and k==m and math.isfinite(z): emit(seqs)
  S=(S-1)&remaining
 return opt,seqs
try:
 z,seqs=partition(0,m)
 if math.isfinite(z): emit(seqs)
except Expired: pass

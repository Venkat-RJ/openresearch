import argparse,json,time,random,sys
wall_started=time.time()
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
sys.setrecursionlimit(10000)
def main():
 a=argparse.ArgumentParser(); a.add_argument('--instance_path'); a.add_argument('--solution_path'); a.add_argument('--time_limit',type=int,default=60); a.add_argument('--log_path'); a=a.parse_args(); start=time.monotonic(); end=start+max(.1,a.time_limit-1); d=json.load(open(a.instance_path)); n=d['n']; adj=[0]*n
 for u,v in d['edges']: adj[u]|=1<<v; adj[v]|=1<<u
 logger=configured_logger(a.log_path,sense='maximize') if a.log_path else None; best=[]; rng=random.Random(0)
 def accept(c):
  nonlocal best
  if len(c)>len(best):
   best=c[:]; s={'objective_value':len(c),'clique_vertices':sorted(c)}; json.dump(s,open(a.solution_path,'w')); logger and logger.log_solution(len(c),s)
 full=(1<<n)-1
 for it in range(min(n,100)):
  mask=full; c=[]
  while mask:
   vals=[]; b=mask
   while b: bit=b&-b; vals.append(bit.bit_length()-1); b-=bit
   vals.sort(key=lambda v:(adj[v]&mask).bit_count(),reverse=True); v=vals[0] if it==0 else rng.choice(vals[:min(3,len(vals))]); c.append(v); mask &= adj[v]
  accept(c)
  if time.monotonic()>start+min(2,a.time_limit*.1): break
 nodes=0
 def color(mask):
  order=[]; bounds=[]; bound=0
  while mask:
   avail=mask; bound+=1
   while avail:
    bit=avail&-avail; v=bit.bit_length()-1; mask ^= bit; avail ^= bit; avail &= ~adj[v]; order.append(v); bounds.append(bound)
  return order,bounds
 def expand(mask,c):
  nonlocal nodes
  nodes+=1
  if nodes%128==0 and time.monotonic()>=end: raise TimeoutError
  order,bounds=color(mask)
  for i in range(len(order)-1,-1,-1):
   if len(c)+bounds[i]<=len(best): return
   v=order[i]; bit=1<<v; newmask=mask&adj[v]; accept(c+[v])
   if newmask: expand(newmask,c+[v])
   mask ^= bit
 try: expand(full,[])
 except TimeoutError: pass
if __name__=='__main__': main()

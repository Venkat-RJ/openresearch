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
 a=argparse.ArgumentParser(); a.add_argument('--instance_path'); a.add_argument('--solution_path'); a.add_argument('--time_limit',type=int,default=60); a.add_argument('--log_path'); a=a.parse_args(); start=time.monotonic(); end=start+max(.1,a.time_limit-1); d=json.load(open(a.instance_path)); vs=d['graph']['vertices']; w=d['vertex_weights']; adj=[0]*len(vs)
 for u,v in d['graph']['edges']: adj[u]|=1<<v; adj[v]|=1<<u
 rng=random.Random(0); logger=configured_logger(a.log_path,sense='maximize') if a.log_path else None; best=-1; bestc=[]
 def accept(c,z):
  nonlocal best,bestc
  if z>best:
   best=z; bestc=c[:]; s={'objective_value':z,'clique_vertices':sorted(c)}; json.dump(s,open(a.solution_path,'w')); logger and logger.log_solution(z,s)
 full=sum(1<<v for v in vs); iteration=0
 while time.monotonic()<start+min(5,a.time_limit*.15):
  c=[]; mask=full
  while mask:
   vals=[]; b=mask
   while b: bit=b&-b; v=bit.bit_length()-1; vals.append(v); b-=bit
   vals.sort(key=lambda v:w[v]*(1+.03*(adj[v]&mask).bit_count()),reverse=True); v=vals[0] if iteration==0 else rng.choice(vals[:min(4,len(vals))]); c.append(v); mask &= adj[v]
  accept(c,sum(w[v] for v in c)); iteration+=1
 def color(mask):
  order=[]; bounds=[]; bound=0
  while mask:
   avail=mask; cls=[]; maxw=0
   while avail:
    bit=avail&-avail; v=bit.bit_length()-1; cls.append(v); maxw=max(maxw,w[v]); mask ^= bit; avail ^= bit; avail &= ~adj[v]
   bound+=maxw
   for v in cls: order.append(v); bounds.append(bound)
  return order,bounds
 nodes=0
 def expand(mask,c,z):
  nonlocal nodes
  nodes+=1
  if nodes%128==0 and time.monotonic()>=end: raise TimeoutError
  order,bounds=color(mask)
  for i in range(len(order)-1,-1,-1):
   if z+bounds[i]<=best: return
   v=order[i]; bit=1<<v
   newmask=mask&adj[v]; nz=z+w[v]
   accept(c+[v],nz)
   if newmask: expand(newmask,c+[v],nz)
   mask ^= bit
 try: expand(full,[],0)
 except TimeoutError: pass
if __name__=='__main__': main()

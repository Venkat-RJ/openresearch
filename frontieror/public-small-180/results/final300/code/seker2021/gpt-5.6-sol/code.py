import argparse,json,time,random
wall_started=time.time()
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger

def main():
 a=argparse.ArgumentParser(); a.add_argument('--instance_path'); a.add_argument('--solution_path'); a.add_argument('--time_limit',type=int,default=60); a.add_argument('--log_path'); a=a.parse_args(); start=time.monotonic(); end=start+max(.1,a.time_limit-1); d=json.load(open(a.instance_path)); clusters=d['partition']['clusters']; n=d['graph']['num_vertices']; adj=[set() for _ in range(n)]
 for u,v in d['graph']['edges']: adj[u].add(v); adj[v].add(u)
 rng=random.Random(0); logger=configured_logger(a.log_path,sense='minimize') if a.log_path else None; best=len(clusters)+1; bestsel=None; iteration=0
 while time.monotonic()<end:
  order=sorted(range(len(clusters)),key=lambda k: min(len(adj[v]) for v in clusters[k]),reverse=iteration%2==0)
  if iteration%3: rng.shuffle(order)
  sets=[]; selected={}; coloring={}
  for k in order:
   options=[]
   for v in clusters[k]:
    col=next((i for i,s in enumerate(sets) if not adj[v]&s),len(sets)); options.append((col,len(adj[v])+rng.random()*(iteration%5)*20,v))
   col,_,v=min(options); selected[str(k)]=v; coloring[str(v)]=col
   if col==len(sets): sets.append(set())
   sets[col].add(v)
  for sweep in range(4):
   for k in rng.sample(list(range(len(clusters))),len(clusters)):
    old=selected[str(k)]; oc=coloring[str(old)]; sets[oc].remove(old)
    options=[(i,len(adj[v]),v) for v in clusters[k] for i,s in enumerate(sets) if not adj[v]&s]
    col,_,v=min(options); selected[str(k)]=v; del coloring[str(old)]; coloring[str(v)]=col; sets[col].add(v)
   occupied=[i for i,s in enumerate(sets) if s]; remap={v:i for i,v in enumerate(occupied)}; sets=[sets[i] for i in occupied]; coloring={v:remap[c] for v,c in coloring.items()}
  z=len(sets)
  if z<best:
   best=z; s={'objective_value':z,'selected_vertices':selected,'coloring':coloring}; json.dump(s,open(a.solution_path,'w')); logger and logger.log_solution(z,s)
  iteration+=1
  if best==1: break
if __name__=='__main__': main()

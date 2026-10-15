import argparse,json,time,random,itertools,math
wall_started=time.time()
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger

def main():
 a=argparse.ArgumentParser(); a.add_argument('--instance_path'); a.add_argument('--solution_path'); a.add_argument('--time_limit',type=int,default=60); a.add_argument('--log_path'); a=a.parse_args(); start=time.monotonic(); end=start+max(.1,a.time_limit-1); d=json.load(open(a.instance_path)); vs=[x['id'] for x in d['nodes']]; idx={v:i for i,v in enumerate(vs)}; p=d['parameters']['p']; R=d['parameters']['R']; dist={}
 for x in d['arcs']: dist[x['from'],x['to']]=dist[x['to'],x['from']]=x['distance']
 req=[]; flows=[]
 for od in d['od_pairs']:
  path=od['shortest_path']; rt=path+list(reversed(path[:-1])); cum=[0.]
  for u,v in zip(rt,rt[1:]): cum.append(cum[-1]+dist[u,v])
  sets=set()
  for j in range(len(rt)-1):
   if cum[j+1]<=R+1e-9: continue
   sets.add(sum(1<<idx[v] for v in set(rt[k] for k in range(j+1) if cum[j+1]-cum[k]<=R+1e-9)))
  sets=[s for s in sets if not any(t!=s and t&s==t for t in sets)]; req.append(sets); flows.append(od['flow'])
 logger=configured_logger(a.log_path,sense='maximize') if a.log_path else None; best=-1; bestmask=0; rng=random.Random(0)
 def score(mask): return sum(f for sets,f in zip(req,flows) if all(s&mask for s in sets))
 def accept(mask):
  nonlocal best,bestmask
  z=score(mask)
  if z>best:
   best=z; bestmask=mask; s={'objective_value':z,'stations':[v for i,v in enumerate(vs) if mask>>i&1],'refueled_od_pairs':[i for i,sets in enumerate(req) if all(s&mask for s in sets)]}; json.dump(s,open(a.solution_path,'w')); logger and logger.log_solution(z,s)
 if math.comb(len(vs),p)<200000:
  for comb in itertools.combinations(range(len(vs)),p):
   accept(sum(1<<i for i in comb))
   if time.monotonic()>=end: break
  return
 mask=0
 for step in range(p):
  def potential(i):
   z=mask|1<<i; return sum(f*(sum(bool(s&z) for s in sets)/max(1,len(sets)))**3 for sets,f in zip(req,flows))
  i=max((i for i in range(len(vs)) if not mask>>i&1),key=potential); mask |= 1<<i
 accept(mask); stale=0
 while time.monotonic()<end:
  mask=bestmask
  if stale:
   chosen=[i for i in range(len(vs)) if mask>>i&1]; free=[i for i in range(len(vs)) if not mask>>i&1]
   for i,j in zip(rng.sample(chosen,min(3,p)),rng.sample(free,min(3,len(free)))): mask ^= (1<<i)|(1<<j)
  improve=True
  while improve and time.monotonic()<end:
   improve=False; current=score(mask); chosen=[i for i in range(len(vs)) if mask>>i&1]; free=[i for i in range(len(vs)) if not mask>>i&1]; rng.shuffle(chosen); rng.shuffle(free)
   for i in chosen:
    for j in free:
     new=mask^(1<<i)^(1<<j); z=score(new)
     if z>current+1e-8: mask=new; current=z; improve=True; break
    if improve or time.monotonic()>=end: break
   accept(mask)
  stale+=1
if __name__=='__main__': main()

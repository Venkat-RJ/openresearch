import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random,math
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));deadline=time.monotonic()+max(.1,float(a.time_limit)-2);items=d['items'];w={z['id']:z['weight'] for z in items};conf={v:set() for v in w};cap=d['bin_capacity']
 for u,v in d['conflict_edges']:conf[u].add(v);conf[v].add(u)
 def pack(order):
  bins=[];loads=[];sets=[]
  for v in order:
   valid=[i for i in range(len(bins)) if loads[i]+w[v]<=cap and not(conf[v]&sets[i])]
   if valid:i=max(valid,key=lambda i:loads[i]);bins[i].append(v);loads[i]+=w[v];sets[i].add(v)
   else:bins.append([v]);loads.append(w[v]);sets.append({v})
  return bins
 order=sorted(w,key=lambda v:(w[v],len(conf[v])),reverse=True);best=pack(order);rng=random.Random(42);lower=math.ceil(sum(w.values())/cap)
 while len(best)>lower and time.monotonic()<deadline:
  order=sorted(w,key=lambda v:w[v]*(.5+rng.random())+len(conf[v])*rng.random());b=pack(order)
  if len(b)<len(best):best=b
  if len(w)<45:
   target=len(best)-1;loads=[0]*target;sets=[set() for _ in range(target)];ordered=sorted(w,key=lambda v:(len(conf[v]),w[v]),reverse=True)
   def dfs(k):
    if time.monotonic()>deadline:return False
    if k==len(ordered):return True
    v=ordered[k];seen=set()
    for i in range(target):
     signature=(loads[i],tuple(sorted(sets[i])))
     if signature in seen:continue
     seen.add(signature)
     if loads[i]+w[v]<=cap and not(conf[v]&sets[i]):
      loads[i]+=w[v];sets[i].add(v)
      if dfs(k+1):return True
      loads[i]-=w[v];sets[i].remove(v)
     if not sets[i]:break
    return False
   if dfs(0):best=[list(s) for s in sets if s]
   else:break
 out={'objective_value':len(best),'bins':{str(i):{'items':s,'total_weight':sum(w[v] for v in s)} for i,s in enumerate(best)}};json.dump(out,open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(len(best))
if __name__=='__main__':main()

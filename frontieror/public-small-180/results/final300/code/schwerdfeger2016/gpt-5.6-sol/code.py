import argparse,json,math,time,random
wall_started=time.time()
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger

def main():
 a=argparse.ArgumentParser(); a.add_argument('--instance_path'); a.add_argument('--solution_path'); a.add_argument('--time_limit',type=int,default=60); a.add_argument('--log_path'); a=a.parse_args(); start=time.monotonic(); end=start+max(0.1,a.time_limit-1)
 d=json.load(open(a.instance_path)); p=d['processing_times']; m=d['num_machines']; n=len(p); mu=sum(p)/m; rng=random.Random(0); logger=configured_logger(a.log_path,sense='minimize') if a.log_path else None
 groups=[[] for _ in range(m)]; loads=[0]*m
 for j in sorted(range(n),key=lambda j:-p[j]):
  i=min(range(m),key=loads.__getitem__); groups[i].append(j); loads[i]+=p[j]
 def save():
  obj=math.sqrt(sum((v-mu)**2 for v in loads))/mu; s={'objective_value':obj,'assignment':{str(i):g[:] for i,g in enumerate(groups)},'machine_completion_times':{str(i):v for i,v in enumerate(loads)}}; json.dump(s,open(a.solution_path,'w')); logger and logger.log_solution(obj,s)
 save(); changed=True
 while changed and time.monotonic()<end:
  changed=False
  for hi in sorted(range(m),key=lambda i:-loads[i]):
   for lo in sorted(range(m),key=loads.__getitem__):
    delta=loads[hi]-loads[lo]
    if delta<=1: continue
    bestgain=0; move=None
    for j in groups[hi]:
     z=p[j]; gain=z*(delta-z)
     if gain>bestgain: bestgain=gain; move=(j,None)
    for j in groups[hi]:
     for k in groups[lo]:
      z=p[j]-p[k]; gain=z*(delta-z)
      if gain>bestgain: bestgain=gain; move=(j,k)
    if move:
     j,k=move; groups[hi].remove(j); groups[lo].append(j); z=p[j]
     if k is not None: groups[lo].remove(k); groups[hi].append(k); z-=p[k]
     loads[hi]-=z; loads[lo]+=z; changed=True; save()
    if time.monotonic()>=end: break
   if time.monotonic()>=end: break
 save()
if __name__=='__main__': main()

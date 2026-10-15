import time
_LOG_START = time.time()
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[_key]='1'
import argparse,json,time,random,numpy as np
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    dist=np.array(d['distance_matrix'],dtype=float);scores=np.array(d['scores'],dtype=float);n=len(scores);o=d['depot_vertex'];limit=d['distance_limitation_d0'];rng=random.Random(0);best=None
    def length(r):return sum(dist[r[i],r[(i+1)%len(r)]] for i in range(len(r)))
    def opt(r):
        changed=True
        while changed and time.monotonic()<deadline:
            changed=False
            for i in range(1,len(r)-1):
                for j in range(i+1,len(r)):
                    if dist[r[i-1],r[j]]+dist[r[i],r[(j+1)%len(r)]]<dist[r[i-1],r[i]]+dist[r[j],r[(j+1)%len(r)]]-1e-8:r[i:j+1]=reversed(r[i:j+1]);changed=True
        return r
    def emit(r):
        nonlocal best
        if len(r)<3:return
        obj=float(sum(scores[r]));ld=length(r)
        if ld<=limit+1e-8 and (best is None or obj>best['objective_value']):
            best={'objective_value':obj,'visited_vertices':r[:],'tour_edges':[[r[i],r[(i+1)%len(r)]] for i in range(len(r))]}
            if logger:logger.log_solution(obj,best)
    count=0
    while time.monotonic()<deadline:
        if best and count%3:
            r=best['visited_vertices'][:];remove=max(1,int(len(r)*rng.uniform(.08,.35)))
            for j in rng.sample(r[1:],min(remove,len(r)-1)):r.remove(j)
        else:r=[o]
        opt(r);cur=length(r);available=set(range(n))-set(r);power=1 if count==0 else rng.uniform(.65,1.7)
        while available and time.monotonic()<deadline:
            av=np.array(sorted(available));rr=np.array(r);nxt=np.roll(rr,-1)
            adds=dist[rr[:,None],av[None,:]]+dist[av[None,:],nxt[:,None]]-dist[rr,nxt][:,None];pos=adds.argmin(axis=0);delta=adds[pos,np.arange(len(av))];valid=np.flatnonzero(cur+delta<=limit+1e-8)
            if not len(valid):break
            rank=scores[av[valid]]/np.maximum(delta[valid],1e-6)**power
            pick=int(valid[int(rank.argmax())]);v=int(av[pick]);r.insert(int(pos[pick])+1,v);cur+=float(delta[pick]);available.remove(v)
        opt(r);emit(r);count+=1
        if n<30 and count>200:break
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

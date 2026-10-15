import time
_LOG_START = time.time()
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[_key]='1'
import argparse,json,time,random,numpy as np
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    act=d['activities'];ids=[a['id'] for a in act];idx={v:i for i,v in enumerate(ids)};n=len(ids);p=np.array([a['duration'] for a in act],int);req=np.array([a['resource_requirements'] for a in act],float);c=np.array(d['resource_costs']);H=d['max_project_duration'];arcs=[(idx[i],idx[j]) for i,j in d['precedence_relations']];pred=[[] for i in ids];succ=[[] for i in ids]
    for i,j in arcs:pred[j].append(i);succ[i].append(j)
    es=np.zeros(n,int);ls=np.full(n,H,int)
    for z in range(n):
        change=False
        for i,j in arcs:
            if es[j]<es[i]+p[i]:es[j]=es[i]+p[i];change=True
        if not change:break
    ls-=p
    for z in range(n):
        change=False
        for i,j in reversed(arcs):
            if ls[i]>ls[j]-p[i]:ls[i]=ls[j]-p[i];change=True
        if not change:break
    rng=random.Random(0);best=None;bests=None
    def usage(s):
        u=np.zeros((H,len(c)))
        for i in range(n):u[s[i]:s[i]+p[i]]+=req[i]
        return u
    def emit(s,u):
        nonlocal best,bests
        peak=u.max(axis=0);obj=float(peak@c)
        if best is None or obj<best['objective_value']:
            best={'objective_value':obj,'start_times':{str(ids[i]):int(s[i]) for i in range(n)},'resource_amounts':{str(k):float(peak[k]) for k in range(len(c))}};bests=s.copy()
            if logger:logger.log_solution(obj,best)
    count=0
    while time.monotonic()<deadline:
        s=es.copy() if count==0 else ls.copy() if count==1 else bests.copy();u=usage(s);emit(s,u);stalled=0
        for cycle in range(100):
            changed=False;order=list(range(n));rng.shuffle(order)
            for i in order:
                if time.monotonic()>=deadline:break
                if p[i]==0:continue
                lo=max(int(es[i]),max((int(s[j]+p[j]) for j in pred[i]),default=0));hi=min(int(ls[i]),min((int(s[j]-p[i]) for j in succ[i]),default=H-int(p[i])))
                if lo>=hi:continue
                old=int(s[i]);u[old:old+p[i]]-=req[i];basepeak=u.max(axis=0);baseenergy=float((u*u).sum());bval=None;bt=old
                times=list(range(lo,hi+1))
                if count>=2 and cycle==0:rng.shuffle(times)
                for t in times:
                    local=u[t:t+p[i]];peak=np.maximum(basepeak,(local+req[i]).max(axis=0));val=float(peak@c);energy=baseenergy+float((2*local*req[i]+req[i]*req[i]).sum());key=(val,energy)
                    if bval is None or key<bval:bval=key;bt=t
                u[bt:bt+p[i]]+=req[i];s[i]=bt;changed|=bt!=old
            emit(s,u)
            if not changed:break
            if time.monotonic()>=deadline:break
        count+=1
        if count>=2 and bests is not None:
            for z in range(10):
                i=rng.randrange(n);lo=max(es[i],max((bests[j]+p[j] for j in pred[i]),default=0));hi=min(ls[i],min((bests[j]-p[i] for j in succ[i]),default=H-p[i]));
                if hi>=lo:bests[i]=rng.randint(int(lo),int(hi))
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

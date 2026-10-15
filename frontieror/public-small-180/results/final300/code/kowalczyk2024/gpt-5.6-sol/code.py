import time
_LOG_START = time.time()
import argparse,json,random,math
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    jobs=d['jobs'];n=len(jobs);m=d['m'];T=d.get('T',math.ceil((sum(j['p'] for j in jobs)-max(j['p'] for j in jobs))/m)+max(j['p'] for j in jobs));rng=random.Random(0);best=None;bestmach=None
    def cost(seq):
        t=0;v=0
        for i in seq:t+=jobs[i]['p'];v+=jobs[i]['w']*max(0,t-jobs[i]['d'])
        return v if t<=T else float('inf')
    def emit(mach):
        nonlocal best,bestmach
        obj=sum(cost(seq) for seq in mach)
        if math.isfinite(obj) and (best is None or obj<best['objective_value']):
            schedule={}
            for seq in mach:
                t=0
                for i in seq:schedule[str(jobs[i]['id'])]=t;t+=jobs[i]['p']
            best={'objective_value':obj,'schedule':schedule};bestmach=[s[:] for s in mach]
            if logger:logger.log_solution(obj,best)
    count=0
    while time.monotonic()<deadline:
        if bestmach and count%3:
            mach=[s[:] for s in bestmach]
            for z in range(3):
                aa=rng.randrange(m);bb=rng.randrange(m)
                if mach[aa]:v=mach[aa].pop(rng.randrange(len(mach[aa])));mach[bb].insert(rng.randrange(len(mach[bb])+1),v)
        else:
            mach=[[] for _ in range(m)];load=[0]*m;left=set(range(n));K=2 if count==0 else rng.uniform(.5,5)
            while left:
                a=min(range(m),key=lambda z:load[z]);t=load[a];avg=sum(jobs[i]['p'] for i in left)/len(left)
                i=max(left,key=lambda z:jobs[z]['w']/jobs[z]['p']*math.exp(-max(0,jobs[z]['d']-jobs[z]['p']-t)/(K*avg)))
                if load[a]+jobs[i]['p']>T:break
                mach[a].append(i);load[a]+=jobs[i]['p'];left.remove(i)
            if left:count+=1;continue
        emit(mach);changed=True
        while changed and time.monotonic()<deadline:
            changed=False;base=[cost(s) for s in mach];move=None;gain=0
            for a in range(m):
                for pos,i in enumerate(mach[a]):
                    reduced=mach[a][:pos]+mach[a][pos+1:];rc=cost(reduced)
                    for b in range(m):
                        seq=reduced if b==a else mach[b]
                        for q in range(len(seq)+1):
                            new=seq[:q]+[i]+seq[q:];v=cost(new);g=base[a]-v if a==b else base[a]+base[b]-rc-v
                            if g>gain:gain=g;move=(a,pos,b,q)
                    if time.monotonic()>=deadline:break
                if time.monotonic()>=deadline:break
            if move:
                a,pos,b,q=move;i=mach[a].pop(pos);mach[b].insert(q,i);changed=True;emit(mach)
        count+=1
        if best and best['objective_value']==0:break
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

import time
_LOG_START = time.time()
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[_key]='1'
import argparse,json,time,random,numpy as np
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    jobs=sorted(d['jobs'],key=lambda j:(-j['w']/max(j['p'],1e-12),j['p']));m=d['m'];n=len(jobs);rng=random.Random(0);best=None
    def cost(seq):
        t=0;v=0
        for i in seq:t+=jobs[i]['p'];v+=jobs[i]['w']*t
        return v
    def emit(mach):
        nonlocal best
        obj=sum(cost(s) for s in mach)
        if best is None or obj<best['objective_value']:
            machines=[]
            for seq in mach:
                t=0;js=[]
                for i in seq:
                    j=jobs[i];t+=j['p'];js.append({'j_orig':j['j'],'p':j['p'],'w':j['w'],'C_j':t})
                machines.append({'schedule_cost':cost(seq),'jobs':js})
            best={'objective_value':obj,'machines':machines}
            if logger:logger.log_solution(obj,best)
    def redistribute(seq):
        total=sum(jobs[i]['p'] for i in seq);dp=np.full(total+1,np.inf);dp[0]=0;history=[];prefix=0
        for i in seq:
            p=jobs[i]['p'];w=jobs[i]['w'];t=np.arange(prefix+1);new=np.full(total+1,np.inf);choice=np.zeros(total+1,dtype=np.int8)
            a=dp[:prefix+1]+w*(t+p);new[p:prefix+p+1]=a;choice[p:prefix+p+1]=1
            b=dp[:prefix+1]+w*(prefix-t+p);mask=b<new[:prefix+1];new[:prefix+1][mask]=b[mask];choice[:prefix+1][mask]=0
            dp=new;history.append(choice);prefix+=p
        t=int(dp.argmin());a=[];b=[]
        for z in range(len(seq)-1,-1,-1):
            i=seq[z]
            if history[z][t]:a.append(i);t-=jobs[i]['p']
            else:b.append(i)
        return a[::-1],b[::-1]
    count=0
    while time.monotonic()<deadline:
        mach=[[] for _ in range(m)];load=[0]*m
        for i in range(n):
            order=sorted(range(m),key=lambda a:load[a]);a=order[0] if count==0 else order[rng.randrange(min(2,m))];mach[a].append(i);load[a]+=jobs[i]['p']
        emit(mach);changed=True
        while changed and time.monotonic()<deadline:
            changed=False;pairs=[(a,b) for a in range(m) for b in range(a+1,m)];rng.shuffle(pairs)
            for a,b in pairs:
                if time.monotonic()>=deadline:break
                seq=sorted(mach[a]+mach[b]);aa,bb=redistribute(seq)
                if cost(aa)+cost(bb)<cost(mach[a])+cost(mach[b]):mach[a]=aa;mach[b]=bb;changed=True;emit(mach)
        count+=1
        if n<=30 and count>=200:break
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

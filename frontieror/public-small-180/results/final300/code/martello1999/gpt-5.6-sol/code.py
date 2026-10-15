import time
_LOG_START = time.time()
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time,numpy as np
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    C=d['capacity'];items=d['items'];dp=np.full(C+1,-1,dtype=np.int64);dp[0]=0;choices=[];processed=[];best={'objective_value':0,'selected_items':[]}
    if logger:logger.log_solution(0,best)
    for item in items:
        if time.monotonic()>=deadline:break
        w=item['weight'];p=item['profit'];old=dp.copy();improved=old[:C+1-w]>=0;candidate=old[:C+1-w]+p;improved&=candidate>dp[w:];chosen=np.zeros(C+1,dtype=np.bool_);chosen[w:]=improved;dp[w:]=np.where(improved,candidate,dp[w:]);choices.append(chosen);processed.append(item);position=int(np.argmax(dp));value=int(dp[position])
        if value>best['objective_value']:
            selected=[]
            for previous,choice in reversed(list(zip(processed,choices))):
                if choice[position]:selected.append(previous['id']);position-=previous['weight']
            best={'objective_value':value,'selected_items':selected}
            if logger:logger.log_solution(value,best)
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

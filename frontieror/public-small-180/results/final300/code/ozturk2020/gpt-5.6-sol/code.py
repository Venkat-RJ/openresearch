import time
_LOG_START = time.time()
import argparse,json,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    jobs=d['jobs'];N=len(jobs);M=d['num_machines'];Cap=d['batch_capacity'];Q=max(j['release_date'] for j in jobs)+sum(j['processing_time'] for j in jobs);m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;x=m.addVars(N,N,M,vtype=GRB.BINARY);s=m.addVars(N,M,lb=0,ub=Q);p=m.addVars(N,M,lb=0,ub=Q);c=m.addVars(N,lb=0,ub=Q)
    for j,job in enumerate(jobs):
        m.addConstr(x.sum(j,'*','*')==1)
        for k in range(N):
            for machine in range(M):m.addConstr(p[k,machine]>=job['processing_time']*x[j,k,machine]);m.addConstr(s[k,machine]>=job['release_date']*x[j,k,machine]);m.addConstr(c[j]>=s[k,machine]+p[k,machine]-2*Q*(1-x[j,k,machine]))
    for k in range(N):
        for machine in range(M):
            m.addConstr(gp.quicksum(job['size']*x[j,k,machine] for j,job in enumerate(jobs))<=Cap)
            if k:m.addConstr(s[k,machine]>=s[k-1,machine]+p[k-1,machine])
    m.setObjective(c.sum());best=None
    def callback(model,where):
        nonlocal best
        if where!=GRB.Callback.MIPSOL:return
        values=model.cbGetSolution(x);start=model.cbGetSolution(s);batches=[];assignments={};total=0
        for machine in range(M):
            index=0;clock=0
            for k in range(N):
                selected=[jobs[j] for j in range(N) if values[j,k,machine]>.5]
                if not selected:continue
                index+=1;duration=max(j['processing_time'] for j in selected);begin=max(clock,max(j['release_date'] for j in selected),start[k,machine]);end=begin+duration;clock=end;ids=[j['job_id'] for j in selected];batches.append({'batch_id':index,'machine':machine+1,'start_time':begin,'processing_time':duration,'jobs':ids})
                for jid in ids:assignments[str(jid)]={'batch':index,'machine':machine+1,'completion_time':end};total+=end
        out={'objective_value':total,'batches':batches,'job_assignments':assignments}
        if best is None or total<best['objective_value']:
            best=out
            if logger:logger.log_solution(total,out)
    m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    if best is None:raise RuntimeError('No batch-scheduling incumbent')
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

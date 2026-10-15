import time
_LOG_START = time.time()
import argparse,json,os,itertools
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    jobs={j['id']:j for j in d['jobs']};N=list(jobs);machines=range(d['parameters']['m']);p={i:jobs[i]['processing_time'] for i in N};H=d.get('time_horizon',(sum(p.values())-max(p.values()))//len(machines)+max(p.values()));m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;x=m.addVars(N,machines,vtype=GRB.BINARY);s={i:m.addVar(vtype=GRB.INTEGER,ub=H-p[i]) for i in N};t=m.addVars(N,lb=0)
    for i in N:m.addConstr(x.sum(i,'*')==1);m.addConstr(t[i]>=s[i]+p[i]-jobs[i]['due_date'])
    for i,j in itertools.combinations(N,2):
        b=m.addVar(vtype=GRB.BINARY)
        for k in machines:m.addConstr(s[j]>=s[i]+p[i]-H*(3-b-x[i,k]-x[j,k]));m.addConstr(s[i]>=s[j]+p[j]-H*(2+b-x[i,k]-x[j,k]))
    m.setObjective(gp.quicksum(jobs[i]['weight']*t[i] for i in N));best=None
    def callback(model,where):
        nonlocal best
        if where!=GRB.Callback.MIPSOL:return
        xv=model.cbGetSolution(x);sv={i:round(model.cbGetSolution(s[i])) for i in N};rows=[]
        for k in machines:
            sequence=sorted((i for i in N if xv[i,k]>.5),key=lambda i:(sv[i],i));previous=0
            for i in sequence:
                completion=sv[i]+p[i];rows.append({'machine_id':k,'prev_job':previous,'job':i,'start_time':sv[i],'completion_time':completion,'weighted_tardiness':jobs[i]['weight']*max(0,completion-jobs[i]['due_date'])});previous=i
        value=sum(row['weighted_tardiness'] for row in rows);out={'objective_value':value,'schedule':rows}
        if best is None or value<best['objective_value']:
            best=out
            if logger:logger.log_solution(value,out)
    m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    if best is None:raise RuntimeError('No scheduling incumbent')
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

import time
_LOG_START = time.time()
import argparse,json,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    items={(g['group_id'],v['item_id']):v for g in d['groups'] for v in g['items']};m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;x=m.addVars(list(items),vtype=GRB.BINARY)
    for g in d['groups']:m.addConstr(gp.quicksum(x[g['group_id'],v['item_id']] for v in g['items'])==1)
    for k,cap in enumerate(d['capacities']):m.addConstr(gp.quicksum(v['weights'][k]*x[key] for key,v in items.items())<=cap)
    m.setObjective(gp.quicksum(v['profit']*x[key] for key,v in items.items()),GRB.MAXIMIZE);best=None
    def callback(model,where):
        nonlocal best
        if where==GRB.Callback.MIPSOL:
            values=model.cbGetSolution(x);selected=[{'group_id':key[0],**v} for key,v in items.items() if values[key]>.5];value=sum(v['profit'] for v in selected)
            if best is None or value>best['objective_value']:
                best={'objective_value':value,'selected_items':selected}
                if logger:logger.log_solution(value,best)
    m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    if best is None:raise RuntimeError('No MMKP incumbent')
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

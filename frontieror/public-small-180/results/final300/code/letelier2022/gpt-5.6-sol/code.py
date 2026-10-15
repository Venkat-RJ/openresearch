import time
_LOG_START = time.time()
import argparse,json,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    T=d['parameters']['T'];W=d['parameters']['W'];items={v['item_id']:v['weight'] for v in d['items'] if v['type']=='treatment'};source=d['source_item_id'];sink=d['sink_item_id'];arcs=d['arcs'];nodes=[v['item_id'] for v in d['items']];earliest={i:0 if i==source else T+1 if i==sink else 1 for i in nodes};latest={i:0 if i==source else T+1 if i==sink else T for i in nodes}
    for _ in nodes:
        for a in arcs:
            earliest[a['to']]=max(earliest[a['to']],earliest[a['from']]+a['lag']);latest[a['from']]=min(latest[a['from']],latest[a['to']]-a['lag'])
    B=min(len(items),d['parameters'].get('N',len(items)))
    if d['parameters']['L']!='infinity':B=min(B,int(d['parameters']['L']))
    keys=[(i,b,p) for i in items for b in range(B) for p in range(int(earliest[i]),int(latest[i])+1)];m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4
    x=m.addVars(keys,vtype=GRB.BINARY);u=m.addVars(range(B),range(1,T+1),vtype=GRB.BINARY);period={i:gp.quicksum(p*x[i,b,p] for ii,b,p in keys if ii==i) for i in items};period[source]=0;period[sink]=T+1
    for i in items:m.addConstr(gp.quicksum(x[k] for k in keys if k[0]==i)==1)
    for b in range(B):
        for p in range(1,T+1):
            m.addConstr(gp.quicksum(items[i]*x[i,b,p] for i in items if (i,b,p) in x)<=W*u[b,p])
            if b+1<B:m.addConstr(u[b,p]>=u[b+1,p])
    for a in arcs:m.addConstr(period[a['from']]+a['lag']<=period[a['to']])
    m.setObjective(u.sum());best=None
    def witness(values):
        assignments=[{'item_id':i,'bin':b,'period':p,'weight':items[i]} for (i,b,p),v in values.items() if v>.5];groups={}
        for a in assignments:groups.setdefault((a['bin'],a['period']),[]).append(a['item_id'])
        active=[{'bin':b,'period':p,'items':v,'total_weight':sum(items[i] for i in v)} for (b,p),v in sorted(groups.items())];return {'objective_value':len(active),'assignments':assignments,'active_bin_periods':active,'num_active_bin_periods':len(active)}
    def callback(model,where):
        nonlocal best
        if where==GRB.Callback.MIPSOL:
            s=witness(model.cbGetSolution(x))
            if best is None or s['objective_value']<best['objective_value']:
                best=s
                if logger:logger.log_solution(s['objective_value'],s)
    m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    if best is None:raise RuntimeError('No feasible bin-period incumbent')
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

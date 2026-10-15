import time
_LOG_START = time.time()
import argparse,json,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    L=[c['index'] for c in d['linehaul_customers']];B=[c['index'] for c in d['backhaul_customers']];N=L+B;depot=d['depot']['index'];Q=d['vehicle_capacity'];D=d['distance_matrix'];q={c['index']:c['demand'] for c in d['linehaul_customers']+d['backhaul_customers']};arcs=[(i,j) for i in [depot]+N for j in [depot]+N if i!=j and not(i in B and j in L) and not(i==depot and j in B)]
    m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;x=m.addVars(arcs,vtype=GRB.BINARY);u=m.addVars(N,lb=1,ub=len(N));load=m.addVars(N,lb=q,ub=Q)
    for i in N:m.addConstr(x.sum(i,'*')==1);m.addConstr(x.sum('*',i)==1)
    m.addConstr(x.sum(depot,'*')==d['num_vehicles']);m.addConstr(x.sum('*',depot)==d['num_vehicles'])
    for i,j in arcs:
        if i!=depot and j!=depot:
            m.addConstr(u[j]>=u[i]+1-len(N)*(1-x[i,j]))
            if (i in L)==(j in L):m.addConstr(load[j]>=load[i]+q[j]-Q*(1-x[i,j]))
    m.setObjective(gp.quicksum(D[i][j]*x[i,j] for i,j in arcs));best=None
    def emit(values):
        nonlocal best
        nxt={i:j for (i,j),v in values.items() if v>.5 and i!=depot};routes=[]
        for j in N:
            if values.get((depot,j),0)>.5:
                r=[depot,j]
                while r[-1]!=depot:r.append(nxt[r[-1]])
                routes.append(r)
        s={'routes':routes,'objective_value':sum(D[i][j] for r in routes for i,j in zip(r,r[1:]))}
        if best is None or s['objective_value']<best['objective_value']:
            best=s
            if logger:logger.log_solution(s['objective_value'],s)
    def callback(model,where):
        if where==GRB.Callback.MIPSOL:emit(model.cbGetSolution(x))
    m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    if best is None:raise RuntimeError('No VRPB incumbent')
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

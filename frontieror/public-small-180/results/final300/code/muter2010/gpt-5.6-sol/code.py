import time
_LOG_START = time.time()
import argparse,json,math,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    depot=d['depot']['id'];nodes={depot:d['depot'],**{c['id']:c for c in d['customers']}};customers=[c['id'] for c in d['customers']];Q=d['vehicle_capacity'];D={(i,j):math.hypot(nodes[i]['x']-nodes[j]['x'],nodes[i]['y']-nodes[j]['y']) for i in nodes for j in nodes};arcs=[(i,j) for i in nodes for j in nodes if i!=j and nodes[i]['ready_time']+nodes[i]['service_time']+D[i,j]<=nodes[j]['due_date']+1e-9];m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;x=m.addVars(arcs,vtype=GRB.BINARY);s={i:m.addVar(lb=nodes[i]['ready_time'],ub=nodes[i]['due_date']) for i in customers};u={i:m.addVar(lb=nodes[i]['demand'],ub=Q) for i in customers}
    for i in customers:m.addConstr(x.sum(i,'*')==1);m.addConstr(x.sum('*',i)==1)
    m.addConstr(x.sum(depot,'*')==x.sum('*',depot));m.addConstr(x.sum(depot,'*')<=d['num_vehicles'])
    for i,j in arcs:
        if i==depot:m.addConstr(s[j]>=nodes[depot]['ready_time']+D[i,j]-nodes[depot]['due_date']*(1-x[i,j]))
        elif j==depot:m.addConstr(s[i]+nodes[i]['service_time']+D[i,j]<=nodes[depot]['due_date']+(nodes[i]['due_date']+nodes[i]['service_time']+D[i,j])*(1-x[i,j]))
        else:
            M=max(0,nodes[i]['due_date']+nodes[i]['service_time']+D[i,j]-nodes[j]['ready_time']);m.addConstr(s[j]>=s[i]+nodes[i]['service_time']+D[i,j]-M*(1-x[i,j]));m.addConstr(u[j]>=u[i]+nodes[j]['demand']-Q*(1-x[i,j]))
    m.setObjective(gp.quicksum(D[i,j]*x[i,j] for i,j in arcs));best=None
    def callback(model,where):
        nonlocal best
        if where!=GRB.Callback.MIPSOL:return
        values=model.cbGetSolution(x);nxt={i:j for (i,j),v in values.items() if v>.5 and i!=depot};routes=[]
        for j in customers:
            if values.get((depot,j),0)>.5:
                path=[depot,j]
                while path[-1]!=depot:path.append(nxt[path[-1]])
                routes.append({'customer_ids':path[1:-1],'path':path})
        value=sum(D[i,j] for r in routes for i,j in zip(r['path'],r['path'][1:]));out={'objective_value':value,'routes':routes}
        if best is None or value<best['objective_value']:
            best=out
            if logger:logger.log_solution(value,out)
    m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    if best is None:raise RuntimeError('No VRPTW incumbent')
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

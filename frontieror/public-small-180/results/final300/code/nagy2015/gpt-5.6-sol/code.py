import time
_LOG_START = time.time()
import argparse,json,os,math
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    n=d['num_customers'];C=d['vehicle_capacity'];N=list(range(1,2*n+1));V=[0]+N;customers={c['id']:c for c in d['customers']};q={i:customers[i if i<=n else i-n]['delivery_demand' if i<=n else 'pickup_demand'] for i in N};orig=lambda i:i if i<=n else i-n;D={(i,j):d['distance_matrix'][orig(i)][orig(j)] for i in V for j in V};arcs=[(i,j) for i in V for j in V if i!=j and not(i>n and j==i-n)];m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;m.Params.LazyConstraints=1;x=m.addVars(arcs,vtype=GRB.BINARY);R=m.addVars(arcs,lb=0);P=m.addVars(arcs,lb=0)
    for j in N:
        m.addConstr(x.sum('*',j)==1);m.addConstr(x.sum(j,'*')==1);m.addConstr(R.sum('*',j)-R.sum(j,'*')==(q[j] if j<=n else 0));m.addConstr(P.sum(j,'*')-P.sum('*',j)==(q[j] if j>n else 0))
    m.addConstr(x.sum(0,'*')==x.sum('*',0));m.addConstr(P.sum(0,'*')==0);m.addConstr(R.sum('*',0)==0);m.addConstr(x.sum(0,'*')>=math.ceil(sum(q[i] for i in range(1,n+1))/C))
    for i,j in arcs:m.addConstr(R[i,j]+P[i,j]<=C*x[i,j])
    for i in N:
        for j in N:
            if i<j and (i,j) in x and (j,i) in x:m.addConstr(x[i,j]+x[j,i]<=1)
    m.setObjective(gp.quicksum(D[i,j]*x[i,j] for i,j in arcs));best=None
    def callback(model,where):
        nonlocal best
        if where!=GRB.Callback.MIPSOL:return
        values=model.cbGetSolution(x);nxt={i:j for (i,j),v in values.items() if v>.5 and i!=0};routes=[];seen=set()
        for j in N:
            if values.get((0,j),0)>.5:
                route=[0,j];seen.add(j)
                while route[-1]!=0:
                    following=nxt[route[-1]];route.append(following)
                    if following:seen.add(following)
                routes.append(route)
        unseen=set(N)-seen
        if unseen:
            while unseen:
                j=next(iter(unseen));cycle={j};k=nxt[j]
                while k not in cycle:cycle.add(k);k=nxt[k]
                model.cbLazy(gp.quicksum(x[i,j] for i in cycle for j in cycle if (i,j) in x)<=len(cycle)-1);unseen-=cycle
            return
        details=[]
        for route in routes:
            row=[]
            for node in route:
                if not node:row.append({'node':0,'type':'depot'})
                elif node<=n:row.append({'node':node,'type':'linehaul','original_customer':node,'delivery_demand':q[node]})
                else:row.append({'node':node,'type':'backhaul','original_customer':node-n,'pickup_demand':q[node]})
            details.append(row)
        value=sum(D[i,j] for route in routes for i,j in zip(route,route[1:]));out={'objective_value':value,'routes':routes,'routes_detailed':details}
        if best is None or value<best['objective_value']:
            best=out
            if logger:logger.log_solution(value,out)
    m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    if best is None:raise RuntimeError('No delivery/pickup incumbent')
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

import time
_LOG_START = time.time()
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[_key]='1'
import argparse,json,time,random,numpy as np,gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')

def solve(d,deadline,logger):
    dem=np.asarray(d['customers']['demands'],float);cap=np.asarray(d['plants']['capacities'],float);fixed=np.asarray(d['plants']['fixed_costs'],float);cost=np.asarray(d['shipping_costs'],float);n,m=cost.shape;alloc=np.zeros((n,m));remaining=cap.copy();opened=np.zeros(m,bool);rng=random.Random(0);best=None
    for i in sorted(range(n),key=lambda i:-dem[i]):
        left=1.
        while left>1e-9:
            room=remaining/ max(dem[i],1e-12);take=np.minimum(left,room);v=cost[i]+np.where(opened,0,fixed/np.maximum(take,1e-12));v[take<1e-9]=np.inf;j=int(v.argmin());q=float(take[j]);alloc[i,j]+=q;remaining[j]-=q*dem[i];opened[j]=True;left-=q
    def emit(candidate=None):
        nonlocal best
        matrix=alloc if candidate is None else candidate
        op=np.any(matrix>1e-9,axis=0);obj=float((cost*matrix).sum()+fixed[op].sum())
        if best is None or obj<best['objective_value']:
            best={'objective_value':obj,'open_plants':np.flatnonzero(op).tolist(),'assignments':{str(i):{str(j):float(matrix[i,j]) for j in np.flatnonzero(matrix[i]>1e-9)} for i in range(n)}}
            if logger:logger.log_solution(obj,best)
    emit();nearest=np.argsort(cost,axis=1)
    if n<=120:
        pairs={(i,int(j)) for i in range(n) for j in nearest[i,:18]};pairs.update(zip(*np.where(alloc>0)))
        md=gp.Model();md.Params.OutputFlag=0;md.Params.Threads=1;md.Params.Seed=0;md.Params.MIPGap=1e-4;md.Params.NumericFocus=0;md.Params.TimeLimit=max(.1,(deadline-time.monotonic())*.85)
        xx=md.addVars(sorted(pairs),lb=0,ub=1);yy=md.addVars(m,vtype=GRB.BINARY)
        for i in range(n):md.addConstr(gp.quicksum(xx[i,j] for j in range(m) if (i,j) in pairs)==1)
        for j in range(m):md.addConstr(gp.quicksum(dem[i]*xx[i,j] for i in range(n) if (i,j) in pairs)<=cap[j]*yy[j])
        md.setObjective(gp.quicksum(cost[i,j]*xx[i,j] for i,j in pairs)+gp.quicksum(fixed[j]*yy[j] for j in range(m)))
        for i,j in pairs:xx[i,j].Start=float(alloc[i,j])
        for j in range(m):yy[j].Start=float(np.any(alloc[:,j]>0))
        def global_callback(model,where):
            if where==GRB.Callback.MIPSOL:
                values=model.cbGetSolution(xx);candidate=np.zeros_like(alloc)
                for i,j in pairs:candidate[i,j]=max(0,values[i,j])
                emit(candidate)
        md.optimize(global_callback)
        if md.SolCount:
            alloc[:]=0
            for i,j in pairs:alloc[i,j]=max(0,xx[i,j].X)
            emit()
    it=0
    while time.monotonic()<deadline-.1:
        if it%2==0:
            centers=np.flatnonzero(np.any(alloc>1e-9,axis=0));center=int(centers[rng.randrange(len(centers))]);customers=np.flatnonzero(alloc[:,center]>1e-9).tolist()
            neighbors=np.argsort(cost[:,center])
            for i in neighbors:
                if len(customers)>=35:break
                if int(i) not in customers:customers.append(int(i))
        else:customers=rng.sample(range(n),min(n,30))
        customers=customers[:45];I=set(customers);fac=set(np.flatnonzero(np.any(alloc[customers]>1e-9,axis=0)).tolist())
        for i in customers:fac.update(int(j) for j in nearest[i,:3])
        if len(fac)>40:it+=1;continue
        J=sorted(fac);external=dem@alloc-dem[customers]@alloc[customers];active=external>1e-7;room=cap-external
        md=gp.Model();md.Params.OutputFlag=0;md.Params.Threads=1;md.Params.Seed=0;md.Params.MIPGap=1e-4;md.Params.NumericFocus=0;md.Params.TimeLimit=max(.02,min(1.5,deadline-time.monotonic()))
        xx=md.addVars(customers,J,lb=0,ub=1);yy=md.addVars(J,vtype=GRB.BINARY)
        for j in J:
            if active[j]:yy[j].LB=1
            md.addConstr(gp.quicksum(dem[i]*xx[i,j] for i in customers)<=room[j]*yy[j])
        for i in customers:md.addConstr(gp.quicksum(xx[i,j] for j in J)==1)
        md.setObjective(gp.quicksum(cost[i,j]*xx[i,j] for i in customers for j in J)+gp.quicksum((0 if active[j] else fixed[j])*yy[j] for j in J))
        old=float((cost[customers]*alloc[customers]).sum()+fixed[(~active)&np.any(alloc[customers]>1e-9,axis=0)].sum())
        for i in customers:
            for j in J:xx[i,j].Start=float(alloc[i,j])
        for j in J:yy[j].Start=float(active[j] or np.any(alloc[customers,j]>0))
        frozen=alloc.copy()
        def neighborhood_callback(model,where):
            if where==GRB.Callback.MIPSOL:
                values=model.cbGetSolution(xx);candidate=frozen.copy();candidate[customers]=0
                for i in customers:
                    for j in J:candidate[i,j]=max(0,values[i,j])
                emit(candidate)
        md.optimize(neighborhood_callback)
        if md.SolCount and md.ObjVal<old-1e-5:
            alloc[customers]=0
            for i in customers:
                for j in J:alloc[i,j]=max(0,xx[i,j].X)
            emit()
        md.dispose();it+=1
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

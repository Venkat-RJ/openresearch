import time
_LOG_START = time.time()
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[_key]='1'
import argparse,json,time,numpy as np,gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')

def solve(d,deadline,logger):
    N=d['N'];k=d['k'];R=np.asarray(d['scenarios'],float);mu=np.asarray(d['mu'],float);prob=np.asarray(d['p_s'] if isinstance(d['p_s'],list) else [d['p_s']]*len(R));tail=1-d['beta'];gamma=d['gamma'];best=None
    def value(x):
        loss=-R@x;order=np.argsort(loss)[::-1];mass=tail;w=np.zeros(len(R));a=float(loss[order[0]])
        for s in order:
            take=min(mass,prob[s]);w[s]=take/tail;mass-=take;a=float(loss[s])
            if mass<=1e-12:break
        v=float(prob@np.maximum(0,loss-a)/tail);obj=float(x@x/(2*gamma)+a+v);gradient=x/gamma-R.T@w
        return obj,gradient,a,v
    def emit(x):
        nonlocal best
        val,g,a,v=value(x)
        if mu@x<d['mu_bar']-1e-7:return val,g
        if best is None or val<best['objective_value']:
            best={'objective_value':val,'x':x.tolist(),'z':(x>1e-9).astype(int).tolist(),'a':a,'v':v}
            if logger:logger.log_solution(val,best)
        return val,g
    m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4
    x=m.addVars(N,lb=0,ub=1);z=m.addVars(N,vtype=GRB.BINARY);eta=m.addVar(lb=-GRB.INFINITY)
    m.addConstr(gp.quicksum(x.values())==1);m.addConstr(gp.quicksum(mu[i]*x[i] for i in range(N))>=d['mu_bar']);m.addConstr(gp.quicksum(z.values())<=k)
    for i in range(N):m.addConstr(x[i]<=z[i])
    quadratic=N<=80
    m.setObjective(eta+gp.quicksum(x[i]*x[i]/(2*gamma) for i in range(N)) if quadratic else eta)
    point=np.zeros(N);point[int(mu.argmax())]=1;val,g=emit(point)
    def cut(point,val,g):
        if quadratic:val-=float(point@point/(2*gamma));g=g-point/gamma
        m.addConstr(eta>=val+gp.quicksum(float(g[i])*(x[i]-float(point[i])) for i in range(N)))
    cut(point,val,g);iters=0
    def callback(model,where):
        if where==GRB.Callback.MIPSOL:
            values=model.cbGetSolution(x)
            emit(np.array([values[i] for i in range(N)]))
    while time.monotonic()<deadline and iters<1500:
        m.Params.TimeLimit=max(.02,min(3,deadline-time.monotonic()));m.optimize(callback)
        if m.SolCount==0:break
        point=np.array([x[i].X for i in range(N)]);val,g=emit(point);cut(point,val,g);iters+=1
        if best and best['objective_value']-m.ObjBound<=1e-9:break
        if m.NumConstrs>1900:break
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

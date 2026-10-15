import time
_LOG_START = time.time()
import argparse,json,os,math
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    n1=d['n1'];n2=d['n2'];integer=set(d['integer_follower_vars']);ub=2*max(d['b1']+d['b2']);best=None
    def model():
        m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;return m
    def follower_variables(m):return [m.addVar(lb=0,ub=ub,vtype=GRB.INTEGER if j+1 in integer else GRB.CONTINUOUS) for j in range(n2)]
    master=model();x=master.addVars(range(n1),vtype=GRB.INTEGER,lb=0,ub=ub);y=follower_variables(master);fobj=gp.quicksum(d['d2'][j]*y[j] for j in range(n2));leader=gp.quicksum(d['c'][j]*x[j] for j in range(n1))
    for A,B,b in ((d['A1'],d['B1'],d['b1']),(d['A2'],d['B2'],d['b2'])):
        for a,row,rhs in zip(A,B,b):master.addConstr(gp.quicksum(a[j]*x[j] for j in range(n1))+gp.quicksum(row[j]*y[j] for j in range(n2))<=rhs)
    master.setObjective(leader,GRB.MAXIMIZE);cuts=0
    while time.monotonic()<deadline and cuts*d['m2']<1700:
        master.Params.TimeLimit=min(4,max(.01,deadline-time.monotonic()));master.optimize()
        if not master.SolCount:break
        xv=[round(x[j].X) for j in range(n1)];master_obj=master.ObjVal;fm=model();fy=follower_variables(fm);fo=gp.quicksum(d['d2'][j]*fy[j] for j in range(n2))
        for a,row,rhs in zip(d['A2'],d['B2'],d['b2']):fm.addConstr(gp.quicksum(row[j]*fy[j] for j in range(n2))<=rhs-sum(a[j]*xv[j] for j in range(n1)))
        fm.setObjective(fo,GRB.MAXIMIZE);fm.Params.TimeLimit=min(3,max(.01,deadline-time.monotonic()));fm.optimize()
        if fm.Status!=GRB.OPTIMAL:break
        value=fm.ObjVal;yhat=[v.X for v in fy];fm.addConstr(fo>=value-1e-7)
        for a,row,rhs in zip(d['A1'],d['B1'],d['b1']):fm.addConstr(gp.quicksum(row[j]*fy[j] for j in range(n2))<=rhs-sum(a[j]*xv[j] for j in range(n1)))
        fm.Params.TimeLimit=min(2,max(.01,deadline-time.monotonic()));fm.optimize()
        if fm.SolCount:
            s={'objective_value':sum(d['c'][j]*xv[j] for j in range(n1)),'x':xv,'y':[v.X for v in fy],'follower_objective_value':sum(d['d2'][j]*fy[j].X for j in range(n2))}
            if best is None or s['objective_value']>best['objective_value']+1e-7:
                best=s
                if logger:logger.log_solution(s['objective_value'],s)
            if abs(s['objective_value']-master_obj)<1e-7 and master.Status==GRB.OPTIMAL:break
        blocked=master.addVars(range(d['m2']),vtype=GRB.BINARY);allblocked=gp.quicksum(blocked[j] for j in range(d['m2']));M=sum(abs(c)*ub for c in d['d2'])+abs(value)+1
        for j,(a,row,rhs) in enumerate(zip(d['A2'],d['B2'],d['b2'])):
            threshold=math.floor(rhs-sum(row[k]*yhat[k] for k in range(n2)))+1
            master.addGenConstrIndicator(blocked[j],True,gp.quicksum(a[k]*x[k] for k in range(n1))>=threshold)
        master.addConstr(fobj>=value-M*allblocked);cuts+=1
    if best is None:raise RuntimeError('No bilevel feasible incumbent')
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

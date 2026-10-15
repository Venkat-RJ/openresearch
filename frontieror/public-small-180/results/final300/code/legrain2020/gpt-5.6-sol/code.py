import time
_LOG_START = time.time()
import argparse,json,random,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    nurses=d['nurses'];N=len(nurses);D=d['num_days'];shifts=[s['name'] for s in d['shifts']];pw=d['penalty_weights'];forbidden=set(map(tuple,d['forbidden_shift_successions']));schedule=[[None]*D for _ in nurses];rng=random.Random(0);best=None
    def model():
        m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;return m
    def cost_row(i,row):
        work=[v is not None for v in row];c=nurses[i]['contract'];value=0
        def streak(a,lo,hi,weight):
            total=sum(max(0,sum(a[k:k+hi+1])-hi) for k in range(D-hi))
            for k in range(D):
                if a[k] and (k==0 or not a[k-1]):total+=sum(1-a[k+j] for j in range(1,lo) if k+j<D)
            return total*weight
        value+=streak(work,c['CD_minus'],c['CD_plus'],pw['c_S2a']);value+=streak([not w for w in work],c['CR_minus'],c['CR_plus'],pw['c_S3'])
        for s in d['shifts']:value+=streak([v is not None and v[0]==s['name'] for v in row],s['CS_minus'],s['CS_plus'],pw['c_S2b'])
        value+=pw['c_S4']*sum(row[p['day']] is not None and row[p['day']][0]==p['shift'] for p in nurses[i].get('preferences',[]));value+=pw['c_S6']*(max(0,c['L_minus']-sum(work))+max(0,sum(work)-c['L_plus']));weekends=[(k,k+1) for k in range(5,D-1,7)];value+=pw['c_S5']*sum(work[a]!=work[b] for a,b in weekends);value+=pw['c_S7']*max(0,sum(work[a] or work[b] for a,b in weekends)-c['WE_plus']);return value
    def emit():
        nonlocal best
        coverage={}
        for row in schedule:
            for day,v in enumerate(row):
                if v:coverage[day,v[0],v[1]]=coverage.get((day,v[0],v[1]),0)+1
        value=sum(cost_row(i,row) for i,row in enumerate(schedule))+pw['c_S1']*sum(max(0,v['optimal']-coverage.get((day,s,k),0)) for day,dem in enumerate(d['demand']) for s,skills in dem.items() for k,v in skills.items());s={'objective_value':value,'schedule':[{'nurse_id':nurse['id'],'nurse_name':nurse['name'],'assignments':[{'day':day,'shift':v[0],'skill':v[1]} for day,v in enumerate(schedule[i]) if v]} for i,nurse in enumerate(nurses)]}
        if best is None or value<best['objective_value']:
            best=s
            if logger:logger.log_solution(value,s)
    for day in range(D):
        m=model();keys=[(i,s,k) for i,nurse in enumerate(nurses) for s in shifts for k in nurse['skills'] if day==0 or schedule[i][day-1] is None or (schedule[i][day-1][0],s) not in forbidden];y=m.addVars(keys,vtype=GRB.BINARY);obj=gp.LinExpr()
        for i in range(N):m.addConstr(gp.quicksum(v for (ii,s,k),v in y.items() if ii==i)<=1)
        for s,skills in d['demand'][day].items():
            for k,v in skills.items():
                cov=gp.quicksum(y[key] for key in keys if key[1:]==(s,k));m.addConstr(cov>=v['minimum']);z=m.addVar(lb=0);m.addConstr(z>=v['optimal']-cov);obj+=pw['c_S1']*z
        for (i,s,k),var in y.items():
            used=sum(v is not None for v in schedule[i]);target=nurses[i]['contract']['L_plus']*(day+1)/D;obj+=(max(0,used-target)*8+sum(pw['c_S4'] for p in nurses[i].get('preferences',[]) if p['day']==day and p['shift']==s)+i*.001)*var
        m.setObjective(obj);m.Params.TimeLimit=min(.8,max(.01,deadline-time.monotonic()));m.optimize()
        if not m.SolCount:raise RuntimeError('Daily minimum cover infeasible')
        for (i,s,k),var in y.items():
            if var.X>.5:schedule[i][day]=(s,k)
    emit();iteration=0;block_limit=3
    while time.monotonic()<deadline:
        block=rng.sample(range(N),min(block_limit,N));m=model();keys=[(i,day,s,k) for i in block for day in range(D) for s in shifts for k in nurses[i]['skills']];y=m.addVars(keys,vtype=GRB.BINARY);obj=gp.LinExpr();work={};shift={}
        def positive(expr,weight):
            nonlocal obj
            z=m.addVar(lb=0);m.addConstr(z>=expr);obj+=weight*z
        for i in block:
            for day in range(D):
                work[i,day]=gp.quicksum(y[i,day,s,k] for s in shifts for k in nurses[i]['skills']);m.addConstr(work[i,day]<=1)
                for s in shifts:shift[i,day,s]=gp.quicksum(y[i,day,s,k] for k in nurses[i]['skills'])
            for day in range(D-1):
                for a,b in forbidden:m.addConstr(shift[i,day,a]+shift[i,day+1,b]<=1)
            c=nurses[i]['contract'];positive(c['L_minus']-gp.quicksum(work[i,day] for day in range(D)),pw['c_S6']);positive(gp.quicksum(work[i,day] for day in range(D))-c['L_plus'],pw['c_S6']);weekvars=[]
            for sat in range(5,D-1,7):
                sun=sat+1;w=m.addVar(lb=0,ub=1);m.addConstr(w>=work[i,sat]);m.addConstr(w>=work[i,sun]);m.addConstr(w<=work[i,sat]+work[i,sun]);weekvars.append(w);positive(work[i,sat]-work[i,sun],pw['c_S5']);positive(work[i,sun]-work[i,sat],pw['c_S5'])
            positive(gp.quicksum(weekvars)-c['WE_plus'],pw['c_S7'])
            def streak(expressions,lo,hi,weight):
                for day in range(D-hi):positive(gp.quicksum(expressions[day:day+hi+1])-hi,weight)
                if lo>1:
                    for day in range(D):
                        start=m.addVar(lb=0,ub=1);m.addConstr(start>=expressions[day]-(expressions[day-1] if day else 0));m.addConstr(start<=expressions[day])
                        if day:m.addConstr(start<=1-expressions[day-1])
                        for j in range(1,lo):
                            if day+j<D:positive(start-expressions[day+j],weight)
            streak([work[i,day] for day in range(D)],c['CD_minus'],c['CD_plus'],pw['c_S2a']);streak([1-work[i,day] for day in range(D)],c['CR_minus'],c['CR_plus'],pw['c_S3'])
            for s in d['shifts']:streak([shift[i,day,s['name']] for day in range(D)],s['CS_minus'],s['CS_plus'],pw['c_S2b'])
            for p in nurses[i].get('preferences',[]):obj+=pw['c_S4']*shift[i,p['day'],p['shift']]
        for day,dem in enumerate(d['demand']):
            for s,skills in dem.items():
                for k,v in skills.items():
                    outside=sum(schedule[i][day]==(s,k) for i in range(N) if i not in block);cov=outside+gp.quicksum(y[key] for key in keys if key[1:]==(day,s,k));m.addConstr(cov>=v['minimum']);positive(v['optimal']-cov,pw['c_S1'])
        for (i,day,s,k),var in y.items():var.Start=int(schedule[i][day]==(s,k))
        m.setObjective(obj);m.update()
        if m.NumVars>1900 or m.NumConstrs>1900:
            block_limit=2;m.dispose();continue
        frozen={i:schedule[i][:] for i in block}
        def callback(model,where):
            if where==GRB.Callback.MIPSOL:
                values=model.cbGetSolution(y)
                for i in block:schedule[i]=[None]*D
                for (i,day,s,k),value in values.items():
                    if value>.5:schedule[i][day]=(s,k)
                emit()
                for i in block:schedule[i]=frozen[i][:]
        m.Params.TimeLimit=min(4,max(.01,deadline-time.monotonic()));m.optimize(callback)
        if m.SolCount:
            candidate={i:[None]*D for i in block}
            for (i,day,s,k),var in y.items():
                if var.X>.5:candidate[i][day]=(s,k)
            for i,row in candidate.items():schedule[i]=row
            emit()
        iteration+=1
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

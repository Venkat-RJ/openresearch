import time
_LOG_START = time.time()
import argparse,json,os,math
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    P=d['num_patients_P'];L=d['provider_service_hours_L_minutes'];costkey=next(iter(d['cost_structures']));cost=d['cost_structures'][costkey];cw=cost['c_w'];cg=cost['c_g'];co=cost['c_o'];patients=d['patients'];R=range(1,P+1);params={}
    names={'aL':'lower_bound_duration_adequate_prep','aU':'upper_bound_duration_adequate_prep','iL':'lower_bound_duration_inadequate_prep','iU':'upper_bound_duration_inadequate_prep','uL':'lower_bound_arrival_time_deviation','uU':'upper_bound_arrival_time_deviation','muA':'mean_duration_adequate_prep','muI':'mean_duration_inadequate_prep','muU':'mean_arrival_time_deviation','muQ':'mean_prep_adequacy'}
    for k,name in names.items():params[k]={p:patients[p-1][name] for p in R};params[k][P+1]=0
    wp={i:cw if i<=P else co for i in range(1,P+2)};gp_cost={i:cg if i<=P else 0 for i in range(1,P+3)};pi={(i,j):-gp_cost[j]+sum(wp[k] for k in range(i,j)) for i in range(1,P+3) for j in range(i,P+3)};M=max(abs(v) for v in pi.values())+L;m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;x=m.addVars(R,R,vtype=GRB.BINARY,name='x');t=m.addVars(R,lb=0,ub=L,name='t');beta=m.addVars(range(1,P+3),lb=-GRB.INFINITY,name='beta');beta[P+2].LB=0
    family={};family['rho']=m.addVars(R,lb=min(params['aL'][p] for p in R),ub=max(params['aU'][p] for p in R),name='rho');family['alpha']=m.addVars(R,lb=min(params['iL'][p] for p in R),ub=max(params['iU'][p] for p in R),name='alpha');family['lambda']=m.addVars(R,lb=min(params['uL'][p] for p in R),ub=max(params['uU'][p] for p in R),name='lambda');family['gamma']=m.addVars(R,lb=-M,ub=M,name='gamma')
    for name,ub in (('z',max(params['aU'][p] for p in R)),('v',max(params['iU'][p] for p in R)),('r',max(params['uU'][p] for p in R)),('e',M)):family[name]=m.addVars(R,lb=0,ub=ub,name=name)
    for i in R:
        m.addConstr(x.sum('*',i)==1);m.addConstr(x.sum(i,'*')==1)
        for a,b in (('z','rho'),('v','alpha'),('r','lambda')):m.addConstr(family[a][i]>=family[b][i])
        m.addConstr(family['e'][i]>=-family['gamma'][i])
        if i>1:m.addConstr(t[i]>=t[i-1])
    m.addConstr(t[1]==0)
    mc={}
    for name,base,low,high in (('eta','rho','aL','aU'),('tau','alpha','iL','iU'),('nu','lambda','uL','uU'),('Gam','gamma',None,None),('zeta','z','zero','aU'),('psi','v','zero','iU'),('phi','r','zero','uU')):
        mc[name]=m.addVars(R,R,lb=0 if low=='zero' else -GRB.INFINITY,name=name)
        for p in R:
            lo=0 if low=='zero' else -M if low is None else params[low][p];hi=M if high is None else params[high][p]
            for i in R:
                v=mc[name][p,i];z=family[base][i];m.addConstr(v<=hi*x[p,i]);m.addConstr(v>=lo*x[p,i]);m.addConstr(v<=z-lo*(1-x[p,i]));m.addConstr(v>=z-hi*(1-x[p,i]))
    obj=gp.quicksum(params['muA'][p]*mc['eta'][p,i]+params['muI'][p]*mc['tau'][p,i]+params['muU'][p]*mc['nu'][p,i]+params['muQ'][p]*mc['Gam'][p,i] for p in R for i in R)+beta.sum();m.setObjective(obj)
    def at(name,i):return gp.quicksum(params[name][p]*x[p,i] for p in R) if i<=P else 0
    def bound_at(i,minimum):return gp.quicksum((min(params['aL'][p],params['iL'][p]) if minimum else max(params['aU'][p],params['iU'][p]))*x[p,i] for p in R) if i<=P else 0
    def K(i):return -gp.quicksum(params['aU'][p]*mc['eta'][p,i]+(params['aL'][p]-params['aU'][p])*mc['zeta'][p,i]+params['iU'][p]*mc['tau'][p,i]+(params['iL'][p]-params['iU'][p])*mc['psi'][p,i]+params['uU'][p]*mc['nu'][p,i]+(params['uL'][p]-params['uU'][p])*mc['phi'][p,i] for p in R)+family['e'][i]
    for j in range(1,P+3):
        rhs=(-t[1]-at('uL',1))*pi[1,j]
        for i in range(2,min(j,P)+1):rhs+=(-t[i]-at('uL',i)+t[i-1]+at('uL',i-1)+bound_at(i-1,True))*pi[i,j]
        for i in range(2,min(j,P+1)+1):rhs+=(bound_at(i-1,False)-bound_at(i-1,True))*max(pi[i,j],0)
        rhs+=gp.quicksum(K(i) for i in range(1,min(j,P)+1))
        if j>=P+1:rhs+=(t[P]+at('uL',P)-L+bound_at(P,True))*pi[P+1,j]
        m.addConstr(gp.quicksum(beta[i] for i in range(1,j+1))>=rhs)
    for k in range(2,P+2):
        for j in range(k,P+3):
            rhs=(at('uU',k-1)-at('uL',k-1))*max(pi[k,j]+gp_cost[k-1],0);start=min(k,P+1)
            for i in range(start,min(j,P)+1):rhs+=(-t[i]-at('uL',i)+t[i-1]+at('uL',i-1)+bound_at(i-1,True))*pi[i,j]+K(i)
            for i in range(start,min(j,P+1)+1):rhs+=(bound_at(i-1,False)-bound_at(i-1,True))*max(pi[i,j],0)
            if j>=P+1:rhs+=(t[P]+at('uL',P)-L+bound_at(P,True))*pi[P+1,j]
            m.addConstr(gp.quicksum(beta[i] for i in range(k,j+1))>=rhs)
    m.update();variables=m.getVars();coefficients={v:v.Obj for v in variables};constant=m.ObjCon;best=None
    def emit(get):
        nonlocal best
        assignment={str(p):max(R,key=lambda i:get(x[p,i])) for p in R};schedule={str(i):get(t[i]) for i in R};value=constant+math.fsum(coefficients[v]*get(v) for v in variables);result={'objective_value':value,'model_variables':{v.VarName:get(v) for v in variables},'cost_structure':costkey,**cost,'assignment':assignment,'schedule':schedule,'patient_start_times':{p:schedule[str(i)] for p,i in assignment.items()}}
        if best is None or value<best['objective_value']:
            best=result
            if logger:logger.log_solution(value,result)
    def callback(model,where):
        if where==GRB.Callback.MIPSOL:
            values=dict(zip(variables,model.cbGetSolution(variables)));emit(values.__getitem__)
    m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    if not m.SolCount:raise RuntimeError('No DRO incumbent')
    emit(lambda v:v.X)
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

import time
_LOG_START = time.time()
import argparse,json,os,math,itertools
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    p=d['global_parameters'];ip=d['instance_parameters'];attack=d['interdiction_parameters'];n1=ip['num_level1_facilities'];n2=ip['num_level2_facilities'];F=n1+n2;I=ip['num_customers'];K=ip['k_max'];B=ip['total_interdiction_budget'];costs=[{0:0,**{int(k):v for k,v in attack['interdiction_cost_level'+str(t)+'_h'+str(t)].items()}} for t in (1,2)];reductions=[{0:0,**{int(k):v for k,v in attack['capacity_reduction_level'+str(t)+'_d'+str(t)].items()}} for t in (1,2)];c1=d['level1_facilities']['capacity_type1']+d['level2_facilities']['capacity_type1'];c2=d['level2_facilities']['capacity_type2'];m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;u1=m.addVars(range(I),range(F),lb=0);u2=m.addVars(range(I),range(n2),lb=0);u3=m.addVars(range(F),range(n2),lb=0);o1=m.addVars(range(I),lb=0);o2=m.addVars(range(I),lb=0);o3=m.addVars(range(F),lb=0);dem=d['customers']['demands'];obj=gp.LinExpr();sigma=p['sigma'];theta=p['theta']
    for i in range(I):
        m.addConstr(u1.sum(i,'*')+o1[i]==theta*dem[i]);m.addConstr(u2.sum(i,'*')+o2[i]==(1-theta)*dem[i]);obj+=(p['alpha_prime']+sigma*p['gamma_prime'])*o1[i]+p['beta_prime']*o2[i]
        for j in range(F):obj+=(p['alpha']*d['distances']['customer_to_level1_facility'][i][j] if j<n1 else p['beta']*d['distances']['customer_to_level2_facility'][i][j-n1])*u1[i,j]
        for j in range(n2):obj+=p['beta']*d['distances']['customer_to_level2_facility'][i][j]*u2[i,j]
    cap1=[];cap2=[]
    for j in range(F):
        m.addConstr(sigma*u1.sum('*',j)==u3.sum(j,'*')+o3[j]);cap1.append(m.addConstr(u1.sum('*',j)<=c1[j]));obj+=p['gamma_prime']*o3[j]
        for f in range(n2):
            if j<n1:distance=d['distances']['level1_to_level2_facility'][j][f]
            else:distance=math.hypot(d['level2_facilities']['x_coordinates'][j-n1]-d['level2_facilities']['x_coordinates'][f],d['level2_facilities']['y_coordinates'][j-n1]-d['level2_facilities']['y_coordinates'][f])
            obj+=p['gamma']*distance*u3[j,f]
    for f in range(n2):cap2.append(m.addConstr(u2.sum('*',f)+u3.sum('*',f)<=c2[f]))
    m.setObjective(obj);best=None
    def patterns(position,spent,pattern):
        if time.monotonic()>=deadline:return
        if position==F:yield pattern;return
        for k in range(K+1):
            value=costs[position>=n1][k]
            if spent+value<=B+1e-8:yield from patterns(position+1,spent+value,pattern+[k])
    for pattern in patterns(0,0,[]):
        for j,k in enumerate(pattern):cap1[j].RHS=c1[j]*(1-reductions[j>=n1][k])
        for f in range(n2):cap2[f].RHS=c2[f]*(1-reductions[1][pattern[n1+f]])
        m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize()
        if m.Status!=GRB.OPTIMAL:continue
        value=m.ObjVal
        if best is None or value>best['objective_value']+1e-6:
            best={'objective_value':value,'interdiction_pattern':{('S1_'+str(j) if j<n1 else 'S2_'+str(j-n1)):k for j,k in enumerate(pattern)}}
            if logger:logger.log_solution(value,best)
    if best is None:raise RuntimeError('No defender optimum obtained')
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

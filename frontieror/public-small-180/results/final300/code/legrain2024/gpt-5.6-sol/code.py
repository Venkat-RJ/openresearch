import time
_LOG_START = time.time()
import argparse,json,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    D=d['instance_metadata']['num_days'];nurses=d['nurses'];ids=[n['id'] for n in nurses];shifts=d['shifts'];allshifts=shifts+['None'];skills=d['skills'];hard=d['constraints']['hard_constraints'];soft=d['constraints']['soft_constraints'];m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;x=m.addVars(ids,range(D),allshifts,vtype=GRB.BINARY);keys=[(n['id'],day,s,k) for n in nurses for day in range(D) for s in shifts for k in skills if not hard.get('skill_requirement') or k in n['skills']];y=m.addVars(keys,vtype=GRB.BINARY);obj=gp.LinExpr();work={(i,day):1-x[i,day,'None'] for i in ids for day in range(D)}
    def positive(expr,weight):
        nonlocal obj
        if not weight:return
        z=m.addVar(lb=0);m.addConstr(z>=expr);obj+=weight*z
    def week_days(w):return sorted(v-1 for k,v in w.items() if k in ('friday','saturday','sunday') and 1<=v<=D)
    for i in ids:
        for day in range(D):
            m.addConstr(gp.quicksum(x[i,day,s] for s in allshifts)==1)
            for s in shifts:m.addConstr(x[i,day,s]==gp.quicksum(y[key] for key in keys if key[:3]==(i,day,s)))
    for day in range(D):
        for s,dem in d['demand'].get('day_'+str(day+1),{}).items():
            for k,required in dem.items():
                cov=gp.quicksum(y[key] for key in keys if key[1:]==(day,s,k))
                if hard.get('demand_equality'):m.addConstr(cov==required)
                else:positive(required-cov,d['unsatisfied_demand_penalty'])
    for i in ids:
        for fp in hard.get('forbidden_patterns',[]):
            pat=[v[1] for v in fp['pattern']]
            for day in range(D-len(pat)+1):m.addConstr(gp.quicksum(x[i,day+j,s] for j,s in enumerate(pat))<=len(pat)-1)
        tw=gp.quicksum(work[i,day] for day in range(D));c=soft.get('total_working_days',{});positive(c.get('min',0)-tw,c.get('penalty_under',0));positive(tw-c.get('max',D),c.get('penalty_over',0))
        hw=hard.get('total_working_days',{})
        if isinstance(hw,dict):m.addConstr(tw>=hw.get('min',0));m.addConstr(tw<=hw.get('max',D))
        weekend_vars=[]
        for weekend in d['weekends']:
            days=week_days(weekend);w=m.addVar(lb=0,ub=1)
            for day in days:m.addConstr(w>=work[i,day])
            m.addConstr(w<=gp.quicksum(work[i,day] for day in days));weekend_vars.append(w)
            iw=soft.get('identical_weekend',{})
            if iw.get('enabled'):
                z=m.addVar(lb=0,ub=1)
                for day in days[1:]:
                    for s in allshifts:m.addConstr(z>=x[i,days[0],s]-x[i,day,s]);m.addConstr(z>=x[i,day,s]-x[i,days[0],s])
                obj+=iw.get('penalty',0)*z
        c=soft.get('total_worked_weekends',{});positive(gp.quicksum(weekend_vars)-c.get('max',len(weekend_vars)),c.get('penalty',0))
        def streak(a,c,short=True):
            hi=c.get('max',len(a));lo=c.get('min',1)
            for day in range(len(a)-hi):positive(gp.quicksum(a[day:day+hi+1])-hi,c.get('penalty_over',c.get('penalty',0)))
            if short:
                for length in range(1,lo):
                    for day in range(1,len(a)-length):positive(gp.quicksum(a[day:day+length])-a[day-1]-a[day+length]-length+1,c.get('penalty_under',0))
        streak([work[i,day] for day in range(D)],soft.get('consecutive_working_days',{}));streak([1-work[i,day] for day in range(D)],soft.get('consecutive_days_off',{}))
        for s in shifts:streak([x[i,day,s] for day in range(D)],soft.get('consecutive_same_shift',{}),False)
        streak(weekend_vars,soft.get('consecutive_worked_weekends',{}),False)
        for fp in soft.get('forbidden_patterns',[]):
            pat=[v[1] for v in fp['pattern']]
            for day in range(D-len(pat)+1):positive(gp.quicksum(x[i,day+j,s] for j,s in enumerate(pat))-len(pat)+1,fp.get('penalty',0))
    for n in nurses:
        i=n['id']
        for pref in n.get('preferences',[]):
            day=pref['day']-1;typ=pref['type'];s=pref.get('shift','None');pen=pref.get('penalty',0)
            if typ=='day_off':obj+=pen*work[i,day]
            elif typ=='day_on':obj+=pen*x[i,day,'None']
            elif typ=='shift_off':obj+=pen*x[i,day,s]
            elif typ=='shift_on':obj+=pen*(1-x[i,day,s])
        if not hard.get('skill_requirement'):
            for key in keys:
                if key[0]==i and key[3] not in n['skills']:obj+=soft.get('skill_requirement_penalty',0)*y[key]
    best=None
    def emit(xv,yv):
        nonlocal best
        schedule={i:{'day_'+str(day+1):max(allshifts,key=lambda s:xv[i,day,s]) for day in range(D)} for i in ids};witness={i:{'day_'+str(day+1):'None' for day in range(D)} for i in ids}
        for (i,day,s,k),value in yv.items():
            if value>.5:witness[i]['day_'+str(day+1)]=k
        value=0
        def streak_cost(flags,c,short=True):
            hi=c.get('max',len(flags));lo=c.get('min',1);total=c.get('penalty_over',c.get('penalty',0))*sum(max(0,sum(flags[day:day+hi+1])-hi) for day in range(len(flags)-hi))
            if short:
                for length in range(1,lo):
                    for day in range(1,len(flags)-length):
                        if all(flags[day:day+length]) and not flags[day-1] and not flags[day+length]:total+=c.get('penalty_under',0)
            return total
        for n in nurses:
            i=n['id'];row=[schedule[i]['day_'+str(day+1)] for day in range(D)];work_flags=[s!='None' for s in row];total=sum(work_flags);c=soft.get('total_working_days',{});value+=c.get('penalty_under',0)*max(0,c.get('min',0)-total)+c.get('penalty_over',0)*max(0,total-c.get('max',D));week_flags=[]
            for weekend in d['weekends']:
                days=week_days(weekend);week_flags.append(any(work_flags[day] for day in days));iw=soft.get('identical_weekend',{})
                if iw.get('enabled') and len({row[day] for day in days})>1:value+=iw.get('penalty',0)
            c=soft.get('total_worked_weekends',{});value+=c.get('penalty',0)*max(0,sum(week_flags)-c.get('max',len(week_flags)));value+=streak_cost(work_flags,soft.get('consecutive_working_days',{}))+streak_cost([not w for w in work_flags],soft.get('consecutive_days_off',{}))
            for s in shifts:value+=streak_cost([v==s for v in row],soft.get('consecutive_same_shift',{}),False)
            value+=streak_cost(week_flags,soft.get('consecutive_worked_weekends',{}),False)
            for fp in soft.get('forbidden_patterns',[]):
                pat=[v[1] for v in fp['pattern']];value+=fp.get('penalty',0)*sum(row[day:day+len(pat)]==pat for day in range(D-len(pat)+1))
            for pref in n.get('preferences',[]):
                actual=row[pref['day']-1];typ=pref['type'];s=pref.get('shift','None');violated=(typ=='day_off' and actual!='None') or (typ=='day_on' and actual=='None') or (typ=='shift_off' and actual==s) or (typ=='shift_on' and actual!=s);value+=pref.get('penalty',0)*violated
            if not hard.get('skill_requirement'):value+=soft.get('skill_requirement_penalty',0)*sum(work_flags[day] and witness[i]['day_'+str(day+1)] not in n['skills'] for day in range(D))
        for day in range(D):
            daykey='day_'+str(day+1)
            for s,dem in d['demand'].get(daykey,{}).items():
                for k,required in dem.items():value+=d['unsatisfied_demand_penalty']*max(0,required-sum(schedule[i][daykey]==s and witness[i][daykey]==k for i in ids))
        result={'objective_value':value,'schedule':schedule,'skill_assignments':witness}
        if best is None or value<best['objective_value']:
            best=result
            if logger:logger.log_solution(value,result)
    def callback(model,where):
        if where==GRB.Callback.MIPSOL:emit(model.cbGetSolution(x),model.cbGetSolution(y))
    m.setObjective(obj);m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    if not m.SolCount:raise RuntimeError('No nurse roster incumbent')
    emit({key:var.X for key,var in x.items()},{key:var.X for key,var in y.items()})
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

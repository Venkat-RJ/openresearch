import time
_LOG_START = time.time()
import argparse,json,itertools,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve_serial(d,deadline,logger):
    S=d['S'];p=d['p'];seaside=sorted(d['seaside_containers'],key=lambda x:x['id']);landside=sorted(d['landside_containers'],key=lambda x:x['deadline']);best=None
    for insert in range(len(seaside)+1):
        if time.monotonic()>=deadline:break
        w=[d['sigma_w']];l=[d['sigma_l']];ev=[];le=[];finish=0
        def tick(ww,ll):
            assert 0<=ww<ll<=S+1 and abs(ww-w[-1])<=1 and abs(ll-l[-1])<=1
            w.append(ww);l.append(ll)
        def move_w(goal):
            while w[-1]!=goal:
                ww=w[-1]+(1 if goal>w[-1] else -1);ll=max(l[-1],ww+1);tick(ww,ll)
        def move_l(goal):
            while l[-1]!=goal:
                ll=l[-1]+(1 if goal>l[-1] else -1);ww=min(w[-1],ll-1);tick(ww,ll)
        def hold(q):
            for z in range(q):tick(w[-1],l[-1])
        def land_jobs():
            for j in landside:
                move_l(j['source_slot']);tl=len(w)-1;hold(p);move_l(S+1);td=max(len(w)-1,j['earliest_finish_time']-p)
                hold(td-(len(w)-1));hold(p);le.append({'container_id':j['id'],'source_slot':j['source_slot'],'lift_time':tl,'drop_time':td})
                if td+p>j['deadline']:return False
            return True
        ok=True
        for k in range(len(seaside)+1):
            if k==insert:
                if not land_jobs():ok=False;break
            if k==len(seaside):break
            j=seaside[k];target=j['target_slot'];move_w(0);hold(p);q=min(target,S);move_w(q);td=len(w)-1;hold(p);e={'container_id':j['id'],'target_slot':target,'seaside_drop_time':td,'seaside_drop_slot':q}
            if target>S:
                move_w(max(0,q-1));move_l(q);tl=len(w)-1;hold(p);move_l(target);ld=len(w)-1;hold(p);e.update({'landside_lift_time':tl,'landside_lift_slot':q,'landside_drop_time':ld});finish=ld+p
            else:finish=td+p
            ev.append(e)
        if not ok:continue
        sol={'objective_value':float(finish),'makespan':float(finish),'crane_w_positions':{str(t):v for t,v in enumerate(w)},'crane_l_positions':{str(t):v for t,v in enumerate(l)},'seaside_events':ev,'landside_events':le}
        if best is None or finish<best['objective_value']:
            best=sol
            if logger:logger.log_solution(float(finish),sol)
    return best

def solve(d,deadline,logger):
    best=solve_serial(d,deadline,logger);S=d['S'];p=d['p'];seaside=sorted(d['seaside_containers'],key=lambda v:v['id']);landside=d['landside_containers'];n=len(seaside)
    if best is None or n>8:return best
    H=max(int(k) for k in best['crane_w_positions']);times=list(range(H-p+1));targets={v['id']:v['target_slot'] for v in seaside};ids=list(targets);mandatory={i for i in ids if targets[i]>S};patterns=[]
    for slot in range(1,S+1):
        useful={i for i in ids if targets[i]>slot+1};patterns.append((slot,useful|mandatory))
    for bits in range(1<<n):
        selected={ids[k] for k in range(n) if bits>>k&1}|mandatory
        for slot in range(1,S+1):patterns.append((slot,selected))
    seen=set()
    for slot,selected in patterns:
        signature=(slot,tuple(sorted(selected)))
        if signature in seen:continue
        seen.add(signature)
        if time.monotonic()>=deadline-.1:break
        m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;w=m.addVars(range(H+1),lb=0,ub=S);l=m.addVars(range(H+1),lb=1,ub=S+1);C=m.addVar(lb=0,ub=best['objective_value']);m.addConstr(w[0]==d['sigma_w']);m.addConstr(l[0]==d['sigma_l']);operations={};starts={};crane_ops={'w':[],'l':[]}
        def operation(key,crane,position,earliest=0,latest=H-p):
            ts=[t for t in times if earliest<=t<=latest];variables=m.addVars(ts,vtype=GRB.BINARY);m.addConstr(variables.sum()==1);operations[key]=(variables,position);starts[key]=gp.quicksum(t*variables[t] for t in ts);crane_ops[crane].append(key)
        for i in ids:
            drop_slot=slot if i in selected else targets[i];operation(('wl',i),'w',0);operation(('wd',i),'w',drop_slot);m.addConstr(starts['wd',i]>=starts['wl',i]+p+drop_slot)
            if i in selected:
                operation(('sl',i),'l',slot);operation(('sd',i),'l',targets[i]);m.addConstr(starts['sl',i]>=starts['wd',i]+p);m.addConstr(starts['sd',i]>=starts['sl',i]+p+abs(targets[i]-slot));m.addConstr(C>=starts['sd',i]+p)
        for i,j in zip(ids,ids[1:]):m.addConstr(starts['wl',j]>=starts['wd',i]+p+operations['wd',i][1])
        m.addConstr(C>=starts['wd',ids[-1]]+p)
        for job in landside:
            i=job['id'];operation(('ll',i),'l',job['source_slot']);operation(('ld',i),'l',S+1,max(0,job['earliest_finish_time']-p),min(H-p,job['deadline']-p));m.addConstr(starts['ld',i]>=starts['ll',i]+p+S+1-job['source_slot'])
        for t in range(H+1):
            m.addConstr(w[t]+1<=l[t])
            if t:
                for trajectory in [w,l]:m.addConstr(trajectory[t]-trajectory[t-1]<=1);m.addConstr(trajectory[t-1]-trajectory[t]<=1)
        for crane,trajectory in [('w',w),('l',l)]:
            keys=crane_ops[crane]
            for t in times:
                active=gp.quicksum(operations[key][0][t] for key in keys if t in operations[key][0]);position=gp.quicksum(operations[key][1]*operations[key][0][t] for key in keys if t in operations[key][0]);m.addConstr(active<=1)
                for tp in range(t,t+p+1):m.addConstr(trajectory[tp]>=position);m.addConstr(trajectory[tp]<=position+(S+1)*(1-active))
            for t in range(H+1):
                busy=gp.quicksum(variables[t0] for key in keys for variables,pos in [operations[key]] for t0 in range(max(0,t-p+1),min(t,H-p)+1) if t0 in variables);m.addConstr(busy<=1)
        liftkeys=[key for key in crane_ops['l'] if key[0] in ('sl','ll')];dropkeys=[key for key in crane_ops['l'] if key[0] in ('sd','ld')]
        for t in range(H+1):
            inventory=gp.quicksum(var[tt] for key in liftkeys for var,pos in [operations[key]] for tt in var if tt<=t)-gp.quicksum(var[tt] for key in dropkeys for var,pos in [operations[key]] for tt in var if tt<=t);m.addConstr(inventory>=0);m.addConstr(inventory<=1)
        m.setObjective(C);m.update()
        if m.NumVars>1900 or m.NumConstrs>1900:m.dispose();continue
        def callback(model,where):
            nonlocal best
            if where!=GRB.Callback.MIPSOL:return
            chosen={key:max(var,key=lambda t:model.cbGetSolution(var[t])) for key,(var,pos) in operations.items()};events=[]
            for i in ids:
                event={'container_id':i,'target_slot':targets[i],'seaside_lift_time':chosen['wl',i],'seaside_drop_time':chosen['wd',i],'seaside_drop_slot':operations['wd',i][1]}
                if i in selected:event.update({'landside_lift_time':chosen['sl',i],'landside_lift_slot':slot,'landside_drop_time':chosen['sd',i]})
                events.append(event)
            land=[{'container_id':j['id'],'source_slot':j['source_slot'],'lift_time':chosen['ll',j['id']],'drop_time':chosen['ld',j['id']]} for j in landside];value=max([chosen['wd',ids[-1]]+p]+[chosen['sd',i]+p for i in selected]);out={'objective_value':value,'makespan':value,'crane_w_positions':{str(t):model.cbGetSolution(w[t]) for t in range(H+1)},'crane_l_positions':{str(t):model.cbGetSolution(l[t]) for t in range(H+1)},'seaside_events':events,'landside_events':land}
            if value<best['objective_value']:
                best=out
                if logger:logger.log_solution(value,out)
        m.Params.TimeLimit=max(.01,min(3,deadline-time.monotonic()));m.optimize(callback);m.dispose()
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

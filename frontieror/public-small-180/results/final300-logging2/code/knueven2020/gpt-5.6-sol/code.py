import time
_LOG_START = time.time()
import argparse,json,math,os,random,sys
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    gens=d['generators'];G=len(gens);T=len(d['demand']);reserve=d.get('reserve_requirement',[0]*T);best=None;rng=random.Random(0)
    def transitions(commit):
        starts=[];stops=[]
        for gi,g in enumerate(gens):
            previous=[int(g['initial_status']>0)]+commit[gi][:-1];starts.append([max(0,b-a) for a,b in zip(previous,commit[gi])]);stops.append([max(0,a-b) for a,b in zip(previous,commit[gi])])
        return starts,stops
    def dispatch_failure(block,reason,status=None,sol_count=0):
        print(f'knueven dispatch block={block} status={status} SolCount={sol_count} reason={reason} remaining={max(0,deadline-time.monotonic()):.3f}',file=sys.stderr,flush=True)
        return None
    def dispatch(commit):
        starts,stops=transitions(commit);upper=[[g['p_max']*commit[gi][t] for t in range(T)] for gi,g in enumerate(gens)]
        for gi,g in enumerate(gens):
            for t in range(T):
                if sum(starts[gi][max(0,t-g['min_up_time']+1):t+1])>commit[gi][t] or sum(stops[gi][max(0,t-g['min_down_time']+1):t+1])>1-commit[gi][t]:return dispatch_failure(t,f'min_up_down:{g["id"]}')
            initial=g['initial_status'];forced=max(0,g['min_up_time']-initial) if initial>0 else max(0,g['min_down_time']+initial)
            if any(commit[gi][t]!=int(initial>0) for t in range(min(T,forced))):return dispatch_failure(0,f'initial_obligation:{g["id"]}')
            RU=g['ramp_up_limit'];RD=g['ramp_down_limit'];SU=g['startup_ramp_limit'];SD=g['shutdown_ramp_limit'];trup=max(0,math.ceil((g['p_max']-SU)/RU)) if RU>0 else 0;trdown=max(0,math.ceil((g['p_max']-SD)/RD)) if RD>0 else 0
            if g['min_up_time']>=trup+trdown+2:
                for t in range(T):upper[gi][t]-=sum(max(0,g['p_max']-SU-i*RU)*starts[gi][t-i] for i in range(trup+1) if t-i>=0)+sum(max(0,g['p_max']-SD-i*RD)*stops[gi][t+1+i] for i in range(trdown+1) if t+1+i<T)
        for t in range(T):
            if sum(g['p_min']*commit[gi][t] for gi,g in enumerate(gens))>d['demand'][t]+1e-7 or sum(g['p_max']*commit[gi][t] for gi,g in enumerate(gens))<d['demand'][t]+reserve[t]-1e-7:return dispatch_failure(t,'demand_reserve_envelope')
        power=[[] for _ in gens];previous=[g['initial_power'] if g['initial_status']>0 else 0 for g in gens]
        for t0 in range(0,T,6):
            if time.monotonic()>=deadline-.1:return dispatch_failure(t0,'deadline')
            H=min(12,T-t0);m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;p={};objective=gp.LinExpr()
            for gi,g in enumerate(gens):
                for k in range(H):
                    t=t0+k;low=g['p_min']*commit[gi][t]
                    if upper[gi][t]<low-1e-7:
                        m.dispose()
                        return dispatch_failure(t0,f'power_bounds:{g["id"]}:period={t}')
                    p[gi,k]=m.addVar(lb=low,ub=max(low,upper[gi][t]));previous_u=commit[gi][t-1] if t else int(g['initial_status']>0);previous_p=p[gi,k-1] if k else previous[gi];m.addConstr(p[gi,k]-previous_p<=g['ramp_up_limit']*previous_u+g['startup_ramp_limit']*starts[gi][t]);m.addConstr(previous_p-p[gi,k]<=g['ramp_down_limit']*commit[gi][t]+g['shutdown_ramp_limit']*stops[gi][t]);segments=g['production_cost_segments'];slope=sum((s['output_mw_end']-s['output_mw_start'])*s['marginal_cost_per_mwh'] for s in segments)/max(1e-9,g['p_max']-g['p_min']);objective+=slope*p[gi,k]
            for k in range(H):m.addConstr(gp.quicksum(p[gi,k] for gi in range(G))==d['demand'][t0+k])
            m.setObjective(objective)
            while time.monotonic()<deadline-.1:
                m.Params.TimeLimit=max(.01,(deadline-time.monotonic()-.1)/(math.ceil((T-t0)/6)+1));m.optimize()
                if m.SolCount:break
                dispatch_failure(t0,'no_incumbent',m.Status,m.SolCount)
                if m.Status in (GRB.INFEASIBLE,GRB.INF_OR_UNBD,GRB.UNBOUNDED):
                    m.dispose()
                    return None
            if not m.SolCount:
                status=m.Status;m.dispose()
                return dispatch_failure(t0,'deadline_without_incumbent',status)
            keep=min(6,T-t0)
            for gi in range(G):power[gi].extend(max(0,p[gi,k].X) for k in range(keep));previous[gi]=power[gi][-1]
            m.dispose()
        return power
    def emit(commit,power):
        nonlocal best
        value=0;starts,stops=transitions(commit)
        for gi,g in enumerate(gens):
            for t in range(T):
                value+=g['no_load_cost']*commit[gi][t];remaining=max(0,power[gi][t]-g['p_min']*commit[gi][t])
                for segment in g['production_cost_segments']:amount=min(remaining,segment['output_mw_end']-segment['output_mw_start']);value+=amount*segment['marginal_cost_per_mwh'];remaining-=amount
                if starts[gi][t]:
                    eligible=[g['startup_costs'][-1]['cost']];previous_hi=0
                    for tier in g['startup_costs'][:-1]:
                        hi=tier['max_downtime_hours'];lo=previous_hi+1;previous_hi=hi
                        if any(lo<=t-s<=hi for s in range(t) if stops[gi][s]) or (g['initial_status']<0 and lo<=t+abs(g['initial_status'])<=hi):eligible.append(tier['cost'])
                    value+=min(eligible)
        out={'objective_value':value,'schedule':{g['id']:{'u':commit[gi][:],'p':power[gi][:],'pb':[g['p_max']*u for u in commit[gi]]} for gi,g in enumerate(gens)}}
        if best is None or value<best['objective_value']:
            best=out
            if logger:logger.log_solution(value,out)
    master=gp.Model();master.Params.OutputFlag=0;master.Params.Seed=0;master.Params.Threads=1;master.Params.NumericFocus=0;master.Params.MIPGap=1e-4;on=master.addVars(G,vtype=GRB.BINARY);low0=[];high0=[]
    for gi,g in enumerate(gens):
        initial=g['initial_status'];forced=max(0,g['min_up_time']-initial) if initial>0 else max(0,g['min_down_time']+initial)
        if forced:
            if initial>0:on[gi].LB=1
            else:on[gi].UB=0
        if initial>0:
            low0.append(max(g['p_min'],g['initial_power']-g['ramp_down_limit']));high0.append(min(g['p_max'],g['initial_power']+g['ramp_up_limit']))
            if g['initial_power']>g['shutdown_ramp_limit']+1e-7:on[gi].LB=1
        else:low0.append(g['p_min']);high0.append(min(g['p_max'],g['startup_ramp_limit']))
        if high0[-1]<low0[-1]-1e-7:on[gi].UB=0
    master.addConstr(gp.quicksum(g['p_min']*on[gi] for gi,g in enumerate(gens))<=min(d['demand']));master.addConstr(gp.quicksum(g['p_max']*on[gi] for gi,g in enumerate(gens))>=max(v+r for v,r in zip(d['demand'],reserve)));master.addConstr(gp.quicksum(low0[gi]*on[gi] for gi in range(G))<=d['demand'][0]);master.addConstr(gp.quicksum(high0[gi]*on[gi] for gi in range(G))>=d['demand'][0]);master.setObjective(gp.quicksum((g['no_load_cost']+g['p_min']*g['production_cost_segments'][0]['marginal_cost_per_mwh'])*on[gi] for gi,g in enumerate(gens)))
    for t in range(T):
        lows=[max(g['p_min'],g['initial_power']-(t+1)*g['ramp_down_limit']) if g['initial_status']>0 else g['p_min'] for g in gens];highs=[min(g['p_max'],g['initial_power']+(t+1)*g['ramp_up_limit']) if g['initial_status']>0 else min(g['p_max'],g['startup_ramp_limit']+t*g['ramp_up_limit']) for g in gens]
        master.addConstr(gp.quicksum(lows[gi]*on[gi] for gi in range(G))<=d['demand'][t]);master.addConstr(gp.quicksum(highs[gi]*on[gi] for gi in range(G))>=d['demand'][t])
    power=None;attempt=0
    while time.monotonic()<deadline-.1:
        master.Params.TimeLimit=max(.01,(deadline-time.monotonic()-.1)/4);master.optimize()
        print(f'knueven master attempt={attempt} masterStatus={master.Status} SolCount={master.SolCount} allocation={master.Params.TimeLimit:.3f} remaining={max(0,deadline-time.monotonic()):.3f}',file=sys.stderr,flush=True)
        if not master.SolCount:
            if master.Status in (GRB.INFEASIBLE,GRB.INF_OR_UNBD,GRB.UNBOUNDED):break
            continue
        selected=[int(on[gi].X>.5) for gi in range(G)];commit=[[selected[gi]]*T for gi in range(G)];power=dispatch(commit)
        if power is not None:break
        master.addConstr(gp.quicksum(1-on[gi] if selected[gi] else on[gi] for gi in range(G))>=1);attempt+=1
    if power is None:
        print(f'knueven search_end attempts={attempt} masterStatus={master.Status} SolCount={master.SolCount} remaining={max(0,deadline-time.monotonic()):.3f}',file=sys.stderr,flush=True)
    master.dispose()
    if power is None:raise RuntimeError('No initial-ramp-compatible full-horizon dispatch obtained')
    emit(commit,power);order=sorted(range(G),key=lambda gi:gens[gi]['no_load_cost']/max(1,gens[gi]['p_max']),reverse=True);iteration=0
    while time.monotonic()<deadline-.2:
        if iteration: rng.shuffle(order)
        for gi in order:
            if time.monotonic()>=deadline-.2:break
            trial=[best['schedule'][g['id']]['u'][:] for g in gens];trial[gi]=[0]*T;candidate=dispatch(trial)
            if candidate is not None:emit(trial,candidate)
        iteration+=1
        if iteration>=2:break
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

import time
_LOG_START = time.time()
import argparse,json,os,itertools
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    ships={s['id']:s for s in d['ships']};ports={p['id']:p['quay_length_meters'] for p in d['ports']};visits={(s['id'],v['port_call_index']):v for s in d['ships'] for v in s['port_visits']};speeds=d['speeds_knots'];cost=d['cost_parameters'];beta=d['handling_time_deviation_factor_beta_per_meter'];H=d['planning_horizon_hours'];m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4
    x={};y={};h={};a={};r={};delay={};late={};z={};occupations={p:[] for p in ports}
    for key,v in visits.items():
        sid,c=key;length=ships[sid]['length_meters'];x[key]=m.addVar(ub=ports[v['port_id']]-length);y[key]=m.addVar(lb=v['earliest_start_time_hours'],ub=H);h[key]=m.addVar();a[key]=m.addVar();r[key]=m.addVar();delay[key]=m.addVar();late[key]=m.addVar()
        signed=m.addVar(lb=-GRB.INFINITY);m.addConstr(signed==x[key]-v['ideal_berthing_position']);m.addGenConstrAbs(r[key],signed);m.addConstr(h[key]==v['min_handling_time_hours']*(1+beta*r[key]));m.addConstr(a[key]<=y[key]);m.addConstr(delay[key]>=y[key]+h[key]-v['expected_finish_time_hours']);m.addConstr(late[key]>=y[key]+h[key]-v['latest_finish_time_hours']);occupations[v['port_id']].append((x[key],y[key],h[key],length,True))
    objective=gp.quicksum(cost['waiting_cost_usd_per_hour']*(y[k]-a[k])+cost['handling_cost_usd_per_hour']*h[k]+cost['delay_cost_usd_per_hour']*delay[k]+cost['lft_penalty_usd']*late[k] for k in visits)
    for sid,ship in ships.items():
        calls=sorted(ship['port_visits'],key=lambda v:v['port_call_index']);first=(sid,calls[0]['port_call_index']);m.addConstr(a[first]==calls[0]['earliest_start_time_hours'])
        for v,w in zip(calls,calls[1:]):
            key=(sid,v['port_call_index']);nxt=(sid,w['port_call_index']);distance=d['distances_nautical_miles'][v['port_id']+'_'+w['port_id']];z[key]=m.addVars(len(speeds),vtype=GRB.BINARY);m.addConstr(z[key].sum()==1);m.addConstr(a[nxt]==y[key]+h[key]+distance*gp.quicksum(d['travel_time_hours_per_nm_by_speed'][str(sp)]*z[key][t] for t,sp in enumerate(speeds)));objective+=cost['fuel_price_usd_per_tonne']*distance*gp.quicksum(ship['fuel_consumption_per_nm_by_speed'][str(sp)]*z[key][t] for t,sp in enumerate(speeds))
    for group in d['external_ships']:
        for v in group['ships']:occupations[group['port_id']].append((v['berthing_position'],v['start_time_hours'],v['handling_time_hours'],v['length_meters'],False))
    for pid,occ in occupations.items():
        for A,B in itertools.combinations(occ,2):
            if not(A[4] or B[4]):continue
            order=m.addVars(4,vtype=GRB.BINARY);m.addConstr(order.sum()>=1);m.addConstr(A[0]+A[3]<=B[0]+ports[pid]*(1-order[0]));m.addConstr(B[0]+B[3]<=A[0]+ports[pid]*(1-order[1]));m.addConstr(A[1]+A[2]<=B[1]+2*H*(1-order[2]));m.addConstr(B[1]+B[2]<=A[1]+2*H*(1-order[3]))
    m.setObjective(objective);best=None
    def emit(get):
        nonlocal best
        output=[];value=0
        for sid,ship in ships.items():
            rows=[];sailing=[]
            for v in ship['port_visits']:
                key=(sid,v['port_call_index']);pos=get(x[key]);start=get(y[key]);arrival=get(a[key]);dev=abs(pos-v['ideal_berthing_position']);handling=v['min_handling_time_hours']*(1+beta*dev);dd=max(0,start+handling-v['expected_finish_time_hours']);ll=max(0,start+handling-v['latest_finish_time_hours']);rows.append({'port_call_index':key[1],'port_id':v['port_id'],'berthing_position':pos,'start_time':start,'handling_time':handling,'arrival_time':arrival,'delay':dd,'lft_violation':ll,'position_deviation':dev});value+=cost['waiting_cost_usd_per_hour']*(start-arrival)+cost['handling_cost_usd_per_hour']*handling+cost['delay_cost_usd_per_hour']*dd+cost['lft_penalty_usd']*ll
                if key in z:
                    sp=speeds[max(range(len(speeds)),key=lambda t:get(z[key][t]))];nxt=visits[sid,key[1]+1];distance=d['distances_nautical_miles'][v['port_id']+'_'+nxt['port_id']];value+=cost['fuel_price_usd_per_tonne']*distance*ship['fuel_consumption_per_nm_by_speed'][str(sp)];sailing.append({'from_call':key[1],'to_call':key[1]+1,'speed_knots':sp})
            output.append({'ship_id':sid,'port_visits':rows,'speed_selections':sailing})
        s={'objective_value':value,'ships':output}
        if best is None or value<best['objective_value']:
            best=s
            if logger:logger.log_solution(value,s)
    def callback(model,where):
        if where==GRB.Callback.MIPSOL:emit(model.cbGetSolution)
    m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    if best is None:raise RuntimeError('No berth-allocation incumbent')
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

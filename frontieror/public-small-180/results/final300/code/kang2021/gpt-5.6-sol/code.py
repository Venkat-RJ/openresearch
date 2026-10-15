import time
_LOG_START = time.time()
import argparse,json,random,math,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    depot=d['depot_index'];nodes=d['customer_indices'];tv=d['truck_travel_times'];bt=d['drone_battery_consumption'];tau=d['drone_delivery_time'];caps=d['parameters']['drone_battery_capacities'];L=len(caps);rng=random.Random(0)
    best=None
    def opt(route):
        changed=True
        while changed and time.monotonic()<deadline:
            changed=False
            for a in range(1,len(route)-2):
                for b in range(a+1,len(route)-1):
                    old=tv[route[a-1]][route[a]]+tv[route[b]][route[b+1]]
                    new=tv[route[a-1]][route[b]]+tv[route[a]][route[b+1]]
                    # Internal reversal terms also matter for asymmetric service times.
                    new+=sum(tv[route[k+1]][route[k]]-tv[route[k]][route[k+1]] for k in range(a,b))
                    if new<old-1e-8:route[a:b+1]=reversed(route[a:b+1]);changed=True
        return route
    def emit(route,assign,wait):
        nonlocal best
        obj=sum(tv[a][b] for a,b in zip(route,route[1:]))+sum(max(v) for v in wait.values())
        if best is None or obj<best['objective_value']:
            edges=[[a,b] for a,b in zip(route,route[1:])];edges[-1][1]=d['num_nodes']
            best={'objective_value':obj,'truck_route':edges,'drone_assignments':[{'drone':l,'dispatch_node':i,'delivery_node':j} for l,i,j in assign],'waiting_times':{str(i):max(v) for i,v in wait.items() if max(v)>0}}
            if logger:logger.log_solution(obj,best)
    count=0
    while time.monotonic()<deadline:
        left=set(nodes);route=[depot]
        while left:
            options=sorted(left,key=lambda j:tv[route[-1]][j]);j=options[0] if count==0 else options[rng.randrange(min(3,len(options)))];route.append(j);left.remove(j)
        route.append(depot);opt(route)
        assign=[];usage=[0]*L;wait={i:[0]*L for i in route[:-1]};emit(route,assign,wait)
        while len(route)>3 and time.monotonic()<deadline:
            moves=[];dispatches={i for l,i,j in assign}
            for pos in range(1,len(route)-1):
                j=route[pos]
                if j in dispatches:continue
                saved=tv[route[pos-1]][j]+tv[j][route[pos+1]]-tv[route[pos-1]][route[pos+1]]
                for i in route[:-1]:
                    if i==j:continue
                    for l in range(L):
                        if usage[l]+bt[l][i][j]>caps[l]+1e-8:continue
                        increase=max(max(wait[i]),wait[i][l]+tau[l][i][j])-max(wait[i])
                        gain=saved-increase
                        if gain>0:moves.append((gain,pos,l,i,j))
            if not moves:break
            moves.sort(reverse=True);m=moves[0] if count==0 else moves[rng.randrange(min(3,len(moves)))];gain,pos,l,i,j=m
            route.pop(pos);assign.append((l,i,j));usage[l]+=bt[l][i][j];wait[i][l]+=tau[l][i][j];wait.pop(j);opt(route);emit(route,assign,wait)
        emit(route,assign,wait);count+=1
        if len(nodes)<=10 and count>=100:break
    if len(nodes)<=10 and time.monotonic()<deadline:
        sink=d['num_nodes'];dispatch=[depot]+nodes;arcs=[(i,j) for i in dispatch for j in nodes+[sink] if i!=j and not(i==depot and j==sink)];triples=[(l,i,j) for l in range(L) for i in dispatch for j in nodes if i!=j];m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;x=m.addVars(arcs,vtype=GRB.BINARY);h=m.addVars(triples,vtype=GRB.BINARY);order=m.addVars(dispatch+[sink],lb=0,ub=len(nodes)+1);w=m.addVars(dispatch,lb=0);order[depot].UB=0
        m.addConstr(x.sum(depot,'*')==1);m.addConstr(x.sum('*',sink)==1)
        for i in nodes:m.addConstr(x.sum(i,'*')==x.sum('*',i));m.addConstr(x.sum('*',i)+h.sum('*','*',i)==1)
        for i,j in arcs:m.addConstr(order[j]>=order[i]+1-(len(nodes)+1)*(1-x[i,j]))
        for i in dispatch:
            m.addConstr(h.sum('*',i,'*')<=len(nodes)*x.sum(i,'*'))
            for l in range(L):m.addConstr(w[i]>=gp.quicksum(tau[l][i][j]*h[l,i,j] for j in nodes if j!=i))
        for l in range(L):m.addConstr(gp.quicksum(bt[l][i][j]*h[l,i,j] for i in dispatch for j in nodes if i!=j)<=caps[l])
        m.setObjective(gp.quicksum(tv[i][depot if j==sink else j]*x[i,j] for i,j in arcs)+w.sum());initial_edges=set(map(tuple,best['truck_route']));initial_h={(a['drone'],a['dispatch_node'],a['delivery_node']) for a in best['drone_assignments']}
        for key,var in x.items():var.Start=int(key in initial_edges)
        for key,var in h.items():var.Start=int(key in initial_h)
        def callback(model,where):
            nonlocal best
            if where!=GRB.Callback.MIPSOL:return
            xv=model.cbGetSolution(x);hv=model.cbGetSolution(h);edges=[[i,j] for (i,j),value in xv.items() if value>.5];assign=[{'drone':l,'dispatch_node':i,'delivery_node':j} for (l,i,j),value in hv.items() if value>.5];waiting={i:max(sum(tau[l][i][j] for ll,ii,j in triples if ll==l and ii==i and hv[ll,ii,j]>.5) for l in range(L)) for i in dispatch};value=sum(tv[i][depot if j==sink else j] for i,j in edges)+sum(waiting.values());out={'objective_value':value,'truck_route':edges,'drone_assignments':assign,'waiting_times':{str(i):v for i,v in waiting.items() if v>0}}
            if value<best['objective_value']:
                best=out
                if logger:logger.log_solution(value,out)
        m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

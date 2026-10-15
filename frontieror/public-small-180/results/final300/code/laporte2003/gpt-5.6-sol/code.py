import time
_LOG_START = time.time()
import argparse,json,random,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']): os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    n=d['num_vertices'];depot=d['depot'];markets=[i for i in range(n) if i!=depot];demands=d['product_demands'];available={int(k):v for k,v in d['product_markets'].items()};cost={tuple(map(int,k.split(','))):v for k,v in d['travel_costs'].items()};prices={tuple(map(int,k.split(','))):v for k,v in d['purchase_costs'].items()};supply={tuple(map(int,k.split(','))):v for k,v in d.get('supplies',{}).items()};unlimited=d.get('supply_type','limited')=='unlimited';rng=random.Random(0);best=None
    def travel(i,j): return cost[min(i,j),max(i,j)] if i!=j else 0
    def purchase(active):
        total=0;purchases={}
        for k,need in enumerate(demands):
            purchases[str(k)]={}
            for i in sorted((i for i in available[k] if i in active),key=lambda i:prices.get((i,k),0)):
                amount=min(need,demands[k] if unlimited else supply.get((k,i),0));need-=amount;total+=amount*prices.get((i,k),0)
                if amount: purchases[str(k)][str(i)]=amount
            if need>1e-7: return None
        return total,purchases
    def route(active):
        r=[depot];remaining=set(active)
        while remaining:
            i=min(remaining,key=lambda j:travel(r[-1],j));r.append(i);remaining.remove(i)
        changed=True
        while changed and time.monotonic()<deadline:
            changed=False
            for i in range(1,len(r)-1):
                for j in range(i+1,len(r)):
                    after=r[(j+1)%len(r)];gain=travel(r[i-1],r[i])+travel(r[j],after)-travel(r[i-1],r[j])-travel(r[i],after)
                    if gain>1e-8:r[i:j+1]=reversed(r[i:j+1]);changed=True
        return r
    def emit(active,r=None):
        nonlocal best
        if len(active)<2: return None
        q=purchase(active)
        if q is None:return None
        if r is None:r=route(active)
        edges=[[r[i],r[(i+1)%len(r)]] for i in range(len(r))];value=q[0]+sum(travel(i,j) for i,j in edges);s={'objective_value':value,'visited_markets':sorted(active),'tour':r+[depot],'tour_edges':edges,'purchases':q[1]}
        if best is None or value<best['objective_value']-1e-8:
            best=s
            if logger:logger.log_solution(value,s)
        return s
    emit(set(markets))
    pairs=[(k,i) for k in range(len(demands)) for i in available[k]];edges=list(cost)
    groups={}
    if unlimited:
        for k,need in enumerate(demands):
            for i in available[k]:
                price=need*prices.get((i,k),0)
                if price<=best['objective_value']+1e-8:groups.setdefault((k,price),[]).append(i)
    if unlimited and len(edges)+n+len(groups)<=1900 and time.monotonic()<deadline:
        pairs=list(groups)
    else:groups={}
    if len(edges)+n+len(pairs)<=1900 and time.monotonic()<deadline:
        m=gp.Model();m.Params.OutputFlag=0;m.Params.Seed=0;m.Params.Threads=1;m.Params.NumericFocus=0;m.Params.MIPGap=1e-4;m.Params.LazyConstraints=1
        x=m.addVars(edges,vtype=GRB.BINARY);y=m.addVars(range(n),vtype=GRB.BINARY);z=m.addVars(pairs,lb=0,ub=1 if groups else GRB.INFINITY,vtype=GRB.BINARY if groups else GRB.CONTINUOUS);y[depot].LB=1
        for i in range(n):m.addConstr(gp.quicksum(x[e] for e in edges if i in e)==2*y[i])
        if groups:
            for k in range(len(demands)):m.addConstr(gp.quicksum(z[key] for key in pairs if key[0]==k)==1)
            for key,market_group in groups.items():m.addConstr(z[key]<=gp.quicksum(y[i] for i in market_group))
            purchase_obj=gp.quicksum(price*z[k,price] for k,price in pairs)
        else:
            for k in range(len(demands)):
                m.addConstr(gp.quicksum(z[k,i] for i in available[k])==demands[k])
                for i in available[k]:m.addConstr(z[k,i]<=(demands[k] if unlimited else supply.get((k,i),0))*y[i])
            purchase_obj=gp.quicksum(prices.get((i,k),0)*z[k,i] for k,i in pairs)
        m.setObjective(gp.quicksum(cost[e]*x[e] for e in edges)+purchase_obj)
        m.Params.Cutoff=best['objective_value']+1e-6
        def callback(model,where):
            if where!=GRB.Callback.MIPSOL:return
            xv=model.cbGetSolution(x);yv=model.cbGetSolution(y);selected={i for i in range(n) if yv[i]>.5};adj={i:[] for i in selected}
            for (i,j),v in xv.items():
                if v>.5:adj[i].append(j);adj[j].append(i)
            remaining=set(selected);components=[]
            while remaining:
                seen={next(iter(remaining))};stack=list(seen)
                while stack:
                    i=stack.pop()
                    for j in adj[i]:
                        if j not in seen:seen.add(j);stack.append(j)
                remaining-=seen;components.append(seen)
            if len(components)>1:
                for component in components:
                    if depot in component:continue
                    crossing=gp.quicksum(x[e] for e in edges if (e[0] in component)!=(e[1] in component))
                    for i in component:model.cbLazy(crossing>=2*y[i])
            else:
                r=[depot];previous=None;current=depot
                while len(r)<len(selected):
                    nxt=next(j for j in adj[current] if j!=previous and (j not in r or len(r)==len(selected)))
                    r.append(nxt);previous,current=current,nxt
                emit(selected-{depot},r)
        m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(callback)
    else:
        iteration=0
        while time.monotonic()<deadline:
            active=set(best['visited_markets']);order=markets[:];rng.shuffle(order)
            if iteration:
                active.update(rng.sample(markets,min(len(markets),max(1,n//8))))
            changed=True
            while changed and time.monotonic()<deadline:
                changed=False;current=emit(active)
                for i in order:
                    candidate=active^{i};s=emit(candidate)
                    if s and s['objective_value']<current['objective_value']-1e-8:active=candidate;current=s;changed=True
                    if time.monotonic()>=deadline:break
            iteration+=1
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

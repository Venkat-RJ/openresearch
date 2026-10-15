import time
_LOG_START = time.time()
import argparse,json,random,math,itertools
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    n=d['num_customers'];Q=d['vehicle_capacity'];dist=d['distance_matrix'];a=d['cost_function']['a'];b=d['cost_function']['b'];customers={c['id']:c for c in d['customers']};depot=d['depot'];rng=random.Random(0);best=None
    def evaluate(route,deliveries):
        current=depot['time_window'][0]+depot['service_time'];prev=depot['id'];load=sum(deliveries.values());value=0;edges=[]
        if load>Q+1e-7:return None
        for i in route:
            value+=dist[prev][i]*(a*load+b);edges.append({'from':prev,'to':i,'load_weight':load});current=max(current+dist[prev][i],customers[i]['time_window'][0])
            if current>customers[i]['time_window'][1]+1e-7:return None
            current+=customers[i]['service_time'];load-=deliveries[i];prev=i
        value+=dist[prev][depot['id']]*b;edges.append({'from':prev,'to':n+1,'load_weight':max(0,load)});current+=dist[prev][depot['id']]
        if current>min(depot['time_window'][1],d['scheduling_horizon'])+1e-7:return None
        return {'objective_value':value,'route':[depot['id']]+route+[n+1],'deliveries':{str(i):v for i,v in deliveries.items()},'edges':edges}
    def optimize(deliveries,seed=None):
        vertices=list(deliveries);winner=None
        if len(vertices)<=6:
            for route in itertools.permutations(vertices):
                if time.monotonic()>=deadline:break
                s=evaluate(list(route),deliveries)
                if s and (winner is None or s['objective_value']<winner['objective_value']):winner=s
        else:
            candidates=[seed] if seed else [];candidates+=[sorted(vertices,key=lambda i:customers[i]['time_window'][1]),sorted(vertices,key=lambda i:dist[0][i])]
            for route in candidates:
                s=evaluate(route,deliveries)
                if not s:continue
                changed=True
                while changed and time.monotonic()<deadline:
                    changed=False
                    for i in range(len(route)):
                        for j in range(len(route)):
                            new=route[:];v=new.pop(i);new.insert(j,v);trial=evaluate(new,deliveries)
                            if trial and trial['objective_value']<s['objective_value']-1e-8:route=new;s=trial;changed=True
                if winner is None or s['objective_value']<winner['objective_value']:winner=s
        return winner
    def emit(routes):
        nonlocal best
        value=sum(r['objective_value'] for r in routes);s={'objective_value':value,'routes':[{k:v for k,v in r.items() if k!='objective_value'}|{'vehicle':i} for i,r in enumerate(routes)]}
        if best is None or value<best['objective_value']-1e-8:
            best=s
            if logger:logger.log_solution(value,s)
    base=[]
    for i,c in customers.items():
        remain=c['demand']
        while remain>1e-8:
            qty=min(Q,remain);s=evaluate([i],{i:qty})
            if s is None:raise RuntimeError('Singleton delivery infeasible')
            base.append(s);remain-=qty
    emit(base);run=0
    while time.monotonic()<deadline:
        routes=base[:];changed=True
        while changed and time.monotonic()<deadline:
            changed=False;chosen=None;gain=0;pairs=list(itertools.combinations(range(len(routes)),2));rng.shuffle(pairs)
            for i,j in pairs:
                merged={int(k):v for k,v in routes[i]['deliveries'].items()}
                for k,v in routes[j]['deliveries'].items():merged[int(k)]=merged.get(int(k),0)+v
                if sum(merged.values())>Q+1e-7:continue
                s=optimize(merged)
                if s:
                    saving=routes[i]['objective_value']+routes[j]['objective_value']-s['objective_value']
                    if saving>gain+1e-8:gain=saving;chosen=(i,j,s)
                if time.monotonic()>=deadline:break
            if chosen:
                i,j,s=chosen;routes=[r for k,r in enumerate(routes) if k not in (i,j)]+[s];changed=True;emit(routes)
        emit(routes);run+=1
        if n<=10 or run>=40:break
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

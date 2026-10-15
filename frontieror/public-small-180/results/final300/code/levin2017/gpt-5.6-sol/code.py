import time
_LOG_START = time.time()
import argparse,json,heapq,collections
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    T=d['time_parameters']['time_horizon_T'];dt=d['time_parameters']['time_step_duration_seconds'];links={(v['from'],v['to']):v for v in d['network']['links']};centroids=sorted(n['id'] for n in d['network']['nodes'] if n['type']=='centroid');out=collections.defaultdict(list)
    for i,j in links:out[i].append(j)
    paths={}
    for origin in centroids:
        queue=[(0,origin,[])];seen=set()
        while queue:
            value,node,path=heapq.heappop(queue)
            if node in seen:continue
            seen.add(node)
            if node in centroids and node!=origin:paths[origin,node]=path;continue
            for nxt in out[node]:heapq.heappush(queue,(value+links[node,nxt]['free_flow_travel_time_steps'],nxt,path+[(node,nxt)]))
    arrivals=collections.defaultdict(float);newdemand=collections.defaultdict(float)
    for od in d['demand']['od_pairs']:
        for t in od['departure_times']:newdemand[od['origin'],od['destination'],t]+=1
    parked={c:float(d['fleet']['initial_distribution'].get(c,0)) for c in centroids};waiting={(r,s):0. for r in centroids for s in centroids};capacity=collections.defaultdict(float);entered=collections.defaultdict(float);exited=collections.defaultdict(float);turns=collections.defaultdict(float);departures=collections.defaultdict(float);picked=collections.defaultdict(float);p={};omega={}
    def key(*v):return '|'.join(map(str,v))
    for t in range(T+1):
        for c in centroids:parked[c]+=arrivals[c,t];p[key(c,t)]=parked[c]
        for (r,s),quantity in waiting.items():omega[key(r,s,t)]=quantity
        if t==T:break
        ordered=sorted(waiting,key=lambda pair:(-waiting[pair],pair))
        for r,s in ordered:
            if r==s or waiting[r,s]<1e-9 or parked[r]<1e-9:continue
            path=paths[r,s];clock=t;plans=[];amount=min(waiting[r,s],parked[r])
            for arc in path:
                duration=links[arc]['free_flow_travel_time_steps'];plans.append((arc,clock,clock+duration));clock+=duration
                if links[arc]['type']=='road':amount=min(amount,max(0,links[arc]['capacity_vph']*dt/3600-capacity[arc,clock]),max(0,links[arc]['jam_density_vehicles']))
            if clock+1>T or amount<1e-9:continue
            waiting[r,s]-=amount;parked[r]-=amount;picked[r,s,t]+=amount;departures[r,path[0][1],s,t]+=amount;arrivals[s,clock+1]+=amount
            for index,(arc,start,finish) in enumerate(plans):
                entered[arc,s,start+1]+=amount;exited[arc,s,finish+1]+=amount
                if links[arc]['type']=='road':capacity[arc,finish]+=amount
                if index+1<len(plans):turns[arc[0],arc[1],plans[index+1][0][1],s,finish]+=amount
        for pair in waiting:waiting[pair]+=newdemand[pair[0],pair[1],t]
        if time.monotonic()>=deadline:raise RuntimeError('SAV construction deadline')
    if any(v>1e-7 for v in waiting.values()):raise RuntimeError('Unserved travelers at horizon')
    NU={};ND={};total=0
    for arc in links:
        for s in centroids:
            u=down=0
            for t in range(T+1):
                u+=entered[arc,s,t];down+=exited[arc,s,t]
                if u:NU[key(*arc,s,t)]=u
                if down:ND[key(*arc,s,t)]=down
                total+=u-down
    total+=sum(omega.values());result={'objective_value':total,'y':{key(*k):v for k,v in turns.items()},'y_centroid':{key(*k):v for k,v in departures.items()},'N_U':NU,'N_D':ND,'p':p,'e':{key(*k):v for k,v in picked.items()},'omega':omega}
    if logger:logger.log_solution(total,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

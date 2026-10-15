import time
_LOG_START = time.time()
import argparse,json
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    nodes=list(range(d['graph']['num_vertices']));adj={i:set() for i in nodes}
    for i,j in d['graph']['edges']:adj[i].add(j);adj[j].add(i)
    best=None
    def emit(colors):
        nonlocal best
        k=max(colors.values())+1
        if best is not None and k>=best['objective_value']:return
        classes=[]
        for c in range(k):
            group={i for i in nodes if colors[i]==c}
            for i in nodes:
                if i not in group and not(adj[i]&group):group.add(i)
            classes.append(sorted(group))
        best={'objective_value':k,'selected_sets':classes,'coloring':{str(i):c for i,c in colors.items()}}
        if logger:logger.log_solution(k,best)
    colors={}
    while len(colors)<len(nodes):
        v=max((i for i in nodes if i not in colors),key=lambda i:(len({colors[j] for j in adj[i] if j in colors}),len(adj[i])));blocked={colors[j] for j in adj[v] if j in colors};c=0
        while c in blocked:c+=1
        colors[v]=c
    emit(colors);colors={}
    def search(used):
        if time.monotonic()>=deadline:return
        if len(colors)==len(nodes):emit(colors);return
        if used>=best['objective_value']:return
        v=max((i for i in nodes if i not in colors),key=lambda i:(len({colors[j] for j in adj[i] if j in colors}),len(adj[i])));blocked={colors[j] for j in adj[v] if j in colors}
        for c in range(min(used+1,best['objective_value']-1)):
            if c not in blocked:colors[v]=c;search(max(used,c+1));del colors[v]
    search(0);return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

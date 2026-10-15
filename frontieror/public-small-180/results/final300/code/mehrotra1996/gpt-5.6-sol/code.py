import time
_LOG_START = time.time()
import argparse,json,random
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    nodes=d['graph']['nodes'];adj={i:set() for i in nodes}
    for i,j in d['graph']['edges']:adj[i].add(j);adj[j].add(i)
    best=None
    def emit(colors):
        nonlocal best
        labels=sorted(set(colors.values()));mapping={c:k for k,c in enumerate(labels)};colors={i:mapping[c] for i,c in colors.items()};s={'objective_value':len(labels),'vertex_colors':{str(i):c for i,c in colors.items()},'color_classes':[[i for i in nodes if colors[i]==c] for c in range(len(labels))]}
        if best is None or s['objective_value']<best['objective_value']:
            best=s
            if logger:logger.log_solution(s['objective_value'],s)
    colors={}
    while len(colors)<len(nodes):
        v=max((i for i in nodes if i not in colors),key=lambda i:(len({colors[j] for j in adj[i] if j in colors}),len(adj[i])));blocked={colors[j] for j in adj[v] if j in colors};c=0
        while c in blocked:c+=1
        colors[v]=c
    emit(colors);colors={};visited=0
    def search(used):
        nonlocal visited
        visited+=1
        if visited%64==0 and time.monotonic()>=deadline:return
        if len(colors)==len(nodes):emit(colors);return
        if used>=best['objective_value']:return
        v=max((i for i in nodes if i not in colors),key=lambda i:(len({colors[j] for j in adj[i] if j in colors}),len(adj[i])));blocked={colors[j] for j in adj[v] if j in colors}
        for c in range(min(used+1,best['objective_value']-1)):
            if c in blocked:continue
            colors[v]=c;search(max(used,c+1));del colors[v]
            if time.monotonic()>=deadline:return
    search(0);return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

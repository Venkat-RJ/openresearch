import time
_LOG_START = time.time()
import argparse,json,random,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[key]='1'
import numpy as np
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    c=np.asarray(d['cost_matrices']['assignment_costs']['matrix'],float); distances=np.asarray(d['cost_matrices']['interconnection_costs']['matrix'],float); weights=np.asarray(d['steiner_nodes']['node_weights'],float); n=len(weights); best=None; rng=random.Random(0)
    def evaluate(active):
        active=sorted(active); assignments=np.argmin(c[:,active],axis=1); total=float(c[np.arange(len(c)),np.asarray(active)[assignments]].sum()+weights[active].sum()); edges=[]; seen={active[0]}; remaining=set(active[1:])
        while remaining:
            value,i,j=min((distances[i,j],i,j) for i in seen for j in remaining); total+=float(value); edges.append([i,j]); seen.add(j); remaining.remove(j)
        return {'objective_value':total,'active_steiner_nodes':active,'assignments':{str(i):active[int(j)] for i,j in enumerate(assignments)},'steiner_tree_edges':edges}
    def emit(s):
        nonlocal best
        if best is None or s['objective_value']<best['objective_value']-1e-8:
            best=s
            if logger: logger.log_solution(s['objective_value'],s)
    for i in range(n): emit(evaluate({i}))
    if n<=19:
        for mask in range(1,1<<n):
            if time.monotonic()>=deadline: break
            emit(evaluate({i for i in range(n) if mask>>i&1}))
    else:
        run=0
        while time.monotonic()<deadline:
            active=set(best['active_steiner_nodes']) if run%3 else {rng.randrange(n)}
            if run: active.symmetric_difference_update(rng.sample(range(n),min(n,max(1,n//15))))
            if not active: active={rng.randrange(n)}
            current=evaluate(active)
            while time.monotonic()<deadline:
                chosen=current; chosen_set=active
                for j in range(n):
                    candidate=active^{j}
                    if not candidate: continue
                    s=evaluate(candidate)
                    if s['objective_value']<chosen['objective_value']-1e-8: chosen=s;chosen_set=candidate
                    if time.monotonic()>=deadline: break
                if chosen_set==active: break
                active=chosen_set;current=chosen;emit(current)
            emit(current);run+=1
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))

import argparse,json,os,time
wall_started=time.time()
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']): del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True)
a=p.parse_args(); end=time.monotonic()+a.time_limit-1; d=json.load(open(a.instance_path)); M=d['parameters']['num_trips_M']; duties=d['duties']; costs=d['costs']; logger=configured_logger(a.log_path,sense='minimize'); best=float('inf'); selected=[]
unique={}
for j,ds in enumerate(duties):
 key=tuple(sorted(ds))
 if key not in unique or costs[j]<costs[unique[key]]: unique[key]=j
js=list(unique.values()); mask={j:sum(1<<i for i in duties[j]) for j in js}; cover=[[j for j in js if i in duties[j]] for i in range(M)]; full=(1<<M)-1
for cs in cover: cs.sort(key=lambda j:costs[j]/len(duties[j]))
def emit(chosen):
 global best,selected
 z=sum(costs[j] for j in chosen)
 if z>=best-1e-9: return
 best=z; selected=list(chosen); sol={'objective_value':z,'selected_duties':selected}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(z,sol)
limit=min(end,time.monotonic()+1)
def find(used,chosen):
 if used==full: emit(chosen); return True
 if time.monotonic()>=limit: return False
 i=min((i for i in range(M) if not (used>>i)&1),key=lambda i:sum(not mask[j]&used for j in cover[i]))
 for j in cover[i]:
  if not mask[j]&used and find(used|mask[j],chosen+[j]): return True
 return False
find(0,[])
if time.monotonic()<end:
 assert len(js)<=2000 and M<=2000
 m=gp.Model(); m.Params.Threads=1; m.Params.Seed=0; m.Params.MIPGap=1e-4; m.Params.NumericFocus=0; x={j:m.addVar(vtype=GRB.BINARY,obj=costs[j]) for j in js}
 for i in range(M): m.addConstr(gp.quicksum(x[j] for j in cover[i])==1)
 for j in js: x[j].Start=int(j in selected) if selected else GRB.UNDEFINED
 def cb(model,where):
  if where==GRB.Callback.MIPSOL: emit([j for j,v in zip(js,model.cbGetSolution(list(x.values()))) if v>.5])
 m.Params.TimeLimit=max(.1,end-time.monotonic()); m.optimize(cb)
 if m.SolCount: emit([j for j in js if x[j].X>.5])

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
a=p.parse_args(); start=time.monotonic(); d=json.load(open(a.instance_path)); logger=configured_logger(a.log_path,sense='minimize'); J=d['num_items']; T=d['num_periods']; M=d['num_machines']; caps=[[v-1 for v in vs] for vs in d['machine_capabilities']]; demand=d['demand']; I0=d.get('initial_inventory',[0]*J)
def cost(name,j,m): return d[name][f'item_{j+1}_machine_{m+1}']
m=gp.Model(); m.Params.Threads=1; m.Params.Seed=0; m.Params.MIPGap=1e-4; m.Params.NumericFocus=0
X={(j,k,t):m.addVar(obj=cost('production_cost',j,k)) for j in range(J) for k in caps[j] for t in range(T)}; Y={key:m.addVar(vtype=GRB.BINARY,obj=cost('setup_cost',key[0],key[1])) for key in X}; I={(j,t):m.addVar(obj=d['holding_cost'][j]) for j in range(J) for t in range(T)}
for j in range(J):
 for t in range(T):
  m.addConstr(gp.quicksum(X[j,k,t] for k in caps[j])+(I[j,t-1] if t else I0[j])-I[j,t]==demand[j][t]); m.addConstr(gp.quicksum(Y[j,k,t] for k in caps[j])<=d['max_machines_per_item_per_period'][j])
for k in range(M):
 for t in range(T): m.addConstr(gp.quicksum(cost('production_time',j,k)*X[j,k,t]+cost('setup_time',j,k)*Y[j,k,t] for j in range(J) if k in caps[j])<=d['capacity'][k][t])
for (j,k,t),v in X.items():
 bound=max(0,min(sum(demand[j][t:]),(d['capacity'][k][t]-cost('setup_time',j,k))/cost('production_time',j,k))); m.addConstr(v<=bound*Y[j,k,t])
m.update(); assert m.NumVars<=2000 and m.NumConstrs<=2000
variables=list(X.values())+list(Y.values())+list(I.values()); best=float('inf')
def emit(vs):
 global best
 xv=vs[:len(X)]; yv=vs[len(X):2*len(X)]; iv=vs[2*len(X):]; setups=[int(v>.5) for v in yv]; xv=[max(0,float(v)) if s else 0. for v,s in zip(xv,setups)]; iv=[max(0,float(v)) for v in iv]; z=sum(cost('production_cost',j,k)*v for (j,k,t),v in zip(X,xv))+sum(cost('setup_cost',j,k)*v for (j,k,t),v in zip(Y,setups))+sum(d['holding_cost'][j]*v for (j,t),v in zip(I,iv))
 if z>=best-1e-7: return
 best=z; sol={'objective_value':z,'production':{f'X_item{j+1}_machine{k+1}_period{t+1}':v for (j,k,t),v in zip(X,xv) if v>1e-8},'setups':{f'Y_item{j+1}_machine{k+1}_period{t+1}':v for (j,k,t),v in zip(Y,setups) if v},'inventory':{f'I_item{j+1}_period{t+1}':v for (j,t),v in zip(I,iv) if v>1e-8}}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(z,sol)
def cb(model,where):
 if where==GRB.Callback.MIPSOL: emit(model.cbGetSolution(variables))
m.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-1); m.optimize(cb)
if m.SolCount: emit([v.X for v in variables])

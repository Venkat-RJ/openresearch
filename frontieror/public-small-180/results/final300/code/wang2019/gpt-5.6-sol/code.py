import argparse,json,os,time
wall_started=time.time()
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
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
a=p.parse_args(); start=time.monotonic(); d=json.load(open(a.instance_path)); logger=configured_logger(a.log_path,sense='minimize')
ns=d['nodes']; arcs=d['arcs']; ks=d['commodities']; fs=d['fleet_types']
def indexed(o,key,i,fallback):
 v=o.get(key)
 if isinstance(v,dict): return v.get(str(i),v.get(i,o[fallback]))
 if isinstance(v,list) and 0<=i<len(v): return v[i]
 return o[fallback]
m=gp.Model(); m.Params.Threads=1; m.Params.Seed=0; m.Params.MIPGap=1e-4; m.Params.NumericFocus=0
x={(k['id'],e['from'],e['to']):m.addVar(obj=indexed(e,'flow_costs',k['id'],'flow_cost'),ub=k['demand']) for k in ks for e in arcs}
y={(f['id'],e['from'],e['to']):m.addVar(vtype=GRB.INTEGER,obj=indexed(f,'arc_fixed_costs',e['id'],'fixed_cost')) for f in fs for e in arcs}
for k in ks:
 for n in ns:
  rhs=k['demand'] if n==k['origin'] else -k['demand'] if n==k['destination'] else 0
  m.addConstr(gp.quicksum(x[k['id'],e['from'],e['to']] for e in arcs if e['from']==n)-gp.quicksum(x[k['id'],e['from'],e['to']] for e in arcs if e['to']==n)==rhs)
for f in fs:
 for n in ns: m.addConstr(gp.quicksum(y[f['id'],e['from'],e['to']] for e in arcs if e['from']==n)==gp.quicksum(y[f['id'],e['from'],e['to']] for e in arcs if e['to']==n))
for e in arcs: m.addConstr(gp.quicksum(x[k['id'],e['from'],e['to']] for k in ks)<=gp.quicksum(f['capacity']*y[f['id'],e['from'],e['to']] for f in fs))
m.update(); assert m.NumVars<=2000 and m.NumConstrs<=2000
allvars=list(x.values())+list(y.values())
def emit(vals):
 flows={f'x_{k}_{i}_{j}':max(0,float(v)) for (k,i,j),v in zip(x,vals[:len(x)]) if v>1e-8}
 vehicles={f'y_{f}_{i}_{j}':int(round(v)) for (f,i,j),v in zip(y,vals[len(x):]) if v>.5}
 obj=sum(indexed(e,'flow_costs',k['id'],'flow_cost')*flows.get(f"x_{k['id']}_{e['from']}_{e['to']}",0) for k in ks for e in arcs)+sum(indexed(f,'arc_fixed_costs',e['id'],'fixed_cost')*vehicles.get(f"y_{f['id']}_{e['from']}_{e['to']}",0) for f in fs for e in arcs)
 s={'objective_value':obj,'flows':flows,'vehicles':vehicles}; json.dump(s,open(a.solution_path,'w')); logger.log_solution(obj,s)
def cb(model,where):
 if where==GRB.Callback.MIPSOL: emit(model.cbGetSolution(allvars))
m.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-1); m.optimize(cb)
if m.SolCount: emit([v.X for v in allvars])

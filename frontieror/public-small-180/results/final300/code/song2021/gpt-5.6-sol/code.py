import argparse,json,time,math,random,os
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

def main():
 a=argparse.ArgumentParser(); a.add_argument('--instance_path'); a.add_argument('--solution_path'); a.add_argument('--time_limit',type=int,default=60); a.add_argument('--log_path'); a=a.parse_args(); start=time.monotonic(); end=start+max(.1,a.time_limit-1); d=json.load(open(a.instance_path)); H=d['planning_horizon_H']; orders=d['orders']; cap=[0]+d['capacities']; sigma=d['unit_cost_nonregular_capacity_sigma']; usage=[0.]*(H+1); chosen={}; logger=configured_logger(a.log_path,sense='minimize') if a.log_path else None; rng=random.Random(0)
 intervals={}
 for o in orders:
  lb=o['intensity_lower_bound_LB']; ub=o['intensity_upper_bound_UB']; r=o['release_date_r']; j=o['order_id']; intervals[j]=[(k,k+L-1) for L in range(math.ceil(1/ub-1e-8),min(H-r+1,math.floor(1/lb+1e-8))+1) for k in range(r,H-L+2)]
 def evaluate(o,k,l):
  p=o['work_content_p']; lb=o['intensity_lower_bound_LB']; ub=o['intensity_upper_bound_UB']; ys={t:lb for t in range(k,l+1)}; remaining=1-(l-k+1)*lb
  for t in sorted(ys,key=lambda t: max(0,usage[t]+lb*p-cap[t])):
   amount=min(remaining,ub-lb,max(0,(cap[t]-usage[t])/p-lb)); ys[t]+=amount; remaining-=amount
  for t in ys:
   amount=min(remaining,ub-ys[t]); ys[t]+=amount; remaining-=amount
  cost=o['tardiness_cost_w']*max(0,l-o['due_date_d'])+sigma*sum(max(0,usage[t]+p*y-cap[t])-max(0,usage[t]-cap[t]) for t,y in ys.items()); return cost,ys
 def place(o):
  j=o['order_id']; best=None
  for k,l in intervals[j]:
   cost,ys=evaluate(o,k,l)
   if best is None or cost<best[0]-1e-9: best=cost,k,l,ys
  cost,k,l,ys=best; chosen[j]=(k,l,ys)
  for t,y in ys.items(): usage[t]+=o['work_content_p']*y
 def save():
  nonreg={str(t):max(0,usage[t]-cap[t]) for t in range(1,H+1)}; z=sum(o['tardiness_cost_w']*max(0,chosen[o['order_id']][1]-o['due_date_d']) for o in orders)+sigma*sum(nonreg.values()); s={'objective_value':z,'orders_solution':[{'order_id':j,'chosen_interval':[k,l],'intensities':{str(t):y for t,y in ys.items()}} for j,(k,l,ys) in chosen.items()],'nonregular_capacity_by_period':nonreg}; json.dump(s,open(a.solution_path,'w')); logger and logger.log_solution(z,s); return z
 for o in sorted(orders,key=lambda o:(o['due_date_d'],-o['tardiness_cost_w'])): place(o)
 best=save(); stale=0
 while time.monotonic()<end and stale<3:
  for o in rng.sample(orders,len(orders)):
   j=o['order_id']; k,l,ys=chosen[j]
   for t,y in ys.items(): usage[t]-=o['work_content_p']*y
   place(o)
   if time.monotonic()>=end: break
  z=save(); stale=stale+1 if z>=best-1e-8 else 0; best=min(best,z)
 if time.monotonic()>=end-2: return
 model=gp.Model(); model.Params.Threads=1; model.Params.Seed=0; model.Params.MIPGap=1e-4; model.Params.NumericFocus=0
 starts={}; finishes={}; intensity={}; overtime=model.addVars(range(1,H+1),obj=sigma)
 for o in orders:
  j=o['order_id']; r=o['release_date_r']; minlen=math.ceil(1/o['intensity_upper_bound_UB']-1e-8); k,l,ys=chosen[j]
  for t in range(r,H-minlen+2): starts[j,t]=model.addVar(vtype=GRB.BINARY); starts[j,t].Start=int(t==k)
  for t in range(r+minlen-1,H+1): finishes[j,t]=model.addVar(vtype=GRB.BINARY,obj=o['tardiness_cost_w']*max(0,t-o['due_date_d'])); finishes[j,t].Start=int(t==l)
  for t in range(r,H+1): intensity[j,t]=model.addVar(); intensity[j,t].Start=ys.get(t,0.)
  model.addConstr(gp.quicksum(v for (jj,t),v in starts.items() if jj==j)==1); model.addConstr(gp.quicksum(v for (jj,t),v in finishes.items() if jj==j)==1); model.addConstr(gp.quicksum(intensity[j,t] for t in range(r,H+1))==1)
  active=gp.LinExpr()
  for t in range(r,H+1):
   if (j,t) in starts: active+=starts[j,t]
   if (j,t-1) in finishes: active-=finishes[j,t-1]
   model.addConstr(intensity[j,t]>=o['intensity_lower_bound_LB']*active); model.addConstr(intensity[j,t]<=o['intensity_upper_bound_UB']*active)
 for t in range(1,H+1): model.addConstr(gp.quicksum(o['work_content_p']*intensity[o['order_id'],t] for o in orders if (o['order_id'],t) in intensity)<=cap[t]+overtime[t]); overtime[t].Start=max(0,usage[t]-cap[t])
 model.update(); assert model.NumVars<=2000 and model.NumConstrs<=2000
 variables=list(starts.values())+list(finishes.values())+list(intensity.values())
 def emit(values):
  nonlocal best
  sv=dict(zip(starts,values[:len(starts)])); fv=dict(zip(finishes,values[len(starts):len(starts)+len(finishes)])); iv=dict(zip(intensity,values[len(starts)+len(finishes):])); result=[]; use=[0.]*(H+1); z=0.
  for o in orders:
   j=o['order_id']; k=max((t for jj,t in starts if jj==j),key=lambda t:sv[j,t]); l=max((t for jj,t in finishes if jj==j),key=lambda t:fv[j,t]); ys={str(t):max(0.,iv[j,t]) for t in range(k,l+1)}; result.append({'order_id':j,'chosen_interval':[k,l],'intensities':ys}); z+=o['tardiness_cost_w']*max(0,l-o['due_date_d'])
   for t,y in ys.items(): use[int(t)]+=o['work_content_p']*y
  nr={str(t):max(0,use[t]-cap[t]) for t in range(1,H+1)}; z+=sigma*sum(nr.values())
  if z>=best-1e-7: return
  best=z; s={'objective_value':z,'orders_solution':result,'nonregular_capacity_by_period':nr}; json.dump(s,open(a.solution_path,'w')); logger and logger.log_solution(z,s)
 def cb(m,where):
  if where==GRB.Callback.MIPSOL: emit(m.cbGetSolution(variables))
 model.Params.TimeLimit=max(.1,end-time.monotonic()-2); model.optimize(cb)
 if model.SolCount: emit([v.X for v in variables])
if __name__=='__main__': main()

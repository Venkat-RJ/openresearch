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
a=p.parse_args(); end=time.monotonic()+a.time_limit-1; d=json.load(open(a.instance_path)); logger=configured_logger(a.log_path,sense='minimize'); vs=d['vessels']; n=len(vs); B=d['num_berths']; K=d['num_subblocks']; H=d['H']; E=d['E']; co=d['penalty_cost_coefficient_co']; DL=d['loading_route_lengths_D_L']; DU=d['average_unloading_route_lengths_D_U']; best=float('inf')
def model():
 m=gp.Model(); m.Params.OutputFlag=0; m.Params.Threads=1; m.Params.Seed=0; m.Params.MIPGap=1e-4; m.Params.NumericFocus=0; return m
m=model(); opts=[]; z=[]
for i,v in enumerate(vs):
 for b in range(B):
  for pi,pr in enumerate(v['qc_profiles']):
   h=pr['handling_time']
   for t in range(max(1,v['feasible_time_window'][0]),min(H,v['feasible_time_window'][1]-h+1)+1):
    penalty=v['penalty_cost_coefficient']*(max(0,v['expected_time_window'][0]-t)+max(0,t+h-1-v['expected_time_window'][1]))
    opts.append((i,b,pi,t,t+h-1)); z.append(m.addVar(vtype=GRB.BINARY,obj=penalty+co*DU[b]*v['unloading_containers']))
w={(i,b):m.addVar(vtype=GRB.BINARY) for i in range(n) for b in range(B)}
for i in range(n):
 m.addConstr(gp.quicksum(z[j] for j,o in enumerate(opts) if o[0]==i)==1)
 for b in range(B): m.addConstr(w[i,b]==gp.quicksum(z[j] for j,o in enumerate(opts) if o[:2]==(i,b)))
for b in range(B):
 rho=m.addVar(lb=1,ub=H+E); last=m.addVar(lb=1,ub=H+E); m.addConstr(last-rho<=H-1)
 for j,(i,bb,pi,t,finish) in enumerate(opts):
  if bb==b: m.addConstr(rho<=t+(H+E)*(1-z[j])); m.addConstr(last>=finish*z[j])
 for t in range(1,H+E+1): m.addConstr(gp.quicksum(z[j] for j,o in enumerate(opts) if o[1]==b and o[3]<=t<=o[4])<=1)
for t in range(1,H+1):
 terms=[]
 for j,(i,b,pi,start,finish) in enumerate(opts):
  q=vs[i]['qc_profiles'][pi]['qc_per_timestep']; val=(q[t-start] if start<=t<=finish else 0)+(q[t+H-start] if t<=E and start<=t+H<=finish else 0)
  if val: terms.append(val*z[j])
 m.addConstr(gp.quicksum(terms)<=d['QC_capacity_per_timestep'][t-1])
eta=m.addVar(lb=0,obj=1); m.update(); assert m.NumVars<=2000 and m.NumConstrs<=2000
yard=model(); x={(i,k):yard.addVar(ub=1) for i in range(n) for k in range(K)}
for i,v in enumerate(vs): yard.addConstr(gp.quicksum(x[i,k] for k in range(K))==v['reserved_subblocks'])
cols=[yard.addConstr(gp.quicksum(x[i,k] for i in range(n))<=1) for k in range(K)]; yard.update(); assert yard.NumVars<=2000
cost={(i,b,k):co*DL[k][b]*v['loading_containers']/v['reserved_subblocks'] for i,v in enumerate(vs) for b in range(B) for k in range(K)}
seen=set()
while time.monotonic()<end:
 m.Params.TimeLimit=max(.1,end-time.monotonic()); m.optimize()
 if not m.SolCount: break
 chosen=[o for j,o in enumerate(opts) if z[j].X>.5]; berths={i:b for i,b,pi,t,f in chosen}; key=tuple(berths[i] for i in range(n))
 for (i,k),var in x.items(): var.Obj=cost[i,berths[i],k]
 yard.Params.TimeLimit=max(.1,end-time.monotonic()); yard.optimize()
 if yard.Status!=GRB.OPTIMAL: break
 out=[]; usage={str(t):0 for t in range(1,H+E+1)}; obj=yard.ObjVal
 for i,b,pi,t,finish in chosen:
  v=vs[i]; pr=v['qc_profiles'][pi]; blocks=[k for k in range(K) if x[i,k].X>.5]; assert len(blocks)==v['reserved_subblocks']
  out.append({'vessel_id':v['vessel_id'],'alpha':t,'beta':finish,'berth':b,'profile':pi,'subblocks':blocks,'handling_time':pr['handling_time'],'qc_per_timestep':pr['qc_per_timestep']})
  obj+=v['penalty_cost_coefficient']*(max(0,v['expected_time_window'][0]-t)+max(0,finish-v['expected_time_window'][1]))+co*DU[b]*v['unloading_containers']
  for offset,q in enumerate(pr['qc_per_timestep']): usage[str(t+offset)]+=q
 if obj<best-1e-8:
  best=obj; sol={'objective_value':obj,'vessels':out,'qc_usage':usage,'berths':[{'berth':b,'rho':min((o['alpha'] for o in out if o['berth']==b),default=1),'varsigma':max((o['beta'] for o in out if o['berth']==b),default=1)} for b in range(B)]}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(obj,sol)
 if best<=m.ObjBound+max(1e-6,abs(best)*1e-4) or key in seen: break
 seen.add(key); prices=[c.Pi for c in cols]
 coeff={(i,b):sum(sorted(cost[i,b,k]-prices[k] for k in range(K))[:v['reserved_subblocks']]) for i,v in enumerate(vs) for b in range(B)}
 m.addConstr(eta>=sum(prices)+gp.quicksum(coeff[i,b]*w[i,b] for i in range(n) for b in range(B)))

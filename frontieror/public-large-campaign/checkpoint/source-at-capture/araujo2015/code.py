import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
_PROCESS_START=time.monotonic()
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','log_path'):p.add_argument('--'+k,required=True)
 p.add_argument('--time_limit',type=int,default=60);a=p.parse_args();start=_PROCESS_START;deadline=start+max(0.,a.time_limit-5.);data=json.load(open(a.instance_path));n=data['num_items'];T=data['num_periods'];d=data['demand'];logger=SolutionLogger(a.log_path,sense='minimize')
 def expand(key):return [v if isinstance(v,list) else [v]*T for v in data[key]]
 sc=expand('setup_cost');hc=expand('holding_cost');vc=expand('variable_cost');st=expand('setup_time');vt=expand('variable_time');fc=data['initial_inventory_cost'];cap=data['capacity'];best=float('inf');best_witness=None
 def save(x,y,s):
  nonlocal best,best_witness
  obj=sum(fc[i]*s[i][0]+sum(sc[i][t]*y[i][t]+vc[i][t]*x[i][t]+hc[i][t]*s[i][t+1] for t in range(T)) for i in range(n))
  if obj>=best:return
  best=obj;best_witness=(x,y,s);sol={'objective_value':obj,'production':x,'setup':y,'inventory':s};json.dump(sol,open(a.solution_path,'w'))
  if hasattr(logger,'log_solution'):logger.log_solution(obj,sol)
  else:logger.log(obj)
 save([[0.]*T for _ in range(n)],[[0]*T for _ in range(n)],[[float(sum(d[i][t:])) for t in range(T+1)] for i in range(n)])
 integer_data=[v for rows in (d,sc,hc,vc,st) for row in rows for v in row]+list(fc)+list(cap)
 if all(v==int(v) for v in integer_data) and all(v==1 for row in vt for v in row):
  from ortools.sat.python import cp_model
  model=cp_model.CpModel()
  cx=[];cy=[];cs=[]
  for i in range(n):
   remaining_demand=[int(sum(d[i][t:])) for t in range(T+1)]
   cx.append([model.NewIntVar(0,max(0,min(remaining_demand[t],int(cap[t]-st[i][t]))),'x_'+str(i)+'_'+str(t)) for t in range(T)])
   cy.append([model.NewBoolVar('y_'+str(i)+'_'+str(t)) for t in range(T)])
   cs.append([model.NewIntVar(0,remaining_demand[t],'s_'+str(i)+'_'+str(t)) for t in range(T+1)])
   for t in range(T):
    model.Add(cs[i][t]+cx[i][t]-cs[i][t+1]==int(d[i][t]))
    model.Add(cx[i][t]<=max(0,min(remaining_demand[t],int(cap[t]-st[i][t])))*cy[i][t])
    model.AddHint(cx[i][t],0);model.AddHint(cy[i][t],0)
   for t in range(T+1):model.AddHint(cs[i][t],remaining_demand[t])
  for t in range(T):model.Add(sum(int(st[i][t])*cy[i][t]+cx[i][t] for i in range(n))<=int(cap[t]))
  model.Minimize(sum(int(fc[i])*cs[i][0]+sum(int(sc[i][t])*cy[i][t]+int(vc[i][t])*cx[i][t]+int(hc[i][t])*cs[i][t+1] for t in range(T)) for i in range(n)))
  class Callback(cp_model.CpSolverSolutionCallback):
   def on_solution_callback(self):
    save([[self.Value(v) for v in row] for row in cx],[[self.Value(v) for v in row] for row in cy],[[self.Value(v) for v in row] for row in cs])
  solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
  solver.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic())
  print('CP_BUILD_END elapsed',time.monotonic()-start,'solve_budget',solver.parameters.max_time_in_seconds,flush=True)
  status=solver.Solve(model,Callback())
  if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):save([[solver.Value(v) for v in row] for row in cx],[[solver.Value(v) for v in row] for row in cy],[[solver.Value(v) for v in row] for row in cs])
  print('CP_SOLVE_END elapsed',time.monotonic()-start,'status',solver.StatusName(status),'best',best,'bound',solver.BestObjectiveBound(),flush=True)
  return
 import highspy
 print('BUILD_START elapsed',time.monotonic()-start,'items',n,'periods',T,'baseline',best,flush=True)
 h=highspy.Highs();h.setOptionValue('threads',1);h.setOptionValue('time_limit',float(a.time_limit));h.setOptionValue('output_flag',False)
 x=[[h.addVariable(lb=0,obj=vc[i][t]) for t in range(T)] for i in range(n)]
 y=[[h.addVariable(lb=0,ub=1,obj=sc[i][t],type=highspy.HighsVarType.kInteger) for t in range(T)] for i in range(n)]
 s=[[h.addVariable(lb=0,ub=0 if t==T else highspy.kHighsInf,obj=fc[i] if t==0 else hc[i][t-1]) for t in range(T+1)] for i in range(n)]
 for i in range(n):
  if time.monotonic()>=deadline:
   print('MODEL_BUILD_DEADLINE elapsed',time.monotonic()-start,'items_built',i,flush=True);return
  for t in range(T):
   h.addConstr(s[i][t]+x[i][t]-s[i][t+1]==d[i][t]);ub=max(0,min(sum(d[i][t:]),(cap[t]-st[i][t])/vt[i][t] if vt[i][t]>0 else sum(d[i][t:])));h.addConstr(x[i][t]<=ub*y[i][t])
 for t in range(T):h.addConstr(sum(st[i][t]*y[i][t]+vt[i][t]*x[i][t] for i in range(n))<=cap[t])
 print('BUILD_END elapsed',time.monotonic()-start,flush=True)
 px=[[0.]*T for _ in range(n)];py=[[0]*T for _ in range(n)];unfilled=[[float(v) for v in row] for row in d];available=list(cap)
 holding=[[0.] for _ in range(n)]
 for i in range(n):
  for cost in hc[i]:holding[i].append(holding[i][-1]+cost)
 candidates=[]
 for i in range(n):
  for due in range(T):
   for t in range(due+1):
    saving=fc[i]+holding[i][due]-vc[i][t]-(holding[i][due]-holding[i][t])
    if saving>0:candidates.append((saving,i,due,t))
 for saving,i,due,t in sorted(candidates,reverse=True):
  if time.monotonic()>=deadline:break
  setup=0. if py[i][t] else st[i][t]
  room=available[t]-setup
  if room<=0 or unfilled[i][due]<=0:continue
  quantity=min(unfilled[i][due],room/vt[i][t] if vt[i][t]>0 else unfilled[i][due])
  if not py[i][t] and quantity*saving<=sc[i][t]:continue
  px[i][t]+=quantity;unfilled[i][due]-=quantity;py[i][t]=1;available[t]-=setup+quantity*vt[i][t]
 ps=[]
 for i in range(n):
  row=[sum(unfilled[i])]
  for t in range(T):row.append(row[-1]+px[i][t]-d[i][t])
  ps.append([max(0.,v) for v in row])
 save(px,py,ps)
 def optimize(label,budget):
  remaining=deadline-time.monotonic()
  if remaining<=0:return
  indices=[];values=[]
  for variables,witness in zip((x,y,s),best_witness):
   for row,wr in zip(variables,witness):
    for variable,value in zip(row,wr):indices.append(int(variable));values.append(value)
  h.setSolution(len(indices),indices,values)
  h.setOptionValue('time_limit',min(budget,remaining));h.run()
  if h.getSolution().value_valid:
   vals=h.getSolution().col_value
   save([[max(0,vals[int(v)]) for v in row] for row in x],[[round(vals[int(v)]) for v in row] for row in y],[[max(0,vals[int(v)]) for v in row] for row in s])
  print('OPTIMIZE_PHASE',label,'elapsed',time.monotonic()-start,'best',best,'primal',h.getInfo().objective_function_value,'dual',h.getInfo().mip_dual_bound,'nodes',h.getInfo().mip_node_count,flush=True)
 remaining=deadline-time.monotonic()
 if n*T<200:
  optimize('FULL',remaining);return
 construction_deadline=min(deadline-10.,time.monotonic()+30.)
 fixed=0
 for first in range(0,T,5):
  available=construction_deadline-time.monotonic()
  if available<=0:break
  last=min(T,first+15)
  for i in range(n):
   for t in range(first,T):h.changeColIntegrality(int(y[i][t]),highspy.HighsVarType.kInteger if t<last else highspy.HighsVarType.kContinuous)
  h.setOptionValue('time_limit',min(2.4,available));h.run()
  if not h.getSolution().value_valid:break
  vals=h.getSolution().col_value
  if any(abs(vals[int(y[i][t])]-round(vals[int(y[i][t])]))>1e-6 for i in range(n) for t in range(first,last)):break
  fixed=min(T,first+5)
  for i in range(n):
   for t in range(first,fixed):
    value=round(vals[int(y[i][t])]);h.changeColBounds(int(y[i][t]),value,value)
  print('RELAX_FIX first',first,'fixed',fixed,'integer_until',last,'relaxed_objective',h.getInfo().objective_function_value,'initial_cost',sum(fc[i]*vals[int(s[i][0])] for i in range(n)),'elapsed',time.monotonic()-start,flush=True)
  if last==T:
   save([[max(0.,vals[int(v)]) for v in row] for row in x],[[round(vals[int(v)]) for v in row] for row in y],[[max(0.,vals[int(v)]) for v in row] for row in s])
 for row in y:
  for variable in row:
   h.changeColBounds(int(variable),0.,1.);h.changeColIntegrality(int(variable),highspy.HighsVarType.kInteger)
 optimize('FINAL_FULL',deadline-time.monotonic())
if __name__=='__main__':main()

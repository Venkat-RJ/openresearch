import argparse,json,os,time,math,itertools
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));p=d['global_parameters'];orders=d['orders'];n=d['num_orders'];P=d['num_products'];T=p['num_periods'];U=p['num_subperiods'];sp=p['subperiods_per_period'];sigma=p['setup_length_subperiods'];periods=d['periods'];sub=d['subperiods'];travel=d['travel_time_matrix'];windows=d['production_windows'];valid={(i,int(k)):v['valid_subperiod_ids'] for i,win in ((i,windows[str(i)]) for i in range(n)) for k,v in win.items()};pu=sorted({(prod,u) for (i,prod),us in valid.items() for u in us});NE=d['num_production_employees'];NW=d['num_workstations'];Q=d['vehicle_capacity'];trips={}
 for t in range(T):
  depart=periods[t]['start_time']
  def dfs(seq,remaining,load,clock,dist,end):
   if seq:
    mask=sum(1<<i for i in seq);key=(mask,t);record=(tuple(seq),t,end,dist+travel[seq[-1]+1][0]);old=trips.get(key)
    if old is None or record[3]+max(end-depart,p['min_paid_time_minutes'])<old[3]+max(old[2]-depart,p['min_paid_time_minutes']):trips[key]=record
   for i in remaining:
    if load+orders[i]['space_requirement']>Q:continue
    if any(not any(sub[u]['end_time']<=depart for u in valid[i,prod]) for prod in [int(k) for k,v in orders[i]['demands'].items() if v>0]):continue
    prev=seq[-1]+1 if seq else 0;arr=max(clock+travel[prev][i+1],orders[i]['time_window_start'])
    if arr>orders[i]['time_window_end']+1e-8:continue
    ee=max(end,arr+p['service_time_minutes']+travel[i+1][0])
    if ee-depart>p['max_shift_route_length_minutes']:continue
    dfs(seq+[i],[j for j in remaining if j!=i],load+orders[i]['space_requirement'],arr+p['service_time_minutes'],dist+travel[prev][i+1],ee)
  dfs([],list(range(n)),0,depart+p['preparation_time_minutes'],0,depart)
 grouped={}
 def add(tt):
  mask=sum(1<<i for r in tt for i in r[0]);depart=min(periods[r[1]]['start_time'] for r in tt);end=max(r[2] for r in tt);duration=end-depart
  if duration>p['max_shift_route_length_minutes']:return
  cost=p['driver_salary_per_minute']*max(duration,p['min_paid_time_minutes'])+p['vehicle_operating_cost_per_minute']*sum(r[3] for r in tt);lst=grouped.setdefault(mask,[]);key=(cost,-sum(r[1] for r in tt),tt)
  if key not in lst:lst.append(key);lst.sort();del lst[4:]
 records=list(trips.values())
 for r in records:add((r,))
 if p['max_trips_per_route']>=2:
  for r1,r2 in itertools.combinations(records,2):
   if r1[1]==r2[1] or set(r1[0]).intersection(r2[0]):continue
   add(tuple(sorted((r1,r2),key=lambda r:r[1])))
 columns=[v for lst in grouped.values() for v in lst];model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.NumericFocus=0;model.Params.MIPGap=1e-4;theta=model.addVars(len(columns),vtype=GRB.BINARY);q=model.addVars([(i,prod,u) for (i,prod),us in valid.items() for u in us],lb=0);f=model.addVars(pu,vtype=GRB.INTEGER,lb=0,ub=NE);beta=model.addVars(pu,vtype=GRB.INTEGER,lb=0,ub=NE);w=model.addVars(T,vtype=GRB.INTEGER,lb=0,ub=NW);alpha=model.addVars(T,vtype=GRB.INTEGER,lb=0,ub=NE);serving={i:[h for h,(_,_,tt) in enumerate(columns) if any(i in r[0] for r in tt)] for i in range(n)}
 for i in range(n):model.addConstr(gp.quicksum(theta[h] for h in serving[i])==1)
 model.addConstr(theta.sum()<=min(d['num_vehicles'],d['num_drivers']))
 for (i,prod),us in valid.items():
  dem=orders[i]['demands'][str(prod)];model.addConstr(gp.quicksum(q[i,prod,u] for u in us)==dem)
  for t in range(T):
   cols=[h for h in serving[i] if any(i in r[0] and r[1]==t for r in columns[h][2])]
   if cols:model.addConstr(gp.quicksum(q[i,prod,u] for u in us if sub[u]['end_time']<=periods[t]['start_time'])>=dem*gp.quicksum(theta[h] for h in cols))
 for prod,u in pu:
  model.addConstr(gp.quicksum(q[i,prod,u] for i in range(n) if (i,prod,u) in q)<=d['products'][prod]['production_rate_per_subperiod']*f[prod,u]);model.addConstr(f[prod,u]<=f.get((prod,u-1),0)+beta.get((prod,u-sigma),0))
 for u in range(U):
  active=gp.quicksum(f.get((prod,u),0)+gp.quicksum(beta.get((prod,j),0) for j in range(max(0,u-sigma+1),u+1)) for prod in range(P));model.addConstr(active<=NE);model.addConstr(active<=w[u//sp])
 paid=p['min_paid_time_subperiods']//sp
 for t in range(T):model.addConstr(alpha[t]>=w[t]-(w[t-1] if t else 0));model.addConstr(w[t]>=gp.quicksum(alpha[t-j] for j in range(1,min(t+1,paid)+1) if t-j>=0))
 model.addConstr(alpha.sum()<=NE);model.setObjective(gp.quicksum(columns[h][0]*theta[h] for h in range(len(columns)))+p['employee_salary_per_subperiod']*sp*w.sum()+p['setup_cost']*beta.sum());logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  chosen=[h for h in range(len(columns)) if get(theta[h])>.5];routes=[{'vehicle':v,'trips':[{'trip':k,'customers':list(r[0]),'start_period':r[1]} for k,r in enumerate(columns[h][2])]} for v,h in enumerate(chosen)];production=[{'order':i,'product':prod,'subperiod':u,'quantity':float(get(var))} for (i,prod,u),var in q.items() if get(var)>1e-9];wv=[int(round(get(w[t]))) for t in range(T)];bv=[{'product':prod,'subperiod':u,'setups':int(round(get(beta[prod,u])))} for prod,u in pu];staff={'employees_by_product_subperiod':[{'product':prod,'subperiod':u,'employees':int(round(get(f[prod,u])))} for prod,u in pu],'setups_by_product_subperiod':bv,'workstations_by_period':wv,'shift_starts_by_period':[int(round(get(alpha[t]))) for t in range(T)]};obj=sum(columns[h][0] for h in chosen)+p['employee_salary_per_subperiod']*sp*sum(wv)+p['setup_cost']*sum(v['setups'] for v in bv);return {'objective_value':obj,'routes':routes,'production_schedule':production,'production_staffing':staff}
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   s=extract(md.cbGetSolution);json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(s['objective_value'],s)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No integrated caterer route-column incumbent')
if __name__=='__main__':main()

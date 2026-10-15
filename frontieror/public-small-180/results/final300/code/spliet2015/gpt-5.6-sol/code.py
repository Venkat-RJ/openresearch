import argparse,json,time,random,os
wall_started=time.time()
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger

def main():
 a=argparse.ArgumentParser(); a.add_argument('--instance_path'); a.add_argument('--solution_path'); a.add_argument('--time_limit',type=int,default=60); a.add_argument('--log_path'); a=a.parse_args(); start=time.monotonic(); end=start+max(.1,a.time_limit-2); d=json.load(open(a.instance_path)); n=d['num_customers']; customers=d['customers']; ids=[c['id'] for c in customers]; dist=d['distance_matrix']; Q=d['vehicle_capacity']; scen=d['scenarios']; depot=d['depot']['exogenous_time_window']; widths={c['id']:c['endogenous_time_window_width'] for c in customers}; tw={c['id']:c['exogenous_time_window'] for c in customers}; rng=random.Random(0); logger=configured_logger(a.log_path,sense='minimize') if a.log_path else None; best=float('inf')
 def schedule(route,windows):
  times={'0':depot[0]}; t=depot[0]; prev=0
  for v in route:
   t=max(t+dist[prev][v],windows[v][0]);
   if t>windows[v][1]+1e-8: return None
   times[str(v)]=t; prev=v
  t+=dist[prev][n+1]
  if t>depot[1]+1e-8: return None
  times[str(n+1)]=t; return times
 def cost(route): return sum(dist[u][v] for u,v in zip([0]+route,route+[n+1]))
 def build(dem,windows,order):
  routes=[]
  for v in order:
   opts=[]
   for ri,r in enumerate(routes):
    if sum(dem[x] for x in r)+dem[v]>Q: continue
    old=cost(r)
    for k in range(len(r)+1):
     new=r[:k]+[v]+r[k:]
     if schedule(new,windows) is not None: opts.append((cost(new)-old,ri,new))
   if opts: _,ri,new=min(opts); routes[ri]=new
   elif schedule([v],windows) is not None: routes.append([v])
   else: return None
  return routes
 def emit(windows,routes):
  nonlocal best
  z=sum(s['probability']*sum(cost(r) for r in rs) for s,rs in zip(scen,routes))
  if z<best:
   best=z; out={'objective_value':z,'time_windows':{str(v):windows[v] for v in ids},'routes':{str(s['scenario_id']):[{'route':[0]+r+[n+1],'arrival_times':schedule(r,windows),'cost':cost(r)} for r in rs] for s,rs in zip(scen,routes)}}; json.dump(out,open(a.solution_path,'w')); logger and logger.log_solution(z,out)
 demmax={v:max(s['demands'][i] for s in scen) for i,v in enumerate(ids)}
 iteration=0
 while time.monotonic()<end:
  order=sorted(ids,key=lambda v:(tw[v][1],-dist[0][v])) if iteration==0 else rng.sample(ids,len(ids)); common=build(demmax,tw,order)
  if common:
   windows={}
   for r in common:
    ts=schedule(r,tw)
    for v in r:
     y=max(tw[v][0],min(tw[v][1]-widths[v],ts[str(v)]-widths[v]/2)); windows[v]=[y,y+widths[v]]
   routes=[]
   for s in scen:
    dem={v:s['demands'][i] for i,v in enumerate(ids)}; rs=build(dem,windows,order); routes.append(rs)
   if all(rs is not None for rs in routes): emit(windows,routes)
  iteration+=1
  if n<=10 and iteration>=100: break
 if n<=12 and time.monotonic()<end:
  if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']): del os.environ['GRB_LICENSE_FILE']
  import gurobipy as gp
  model=gp.Model(); model.Params.OutputFlag=0; model.Params.Seed=0; model.Params.MIPGap=1e-4; model.Params.NumericFocus=0; model.Params.Threads=1; model.Params.TimeLimit=max(.1,end-time.monotonic()); arcs=[(i,j) for i in range(n+1) for j in range(1,n+2) if i!=j and (i,j)!=(0,n+1)]; windowsvars={v:model.addVar(lb=tw[v][0],ub=tw[v][1]-widths[v]) for v in ids}; xs={}; ts={}; us={}
  for si,s in enumerate(scen):
   for i in range(n+2): ts[si,i]=model.addVar(lb=depot[0] if i in [0,n+1] else tw[i][0],ub=depot[1] if i in [0,n+1] else tw[i][1])
   for v in ids: us[si,v]=model.addVar(lb=s['demands'][ids.index(v)],ub=Q)
   for i,j in arcs: xs[si,i,j]=model.addVar(vtype=gp.GRB.BINARY,obj=s['probability']*dist[i][j])
   for v in ids:
    model.addConstr(gp.quicksum(xs[si,i,j] for i,j in arcs if j==v)==1); model.addConstr(gp.quicksum(xs[si,i,j] for i,j in arcs if i==v)==1); model.addConstr(ts[si,v]>=windowsvars[v]); model.addConstr(ts[si,v]<=windowsvars[v]+widths[v])
   for i,j in arcs:
    model.addConstr(ts[si,j]>=ts[si,i]+dist[i][j]-100*(1-xs[si,i,j]))
    if i in ids and j in ids: model.addConstr(us[si,j]>=us[si,i]+s['demands'][ids.index(j)]-Q*(1-xs[si,i,j]))
  model.update()
  if model.NumVars<=2000 and model.NumConstrs<=2000:
   variables=list(windowsvars.values())+list(xs.values())
   def extract(values):
    wv=dict(zip(windowsvars,values[:len(windowsvars)])); xv=dict(zip(xs,values[len(windowsvars):])); windows={v:[wv[v],wv[v]+widths[v]] for v in ids}; routes=[]
    for si,s in enumerate(scen):
     rs=[]; succ={i:j for i,j in arcs if i!=0 and xv[si,i,j]>.5}
     for i,j in arcs:
      if i==0 and xv[si,i,j]>.5:
       r=[]; v=j
       while v!=n+1: r.append(v); v=succ[v]
       rs.append(r)
     routes.append(rs)
    emit(windows,routes)
   def cb(m,where):
    if where==gp.GRB.Callback.MIPSOL: extract(m.cbGetSolution(variables))
   model.optimize(cb)
   if model.SolCount: extract([v.X for v in variables])
if __name__=='__main__': main()

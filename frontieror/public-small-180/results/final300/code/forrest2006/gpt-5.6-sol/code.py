import argparse,json,time,os,collections
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-4)
 z=json.load(open(a.instance_path));n,m=z['num_orders'],z['num_slabs'];w=z['order_weights'];W=z['slab_weights'];color=z['order_colors'];K=z['max_colors_per_knapsack'];eligible=z['eligible_slabs_per_order'];logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None;bestval=-1e100
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 def save(t):
  nonlocal bestval
  used=sorted(set(t.values()));val=sum(2*w[i] for i in t)-sum(W[j] for j in used)
  if val>bestval+1e-8:
   bestval=val;sol={'objective_value':val,'assignments':[{'order':i,'slab':j} for i,j in t.items()],'slabs_used':used,'runtime':time.monotonic()-start};json.dump(sol,open(a.solution_path,'w'))
   if logger:logger.log_solution(val,sol)
 save({});incident=[[] for j in range(m)]
 for i in range(n):
  for j in eligible[i]:
   if w[i]<=W[j]+1e-8:incident[j].append(i)
 if n<=50:
  lic=os.environ.get('GRB_LICENSE_FILE')
  if lic and not os.path.exists(lic):os.environ.pop('GRB_LICENSE_FILE')
  import gurobipy as gp
  md=gp.Model();md.Params.Seed=0;md.Params.MIPGap=1e-4;md.Params.NumericFocus=0;md.Params.Threads=1;md.Params.TimeLimit=max(.1,end-time.monotonic());md.Params.OutputFlag=0;pairs=[(i,j) for j in range(m) for i in incident[j]];x=md.addVars(pairs,vtype=gp.GRB.BINARY);y=md.addVars([(c,j) for j in range(m) for c in set(color[i] for i in incident[j])],vtype=gp.GRB.BINARY);opened=md.addVars(m,vtype=gp.GRB.BINARY)
  for i in range(n):md.addConstr(gp.quicksum(x[i,j] for j in eligible[i] if (i,j) in x)<=1)
  for j in range(m):md.addConstr(gp.quicksum(w[i]*x[i,j] for i in incident[j])<=W[j]*opened[j]);md.addConstr(gp.quicksum(y[c,j] for c in set(color[i] for i in incident[j]))<=K)
  for i,j in pairs:md.addConstr(x[i,j]<=y[color[i],j])
  md.setObjective(gp.quicksum(2*w[i]*x[i,j] for i,j in pairs)-gp.quicksum(W[j]*opened[j] for j in range(m)),gp.GRB.MAXIMIZE)
  def cb(model,where):
   if where==gp.GRB.Callback.MIPSOL:
    vals=model.cbGetSolution(x);save({i:j for i,j in pairs if vals[i,j]>.5})
  md.optimize(cb);return
 assignment={}
 for j in sorted(range(m),key=lambda j:W[j]):
  if time.monotonic()>=end:break
  bycolor=collections.defaultdict(list)
  for i in incident[j]:
   if i not in assignment:bycolor[color[i]].append(i)
  chosen=sorted(bycolor,key=lambda c:sum(w[i] for i in bycolor[c]),reverse=True)[:K];items=sorted([i for c in chosen for i in bycolor[c]],key=lambda i:w[i],reverse=True);packed=[];load=0
  for i in items:
   if load+w[i]<=W[j]+1e-9:packed.append(i);load+=w[i]
  if 2*load>W[j]:
   for i in packed:assignment[i]=j
   save(assignment)
if __name__=='__main__':main()

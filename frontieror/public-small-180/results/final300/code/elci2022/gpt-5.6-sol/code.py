import argparse,json,os,time,math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();begin=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));n=d['n'];m=d['m'];S=d['num_scenarios'];K=d['facility_capacity_limits'];c=d['capacity_requirements'];release=d['release_times'];p=[s['processing_times'] for s in d['scenarios']];prob=[s['probability'] for s in d['scenarios']];generic=max(release)+sum(max(p[w][i][j] for w in range(S) for i in range(m)) for j in range(n));assigned={j:min((i for i in range(m) if c[i][j]<=K[i]),key=lambda i:sum(prob[w]*p[w][i][j] for w in range(S))) for j in range(n)};greedy={};ub=0
 for w in range(S):
  occ=[[0]*(generic+1) for i in range(m)];sched={}
  for j in sorted(range(n),key=lambda j:(release[j],-p[w][assigned[j]][j])):
   i=assigned[j];dur=p[w][i][j];t=next(t for t in range(release[j],generic-dur+1) if all(occ[i][h]+c[i][j]<=K[i] for h in range(t,t+dur)));sched[str(j)]={'facility':i,'start':t,'finish':t+dur}
   for h in range(t,t+dur):occ[i][h]+=c[i][j]
  greedy[str(w)]=sched;ub+=prob[w]*max(v['finish'] for v in sched.values())
 logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 initial={'objective_value':ub,'assignment':{str(j):i for j,i in assigned.items()},'schedule':greedy};json.dump(initial,open(a.solution_path,'w'))
 if logger:logger.log_solution(ub,initial)
 H=[min(generic,math.floor((ub+1e-7)/prob[w])) if prob[w]>0 else generic for w in range(S)];model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.NumericFocus=0;model.Params.MIPGap=1e-4;x=model.addVars(m,n,vtype=GRB.BINARY);b=model.addVars(S,lb=0);keys=[(w,i,j,t) for w in range(S) for i in range(m) for j in range(n) if c[i][j]<=K[i] for t in range(release[j],H[w]-p[w][i][j]+1)];z=model.addVars(keys,vtype=GRB.BINARY)
 for j in range(n):model.addConstr(gp.quicksum(x[i,j] for i in range(m))==1)
 for w in range(S):
  for i in range(m):
   for j in range(n):
    ts=[t for t in range(release[j],H[w]-p[w][i][j]+1) if (w,i,j,t) in z];model.addConstr(gp.quicksum(z[w,i,j,t] for t in ts)==x[i,j]);model.addConstr(b[w]>=gp.quicksum((t+p[w][i][j])*z[w,i,j,t] for t in ts))
   for t in range(H[w]):model.addConstr(gp.quicksum(c[i][j]*z[w,i,j,u] for j in range(n) for u in range(max(release[j],t-p[w][i][j]+1),min(t,H[w]-p[w][i][j])+1) if (w,i,j,u) in z)<=K[i])
 model.setObjective(gp.quicksum(prob[w]*b[w] for w in range(S)));model.addConstr(gp.quicksum(prob[w]*b[w] for w in range(S))<=ub+1e-7)
 for i in range(m):
  for j in range(n):x[i,j].Start=int(assigned[j]==i)
 for key,var in z.items():w,i,j,t=key;var.Start=int(greedy[str(w)][str(j)]['facility']==i and greedy[str(w)][str(j)]['start']==t)
 def extract(get):
  assign={str(j):next(i for i in range(m) if get(x[i,j])>.5) for j in range(n)};sched={str(w):{} for w in range(S)}
  for (w,i,j,t),v in z.items():
   if get(v)>.5:sched[str(w)][str(j)]={'facility':i,'start':t,'finish':t+p[w][i][j]}
  obj=sum(prob[w]*max(v['finish'] for v in sched[str(w)].values()) for w in range(S));return {'objective_value':obj,'assignment':assign,'schedule':sched}
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   s=extract(md.cbGetSolution);json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(s['objective_value'],s)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-begin)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
if __name__=='__main__':main()

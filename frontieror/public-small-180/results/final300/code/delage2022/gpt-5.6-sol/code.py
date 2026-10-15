import argparse,json,os,time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));products=d['products'];n=d['num_products'];P=d['num_patterns'];M=d['M'];closed=d['problem_type']=='closed_ended_cscp';knife=d['problem_type']=='cscp_with_knife_dependent_setups';model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;s=model.addVars(P,vtype=GRB.BINARY);z=model.addVars(P,vtype=GRB.INTEGER,lb=0,ub=M);extra=model.addVars(n,lb=0,ub=0 if closed else GRB.INFINITY);bits={};prod={};pieces={}
 for i in range(n):
  for j in range(P):
   aa=gp.LinExpr();pp=gp.LinExpr()
   for b in range(products[i]['N_i'].bit_length()):
    bit=model.addVar(vtype=GRB.BINARY);u=model.addVar(lb=0,ub=M);bits[i,j,b]=bit;model.addConstr(u<=M*bit);model.addConstr(u<=z[j]);model.addConstr(u>=z[j]-M*(1-bit));aa+=(1<<b)*bit;pp+=(1<<b)*u
   pieces[i,j]=aa;prod[i,j]=pp
 for j in range(P):model.addConstr(gp.quicksum(products[i]['length']*pieces[i,j] for i in range(n))<=d['machine_length']*s[j]);model.addConstr(z[j]<=M*s[j])
 for j in range(1,P):model.addConstr(z[j-1]>=z[j])
 for i in range(n):model.addConstr(gp.quicksum(prod[i,j] for j in range(P))+extra[i]>=products[i]['demand'])
 model.addConstr(extra.sum()<=z.sum());model.setObjective(d['cost_setup_pattern']*s.sum()+d['cost_repetition']*z.sum()+(d.get('cost_knife_setup',0)*gp.quicksum(pieces[i,j] for i in range(n) for j in range(P)) if knife else 0));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  counts={(i,j):sum((1<<b)*round(get(bits[i,j,b])) for b in range(products[i]['N_i'].bit_length())) for i in range(n) for j in range(P)};reps=[int(round(get(z[j]))) for j in range(P)];used=[int(round(get(s[j]))) for j in range(P)];patterns={str(j):{'used':bool(used[j]),'repetitions':reps[j],'pieces':{str(products[i]['id']):counts[i,j] for i in range(n)}} for j in range(P)}
  if knife:
   for j in range(P):patterns[str(j)]['knife_count']=sum(counts[i,j] for i in range(n))
  obj=d['cost_setup_pattern']*sum(used)+d['cost_repetition']*sum(reps)+(d.get('cost_knife_setup',0)*sum(counts.values()) if knife else 0);return {'objective_value':obj,'patterns':patterns,'extra_pieces':{str(products[i]['id']):float(get(extra[i])) for i in range(n)},'production_quantities':{f'product_{products[i]["id"]}_pattern_{j}':counts[i,j]*reps[j] for i in range(n) for j in range(P)}}
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   result=extract(md.cbGetSolution);json.dump(result,open(a.solution_path,'w'))
   if logger:logger.log_solution(result['objective_value'],result)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No cutting configuration incumbent')
if __name__=='__main__':main()

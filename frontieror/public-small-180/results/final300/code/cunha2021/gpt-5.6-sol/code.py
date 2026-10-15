import argparse,json,os,time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));R=d['num_retailers'];W=d['num_warehouses'];T=d['num_periods'];N=1+W+R;demand=[[sum(d['demands'][r][t] for r in range(R)) for t in range(T)]]+[[sum(d['demands'][r][t] for r in d['warehouse_retailer_sets'][str(w)]) for t in range(T)] for w in range(W)]+d['demands'];fixed=[d['fixed_costs']['plant']]+d['fixed_costs']['warehouses']+d['fixed_costs']['retailers'];holding=[d['holding_costs']['plant']]+[d['holding_costs']['warehouses']]*W+d['holding_costs']['retailers'];children={0:list(range(1,W+1))};children.update({1+w:[1+W+r for r in d['warehouse_retailer_sets'][str(w)]] for w in range(W)})
 model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;x=model.addVars(N,T,lb=0);s={};y=model.addVars(N,T,vtype=GRB.BINARY)
 for i in range(N):
  for t in range(T):
   s[i,t]=(s[i,t-1] if t else gp.LinExpr(0))+x[i,t]-(gp.quicksum(x[j,t] for j in children[i]) if i in children else demand[i][t]);model.addConstr(s[i,t]>=0);model.addConstr(x[i,t]<=sum(demand[i][t:])*y[i,t])
 model.setObjective(gp.quicksum(fixed[i][t]*y[i,t]+holding[i]*s[i,t] for i in range(N) for t in range(T)));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  original=get
  def get(v):
   if isinstance(v,gp.LinExpr):return v.getConstant()+sum(v.getCoeff(k)*original(v.getVar(k)) for k in range(v.size()))
   return original(v)
  result={};obj=0
  for name,vars in [('x',x),('s',s),('y',y)]:
   values=[[int(round(get(vars[i,t]))) if name=='y' else max(0,float(get(vars[i,t]))) for t in range(T)] for i in range(N)];result[name+'_plant']=values[0];result[name+'_warehouses']=values[1:1+W];result[name+'_retailers']=values[1+W:]
  obj=sum(fixed[i][t]*round(get(y[i,t]))+holding[i]*get(s[i,t]) for i in range(N) for t in range(T));result['objective_value']=obj;return result
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   result=extract(md.cbGetSolution);json.dump(result,open(a.solution_path,'w'))
   if logger:logger.log_solution(result['objective_value'],result)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No three-echelon incumbent')
if __name__=='__main__':main()

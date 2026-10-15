import argparse,json,os,time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));W=d['bin_capacity'];items=sorted(d['items'],key=lambda x:-x['width']);B=[];A=set();vertices={0,W}
 for i,it in enumerate(items):
  cap=W-it['width'];bits=1;mask=(1<<(cap+1))-1
  for j in range(i+1):
   count=items[j]['demand']-(j==i);k=1
   while count>0:take=min(k,count);bits=(bits|(bits<<(take*items[j]['width'])))&mask;count-=take;k*=2
  positions={p for p in range(cap+1) if bits>>p&1};positions={0} if it['width']>W/2 else positions;B.append(positions)
  for p in positions:A.add((p,p+it['width']));vertices.update((p,p+it['width']))
 sortedV=sorted(vertices);minimum=min(it['width'] for it in items)
 for p,q in zip(sortedV,sortedV[1:]):
  if p>=minimum and (p,q) not in A:A.add((p,q))
 model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;x={e:model.addVar(vtype=GRB.INTEGER,lb=0,name=f'x_{e[0]}_{e[1]}') for e in sorted(A)};z=model.addVar(vtype=GRB.INTEGER,lb=0,name='z')
 for v in sortedV:model.addConstr(gp.quicksum(x[p,q] for p,q in A if p==v)-gp.quicksum(x[p,q] for p,q in A if q==v)==(z if v==0 else -z if v==W else 0))
 for i,it in enumerate(items):model.addConstr(gp.quicksum(x[p,p+it['width']] for p in B[i])>=it['demand'])
 model.setObjective(z);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  val=int(round(get(z)));variables={f'x_{p}_{q}':int(round(get(v))) for (p,q),v in x.items() if get(v)>.5};variables['z']=val;return {'objective_value':val,'model_variables':variables}
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   s=extract(md.cbGetSolution);json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(s['objective_value'],s)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No arc-flow incumbent')
if __name__=='__main__':main()

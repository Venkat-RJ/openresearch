import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 arcs=data['network']['arcs'];nodes=data['network']['nodes'];com=data['commodities']['commodity_list'];m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0
 x={(k,a):m.addVar(vtype=GRB.BINARY,obj=arc['cost']*c['demand']) for k,c in enumerate(com) for a,arc in enumerate(arcs) if c['demand']<=arc['capacity']};reject=m.addVars(len(com),vtype=GRB.BINARY)
 for k,c in enumerate(com):
  reject[k].Obj=c['demand']*c['artificial_arc_cost'];reject[k].Start=1
  for v in nodes:
   out=gp.quicksum(x[k,a] for a,arc in enumerate(arcs) if (k,a) in x and arc['from_node']==v);inc=gp.quicksum(x[k,a] for a,arc in enumerate(arcs) if (k,a) in x and arc['to_node']==v);b=int(v==c['origin'])-int(v==c['destination']);m.addConstr(out-inc==b*(1-reject[k]));m.addConstr(out<=1-reject[k]);m.addConstr(inc<=1-reject[k])
 for a,arc in enumerate(arcs):m.addConstr(gp.quicksum(c['demand']*x[k,a] for k,c in enumerate(com) if (k,a) in x)<=arc['capacity'])
 for v in x.values():v.Start=0
 def extract(get):
  output=[];obj=0
  for k,c in enumerate(com):
   rejected=get(reject[k])>.5;path=[]
   if rejected:obj+=c['demand']*c['artificial_arc_cost']
   else:
    nxt={arc['from_node']:a for a,arc in enumerate(arcs) if (k,a) in x and get(x[k,a])>.5};u=c['origin']
    while u!=c['destination']:
     a=nxt[u];arc=arcs[a];path.append({'from':u,'to':arc['to_node'],'arc_id':arc['arc_id']});obj+=c['demand']*arc['cost'];u=arc['to_node']
   output.append({'commodity_id':c['commodity_id'],'rejected':rejected,'path_arcs':path})
  return {'objective_value':obj,'commodities':output}
 def cb(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb);return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

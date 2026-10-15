import argparse,json,time,os,math
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 bars=data['bars'];dofs=data['degrees_of_freedom'];loads=data['loading_conditions'];areas=data['discrete_areas'];nodes={n['node_id']:n for n in data['nodes']};I=len(bars);J=len(dofs);K=len(areas);L=len(loads);E=data['material_properties']['modulus_of_elasticity'];density=data['material_properties']['cost_density'];B=[]
 for bar in bars:
  a=nodes[bar['node_i']];b=nodes[bar['node_j']];dirs=['x','y','z'][:data.get('dimension',2)];delta={c:b.get(c,0)-a.get(c,0) for c in dirs};norm=math.sqrt(sum(v*v for v in delta.values()));B.append([delta[d['direction']]/norm*(-1 if d['node']==bar['node_i'] else 1 if d['node']==bar['node_j'] else 0) for d in dofs])
 stress={s['bar_id']:(s['lower'],s['upper']) for s in data.get('bar_specific_stress_bounds',[])};m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;y=m.addVars(I,K,vtype=GRB.BINARY);force=m.addVars(I,L,lb=-GRB.INFINITY);lo=data['displacement_bounds']['lower'];hi=data['displacement_bounds']['upper'];disp=m.addVars(J,L,lb=lo if lo is not None else -GRB.INFINITY,ub=hi if hi is not None else GRB.INFINITY);v=m.addVars(I,K,L,lb=-GRB.INFINITY)
 for i,bar in enumerate(bars):
  m.addConstr(gp.quicksum(y[i,k] for k in range(K))==1);sl,su=stress.get(bar['bar_id'],(data['stress_bounds']['lower'],data['stress_bounds']['upper']))
  for l in range(L):
   m.addConstr(gp.quicksum(B[i][j]*disp[j,l] for j in range(J))==gp.quicksum(v[i,k,l] for k in range(K)));m.addConstr(E/bar['length']*gp.quicksum(areas[k]*v[i,k,l] for k in range(K))==force[i,l])
   for k in range(K):m.addConstr(v[i,k,l]>=bar['length']/E*sl*y[i,k]);m.addConstr(v[i,k,l]<=bar['length']/E*su*y[i,k])
 for l,load in enumerate(loads):
  rhs={d['dof_id']:d['force'] for d in load['loads']}
  for j,d in enumerate(dofs):m.addConstr(gp.quicksum(B[i][j]*force[i,l] for i in range(I))==rhs.get(d['dof_id'],0))
 indices={bar['bar_id']:i for i,bar in enumerate(bars)}
 for group in data.get('linking_groups',[]):
  ref=indices[group['bar_ids'][0]]
  for bid in group['bar_ids'][1:]:
   for k in range(K):m.addConstr(y[indices[bid],k]==y[ref,k])
 m.setObjective(gp.quicksum(density*bars[i]['length']*areas[k]*y[i,k] for i in range(I) for k in range(K)))
 def extract(get):
  chosen=[max(range(K),key=lambda k:get(y[i,k])) for i in range(I)];return {'objective_value':sum(density*bars[i]['length']*areas[chosen[i]] for i in range(I)),'bar_areas':[{'bar_id':bars[i]['bar_id'],'area':areas[chosen[i]],'area_index':chosen[i]} for i in range(I)],'bar_forces':[{'bar_id':bars[i]['bar_id'],'load':l+1,'force':get(force[i,l])} for i in range(I) for l in range(L)],'displacements':[{'dof_id':dofs[j]['dof_id'],'load':l+1,'value':get(disp[j,l])} for j in range(J) for l in range(L)]}
 def cb(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 if not m.SolCount:raise RuntimeError('no feasible truss design')
 return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

import argparse,json,time,os
from collections import deque
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 edges={e['id']:e for e in data['network']['edges']};modes=data['modes'];N=data['network']['num_nodes'];params=data['global_parameters'];F=params['frequency_upper_bound_F'];lam=params['lambda'];limit=params['max_line_length_edges'];lines=[];seen=set()
 for mi,mode in enumerate(modes):
  adj={v:[] for v in range(N)};term=set(mode['terminals'])
  for eid in mode['edge_indices']:
   u,v=edges[eid]['endpoints'];adj[u].append((v,eid));adj[v].append((u,eid))
  for s in sorted(term):
   todo=[([s],[])]
   while todo and time.monotonic()<deadline:
    nodes,path=todo.pop();u=nodes[-1]
    if len(path) and u in term and u>s:
     key=(mi,tuple(sorted(path)))
     if key not in seen:seen.add(key);lines.append((mi,nodes,path))
    if len(path)<limit:
     for v,e in adj[u]:
      if v not in nodes:todo.append((nodes+[v],path+[e]))
 arcs={};out={v:[] for v in range(N)};inc={v:[] for v in range(N)}
 for eid,e in edges.items():
  u,v=e['endpoints']
  for aid,a,b in [(2*eid,u,v),(2*eid+1,v,u)]:arcs[aid]=(a,b,eid,e['traveling_time_seconds']);out[a].append(aid);inc[b].append(aid)
 ods=[od for od in data['od_matrix'] if od['demand']>0 and od['origin']!=od['destination']];origins=sorted({od['origin'] for od in ods});maxlines=max(1,(1900-len(origins)*len(arcs))//2)
 if len(lines)>maxlines:
  pool=sorted(lines,key=lambda z:(-len(z[2]),sum(edges[e]['traveling_time_seconds'] for e in z[2])));chosen=[];covered=set();remaining=[]
  for line in pool:
   if set(line[2])-covered and len(chosen)<maxlines:chosen.append(line);covered.update(line[2])
   else:remaining.append(line)
  lines=chosen+remaining[:maxlines-len(chosen)]
 model=gp.Model();model.Params.Threads=1;model.Params.Seed=0;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;model.Params.OutputFlag=0;x=model.addVars(len(lines),vtype=GRB.BINARY);freq=model.addVars(len(lines),ub=F);flow=model.addVars(origins,list(arcs));cover={eid:[l for l,line in enumerate(lines) if eid in line[2]] for eid in edges}
 for l in range(len(lines)):model.addConstr(freq[l]<=F*x[l])
 for eid,e in edges.items():
  model.addConstr(gp.quicksum(freq[l] for l in cover[eid])<=e['edge_capacity']);capacity=gp.quicksum(modes[lines[l][0]]['vehicle_capacity']*freq[l] for l in cover[eid])
  for aid in [2*eid,2*eid+1]:model.addConstr(gp.quicksum(flow[o,aid] for o in origins)<=capacity)
 for o in origins:
  demand={v:sum(od['demand'] for od in ods if od['origin']==o and od['destination']==v) for v in range(N)}
  for v in range(N):model.addConstr(gp.quicksum(flow[o,a] for a in out[v])-gp.quicksum(flow[o,a] for a in inc[v])==(sum(demand.values()) if v==o else -demand[v]))
 model.setObjective(lam*gp.quicksum(modes[lines[l][0]]['fixed_cost_per_line']*x[l]+sum(edges[e]['operating_cost'] for e in lines[l][2])*freq[l] for l in range(len(lines)))+(1-lam)*gp.quicksum(arcs[a][3]*flow[o,a] for o in origins for a in arcs))
 def extract(get):
  active=[{'line_index':l,'mode':modes[line[0]]['name'],'nodes':line[1],'edges':line[2],'frequency':max(0,get(freq[l]))} for l,line in enumerate(lines) if get(x[l])>.5];paths=[]
  for o in origins:
   residual={a:max(0,get(flow[o,a])) for a in arcs}
   for od in ods:
    if od['origin']!=o:continue
    target=od['destination'];need=od['demand']
    while need>1e-7:
     parent={o:None};q=deque([o])
     while q and target not in parent:
      u=q.popleft()
      for a in out[u]:
       v=arcs[a][1]
       if residual[a]>1e-9 and v not in parent:parent[v]=(u,a);q.append(v)
     if target not in parent:raise RuntimeError('passenger flow decomposition failed')
     path=[];v=target
     while v!=o:u,a=parent[v];path.append(a);v=u
     path.reverse();amount=min(need,min(residual[a] for a in path));need-=amount
     for a in path:residual[a]-=amount
     paths.append({'origin':o,'destination':target,'arcs':path,'flow':amount})
  linecost=sum(modes[lines[l['line_index']][0]]['fixed_cost_per_line']+sum(edges[e]['operating_cost'] for e in l['edges'])*l['frequency'] for l in active);travel=sum(sum(arcs[a][3] for a in p['arcs'])*p['flow'] for p in paths)
  return {'objective_value':lam*linecost+(1-lam)*travel,'active_lines':active,'active_passenger_paths':paths}
 def cb(m,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(m.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 model.Params.TimeLimit=max(.01,deadline-time.monotonic());model.optimize(cb)
 if not model.SolCount:raise RuntimeError('no feasible line plan')
 return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))

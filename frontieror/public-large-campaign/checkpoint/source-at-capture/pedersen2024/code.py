import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,heapq,numpy as np
import highspy
from scipy.sparse import coo_matrix
def primal(nodes,edges,c,Y,yy,subs,quota):
 adjacency={z['id']:[] for z in nodes};root=min(subs);connected={root};chosen=set();selected=set();profit={z['id']:z['profit'] for z in nodes}
 for k,e in enumerate(edges):
  adjacency[e['from']].append((e['to'],k));adjacency[e['to']].append((e['from'],k))
 while not subs<=connected or sum(profit[v] for v in selected)<quota-1e-9:
  dist={v:0. for v in connected};parent={};queue=[(0.,v) for v in connected];heapq.heapify(queue)
  while queue:
   cost,u=heapq.heappop(queue)
   if cost!=dist[u]:continue
   for v,k in adjacency[u]:
    extra=c[Y+k]+(c[yy[v]] if v in yy and v not in selected else 0.);value=cost+extra
    if value<dist.get(v,float('inf')):
     dist[v]=value;parent[v]=(u,k);heapq.heappush(queue,(value,v))
  missing=subs-connected
  if missing:target=min(missing,key=lambda v:dist.get(v,float('inf')))
  else:
   choices=[v for v in yy if v not in connected and v in dist and profit[v]>0]
   if not choices:raise RuntimeError('Reachable turbine yield cannot meet quota')
   target=min(choices,key=lambda v:dist[v]/profit[v])
  if target not in dist:raise RuntimeError('Fixed substations disconnected in input graph')
  while target not in connected:
   connected.add(target)
   if target in yy:selected.add(target)
   u,k=parent[target];chosen.add(k);target=u
 tree={v:[] for v in connected}
 for k in chosen:
  e=edges[k];tree[e['from']].append((e['to'],k));tree[e['to']].append((e['from'],k))
 total=sum(profit[v] for v in selected);leaves=[v for v in connected if v!=root and v not in subs and len(tree[v])==1]
 while leaves:
  v=leaves.pop()
  if v in selected and total-profit[v]<quota-1e-9:continue
  u,k=tree[v][0];tree[u]=[(w,j) for w,j in tree[u] if w!=v];tree[v]=[];chosen.remove(k);connected.remove(v)
  if v in selected:selected.remove(v);total-=profit[v]
  if u!=root and u not in subs and len(tree[u])==1:leaves.append(u)
 values=np.zeros(Y+3*len(edges)+1)
 for v in selected:values[yy[v]]=1
 for k in chosen:values[Y+k]=1
 order=[root];parents={root:None}
 for u in order:
  for v,k in tree[u]:
   if v not in parents:parents[v]=(u,k);order.append(v)
 demand={v:int(v in selected or (v in subs and v!=root)) for v in order}
 for v in reversed(order[1:]):
  u,k=parents[v];edge=edges[k];column=Y+len(edges)+2*k+int(edge['from']!=u);values[column]=demand[v];demand[u]+=demand[v]
 values[-1]=demand[root]
 return values
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();started=time.monotonic();d=json.load(open(a.instance_path));nodes=d['nodes'];edges=d['edges'];alpha=d['parameters']['alpha'];turb=[z for z in nodes if z['type']=='potential_turbine'];subs={z['id'] for z in nodes if z['type']=='substation'};root=min(subs);ids=[z['id'] for z in nodes];N=len(turb)+len(subs);E=len(edges);Y=len(turb);yy={z['id']:i for i,z in enumerate(turb)};c=[alpha*z['cost']+(1-alpha)*z['scenic_impact'] for z in turb]+[alpha*z['cost']+(1-alpha)*z['scenic_impact'] for z in edges]+[0]*(2*E+1);ub=[1]*(Y+E)+[N]*(2*E+1);rr=[];cc=[];vv=[];lo=[];hi=[];balance={i:[] for i in ids}
 def row(terms,l,h):
  r=len(lo);lo.append(l);hi.append(h)
  for j,v in terms:rr.append(r);cc.append(j);vv.append(v)
 row([(i,z['profit']) for i,z in enumerate(turb)],d['parameters']['quota'],np.inf)
 for k,z in enumerate(edges):
  u=z['from'];v=z['to'];f=Y+E+2*k;g=f+1;balance[u].extend([(f,-1),(g,1)]);balance[v].extend([(f,1),(g,-1)]);row([(f,1),(Y+k,-N)],-np.inf,0);row([(g,1),(Y+k,-N)],-np.inf,0)
  for i in (u,v):
   if i in yy:row([(Y+k,1),(yy[i],-1)],-np.inf,0)
 balance[root].append((Y+3*E,1))
 for i in ids:
  rhs=int(i in subs and i!=root);row(balance[i]+([(yy[i],-1)] if i in yy else []),rhs,rhs)
 A=coo_matrix((vv,(rr,cc)),shape=(len(lo),len(c))).tocsc();seed=primal(nodes,edges,c,Y,yy,subs,d['parameters']['quota'])
 logger=None
 if a.log_path:
  from solution_logger import SolutionLogger
  logger=SolutionLogger(a.log_path,sense='minimize')
 def emit(values):
  selected=[z['id'] for i,z in enumerate(turb) if values[i]>.5];es=[{'from':z['from'],'to':z['to']} for k,z in enumerate(edges) if values[Y+k]>.5];obj=sum(c[i] for i in range(Y+E) if values[i]>.5);json.dump({'objective_value':obj,'selected_turbines':selected,'selected_edges':es},open(a.solution_path,'w'))
  if logger:logger.log(obj)
 emit(seed)
 lp=highspy.HighsLp();lp.num_col_=len(c);lp.num_row_=len(lo);lp.col_cost_=c;lp.col_lower_=np.zeros(len(c));lp.col_upper_=ub;lp.row_lower_=lo;lp.row_upper_=hi;lp.integrality_=[highspy.HighsVarType.kInteger]*(Y+E)+[highspy.HighsVarType.kContinuous]*(2*E+1);lp.a_matrix_.format_=highspy.MatrixFormat.kColwise;lp.a_matrix_.start_=A.indptr;lp.a_matrix_.index_=A.indices;lp.a_matrix_.value_=A.data
 solver=highspy.Highs();solver.setOptionValue('threads',1);solver.setOptionValue('output_flag',False);solver.setOptionValue('time_limit',max(.1,float(a.time_limit)-3-(time.monotonic()-started)));solver.setOptionValue('mip_rel_gap',.001);solver.passModel(lp);solver.setSolution(len(c),np.arange(len(c),dtype=np.int32),seed);solver.run();solution=solver.getSolution()
 if solution.value_valid and solver.getInfo().primal_solution_status==2 and np.dot(c,solution.col_value)<np.dot(c,seed)-1e-8:emit(solution.col_value)
if __name__=='__main__':main()

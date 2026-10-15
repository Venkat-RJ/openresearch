import argparse,json,time,os
os.environ['OMP_NUM_THREADS']='1'
from ortools.sat.python import cp_model
import numpy as np

def heuristic(d,dist,clusters,ids,limit,emit):
 D=np.asarray(dist);groups=[np.asarray(g,dtype=int) for g in clusters];m=len(groups);deadline=time.monotonic()+max(.1,limit-3);rng=np.random.default_rng(1997);best=float('inf');result=None
 def cluster_opt(order):
  pivot=min(range(m),key=lambda h:len(groups[order[h]]));order=order[pivot:]+order[:pivot];layers=[groups[c] for c in order];bv=float('inf');tour=None
  for start in layers[0]:
   previous=np.array([start]);cost=np.array([0]);parents=[]
   for layer in layers[1:]:
    matrix=cost[:,None]+D[np.ix_(previous,layer)];parent=matrix.argmin(axis=0);parents.append(parent);cost=matrix[parent,np.arange(len(layer))];previous=layer
   final=cost+D[previous,start];index=int(final.argmin());v=int(final[index])
   if v<bv:
    path=[int(layers[-1][index])]
    for h in range(m-2,0,-1):index=int(parents[h][index]);path.append(int(layers[h][index]))
    path.append(int(start));tour=path[::-1];bv=v
  return order,tour,bv
 def improve(order):
  order,tour,v=cluster_opt(order)
  for sweep in range(100):
   if time.monotonic()>deadline:break
   r=np.array(tour);following=np.roll(r,-1);delta=D[np.ix_(r,r)]+D[np.ix_(following,following)]-D[r,following][:,None]-D[r,following][None,:];bestdelta=0;pair=None
   for i in range(m-2):
    for j in range(i+2,m):
     if i==0 and j==m-1:continue
     if delta[i,j]<bestdelta:bestdelta=delta[i,j];pair=(i,j)
   if pair is None:break
   i,j=pair;order=order[:i+1]+order[i+1:j+1][::-1]+order[j+1:];order,tour,v=cluster_opt(order)
  return order,tour,v
 centers=np.array([[np.mean([d['nodes'][v]['x'] for v in g]),np.mean([d['nodes'][v]['y'] for v in g])] for g in clusters]);center=centers.mean(axis=0);order=np.argsort(np.arctan2(centers[:,1]-center[1],centers[:,0]-center[0])).tolist();order,tour,v=cluster_opt(order);selected=[ids[i] for i in tour]
 if emit:emit({'objective_value':v,'tour':selected,'visited_nodes':selected})
 order,tour,v=improve(order)
 while True:
  if v<best:
   best=v;selected=[ids[i] for i in tour];result={'objective_value':v,'tour':selected,'visited_nodes':selected}
   if emit:emit(result)
  if time.monotonic()>deadline:break
  start=int(rng.integers(m));remaining=set(range(m));remaining.remove(start);order=[start];point=int(rng.choice(groups[start]))
  while remaining:
   candidates=[(int(D[point,g].min())*(1+float(rng.uniform(0,.15))),c) for c in remaining for g in [groups[c]]];c=min(candidates)[1];order.append(c);remaining.remove(c);point=int(groups[c][D[point,groups[c]].argmin()])
  order,tour,v=improve(order)
 return result

def solve(d,limit,emit=None):
 nodes=d['nodes'];n=len(nodes);ids=[v['id'] for v in nodes];index={v:i for i,v in enumerate(ids)}
 values=d['distance_matrix_upper_triangular']['values'];dist=[[0]*n for _ in range(n)];k=0
 for i in range(n):
  for j in range(i+1,n):dist[i][j]=dist[j][i]=int(values[k]);k+=1
 clusters=[[index[v] for v in c['node_ids']] for c in d['clusters']]
 if n>80:return heuristic(d,dist,clusters,ids,limit,emit)
 m=cp_model.CpModel();selected=[m.new_bool_var('v'+str(i)) for i in range(n)];arcs=[];edges={};terms=[]
 for c in clusters:m.add_exactly_one(selected[i] for i in c)
 for i in range(n):
  arcs.append((i,i,selected[i].Not()))
  for j in range(n):
   if i==j or nodes[i]['cluster']==nodes[j]['cluster']:continue
   a=m.new_bool_var(f'e{i}_{j}');arcs.append((i,j,a));edges[i,j]=a;terms.append(dist[i][j]*a)
 m.add_circuit(arcs);m.minimize(sum(terms));s=cp_model.CpSolver();s.parameters.max_time_in_seconds=max(.1,limit-1);s.parameters.num_search_workers=1
 status=s.solve(m)
 if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError('No feasible GTSP tour within time limit')
 nxt={i:j for (i,j),v in edges.items() if s.value(v)};start=min(nxt);tour=[start]
 while nxt[tour[-1]]!=start:tour.append(nxt[tour[-1]])
 cost=sum(dist[tour[k]][tour[(k+1)%len(tour)]] for k in range(len(tour)));tour=[ids[i] for i in tour]
 return {'objective_value':cost,'tour':tour,'visited_nodes':tour}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 start=time.time()
 def emit(result):
  with open(a.solution_path,'w') as f:json.dump(result,f)
  if a.log_path:
   with open(a.log_path,'a') as f:f.write(json.dumps({'time':time.time()-start,'objective_value':result['objective_value']})+'\n')
 if a.log_path:open(a.log_path,'w').close()
 result=solve(json.load(open(a.instance_path)),a.time_limit,emit)
 with open(a.solution_path,'w') as f:json.dump(result,f)
 if a.log_path:
  with open(a.log_path,'w') as f:json.dump({'time':time.time()-start,'objective_value':result['objective_value']},f);f.write('\n')

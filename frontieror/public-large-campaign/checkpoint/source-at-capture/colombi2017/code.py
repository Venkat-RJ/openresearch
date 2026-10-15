import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,collections
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from ortools.sat.python import cp_model
from ortools.graph.python import min_cost_flow

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));n=d['num_nodes'];depot=d['depot'];arcs=d['arcs'];prof=d['profitable_arcs'];costs={(i,j):c for i,j,c in arcs};indices={(i,j):z for z,(i,j,c)in enumerate(arcs)};matrix=coo_matrix(([c for i,j,c in arcs],([i for i,j,c in arcs],[j for i,j,c in arcs])),shape=(n,n)).tocsr();distance,predecessor=shortest_path(matrix,directed=True,return_predecessors=True);tails=np.array([i for i,j,c in prof]);heads=np.array([j for i,j,c in prof]);profits=np.array([c for i,j,c in prof]);arc_cost=np.array([costs[i,j]for i,j,c in prof]);strong=np.zeros((n,n),dtype=bool);weak=np.zeros((n,n))
 for i,j in d['strong_incompatibilities']:strong[i,j]=strong[j,i]=True
 for i,j,c in d['weak_incompatibilities']:weak[i,j]+=c;weak[j,i]+=c
 rng=np.random.default_rng(2017);best_value=0;best_selected=[];best_counts=[0]*len(arcs);construction_end=start+min(8 if n>100 else 2,float(a.time_limit)*.15)
 def path(i,j):
  row=[j]
  while row[-1]!=i:row.append(int(predecessor[i,row[-1]]))
  return list(reversed(row))
 def balanced_tour(chosen):
  counts=[0]*len(arcs);balance=np.zeros(n,dtype=int)
  for z in np.flatnonzero(chosen):i,j,c=prof[z];counts[indices[i,j]]=1;balance[i]+=1;balance[j]-=1
  optimizer=min_cost_flow.SimpleMinCostFlow()
  for i,j,c in arcs:optimizer.add_arc_with_capacity_and_unit_cost(i,j,len(prof)+1,int(c))
  for i in range(n):optimizer.set_node_supply(i,int(-balance[i]))
  if optimizer.solve()!=optimizer.OPTIMAL:return None
  for z in range(len(arcs)):counts[z]+=optimizer.flow(z)
  while True:
   adjacency={i:set()for i in range(n)};touched={depot}
   for z,q in enumerate(counts):
    if q:i,j,c=arcs[z];adjacency[i].add(j);adjacency[j].add(i);touched.update((i,j))
   reached={depot};queue=[depot]
   for i in queue:
    for j in adjacency[i]:
     if j not in reached:reached.add(j);queue.append(j)
   missing=touched-reached
   if not missing:return counts
   source=np.array(list(reached));target=np.array(list(missing));roundtrip=distance[np.ix_(source,target)]+distance[np.ix_(target,source)].T;r,c=np.unravel_index(np.argmin(roundtrip),roundtrip.shape);u=int(source[r]);v=int(target[c])
   for row in (path(u,v),path(v,u)):
    for i,j in zip(row,row[1:]):counts[indices[i,j]]+=1
 for attempt in range(30):
  order=[];chosen=np.zeros(len(prof),dtype=bool);enabled=np.zeros(n,dtype=bool)
  while time.monotonic()<construction_end:
   previous=np.array([depot]+[int(heads[z])for z in order]);following=np.array([int(tails[z])for z in order]+[depot]);extra=distance[previous[None,:],tails[:,None]]+arc_cost[:,None]+distance[heads[:,None],following[None,:]]-distance[previous,following][None,:];positions=np.argmin(extra,axis=1);travel=extra[np.arange(len(prof)),positions];penalty=weak@enabled;gains=profits-travel-np.where(enabled[tails],0,penalty[tails]);blocked=chosen|np.any(strong[:,enabled],axis=1)[tails];gains[blocked]=-np.inf
   bundles=np.bincount(tails,weights=np.where(chosen,0,np.maximum(profits-travel,0)),minlength=n)-penalty;bundles[enabled|np.any(strong[:,enabled],axis=1)]=-np.inf;added=False
   for tail in np.argsort(bundles)[-3:][::-1]:
    if bundles[tail]<=max(0,float(np.max(gains))):break
    trial=order[:];new=[];available=[int(z)for z in np.flatnonzero((tails==tail)&(~chosen))];bundle_profit=0.
    while available:
     before=np.array([depot]+[int(heads[z])for z in trial]);after=np.array([int(tails[z])for z in trial]+[depot]);zz=np.array(available);delta=distance[before[None,:],tails[zz,None]]+arc_cost[zz,None]+distance[heads[zz,None],after[None,:]]-distance[before,after][None,:];where=np.argmin(delta,axis=1);benefit=profits[zz]-delta[np.arange(len(zz)),where];index=int(np.argmax(benefit))
     if benefit[index]<=0:break
     z=available.pop(index);trial.insert(int(where[index]),z);new.append(z);bundle_profit+=float(benefit[index])
    if bundle_profit>penalty[tail]:order=trial;chosen[new]=True;enabled[tail]=True;added=True;break
   if added:continue
   if not np.any(gains>0):
    current_travel=float(np.sum(arc_cost[order])+np.sum(distance[np.array([depot]+[int(heads[z])for z in order]),np.array([int(tails[z])for z in order]+[depot]) ]));best_remove=None;remove_gain=0.
    for tail in np.flatnonzero(enabled):
     reduced=[z for z in order if tails[z]!=tail];new_travel=float(np.sum(arc_cost[reduced])+np.sum(distance[np.array([depot]+[int(heads[z])for z in reduced]),np.array([int(tails[z])for z in reduced]+[depot])]));gain=penalty[tail]-float(profits[chosen&(tails==tail)].sum())+current_travel-new_travel
     if gain>remove_gain+1e-8:remove_gain=gain;best_remove=(int(tail),reduced)
    if best_remove is None:break
    tail,order=best_remove;chosen[tails==tail]=False;enabled[tail]=False;continue
   score=gains if attempt==0 else gains*(1+rng.uniform(-.2,.2,len(prof)));z=int(np.argmax(score))
   if gains[z]<=0:break
   order.insert(int(positions[z]),z);chosen[z]=True;enabled[tails[z]]=True
  counts=[0]*len(arcs);previous=depot
  for z in order:
   row=path(previous,int(tails[z]))
   for i,j in zip(row,row[1:]):counts[indices[i,j]]+=1
   counts[indices[int(tails[z]),int(heads[z])]]+=1;previous=int(heads[z])
  row=path(previous,depot)
  for i,j in zip(row,row[1:]):counts[indices[i,j]]+=1
  balanced=balanced_tour(chosen)
  if balanced is not None and sum(c*q for(i,j,c),q in zip(arcs,balanced))<sum(c*q for(i,j,c),q in zip(arcs,counts)):counts=balanced
  possible=np.array([counts[indices[int(i),int(j)]]>0 for i,j,c in prof])
  while True:
   gains=profits-np.where(enabled[tails],0,(weak@enabled)[tails]);gains[chosen|(~possible)|np.any(strong[:,enabled],axis=1)[tails]]=-np.inf;z=int(np.argmax(gains))
   if gains[z]<=0:break
   chosen[z]=True;enabled[tails[z]]=True
  value=int(profits@chosen-sum(c*q for(i,j,c),q in zip(arcs,counts))-sum(c for i,j,c in d['weak_incompatibilities']if enabled[i]and enabled[j]))
  if value>best_value:best_value=value;best_selected=list(np.flatnonzero(chosen));best_counts=counts
  if time.monotonic()>construction_end:break
 hint_visited={i for z,q in enumerate(best_counts)if q for i in arcs[z][:2]};adjacency={i:[]for i in range(n)}
 for z,q in enumerate(best_counts):
  if q:adjacency[arcs[z][0]].append((arcs[z][1],z))
 discovery=[depot];parents={};seen={depot}
 for i in discovery:
  for j,z in adjacency[i]:
   if j not in seen:seen.add(j);parents[j]=(i,z);discovery.append(j)
 subtree={i:1 for i in seen};hint_flow={}
 for j in reversed(discovery[1:]):i,z=parents[j];hint_flow[z]=subtree[j];subtree[i]+=subtree[j]
 m=cp_model.CpModel();x=[];u=[];f=[];visited=[m.NewBoolVar('v'+str(i))for i in range(n)];lookup={};obj=[]
 for z,(i,j,c)in enumerate(arcs):
  xx=m.NewIntVar(0,len(prof)+1,'x'+str(z));uu=m.NewBoolVar('u'+str(z));ff=m.NewIntVar(0,n,'f'+str(z));m.Add(xx<=(len(prof)+1)*uu);m.Add(xx>=uu);m.Add(ff<=n*uu);m.Add(visited[i]>=uu);m.Add(visited[j]>=uu);x.append(xx);u.append(uu);f.append(ff);lookup[i,j]=z;obj.append(-c*xx)
 for i in range(n):
  ins=[z for z,(k,j,c)in enumerate(arcs)if j==i];outs=[z for z,(k,j,c)in enumerate(arcs)if k==i];m.Add(sum(x[z]for z in ins)==sum(x[z]for z in outs));m.Add(visited[i]<=sum(u[z]for z in ins))
  if i==depot:m.Add(sum(f[z]for z in outs)-sum(f[z]for z in ins)==sum(visited[j]for j in range(n)if j!=depot))
  else:m.Add(sum(f[z]for z in ins)-sum(f[z]for z in outs)==visited[i])
 served=[];active=[m.NewBoolVar('a'+str(i))for i in range(n)]
 for z,(i,j,profit)in enumerate(prof):
  y=m.NewBoolVar('s'+str(z));served.append(y);m.Add(y<=u[lookup[i,j]]);m.Add(active[i]>=y);obj.append(profit*y)
 for i in range(n):m.Add(active[i]<=sum(served[z]for z,(k,j,c)in enumerate(prof)if k==i))
 for i,j in d['strong_incompatibilities']:m.Add(active[i]+active[j]<=1)
 for z,(i,j,c)in enumerate(d['weak_incompatibilities']):
  y=m.NewBoolVar('pen'+str(z));m.Add(y>=active[i]+active[j]-1);obj.append(-c*y)
 selected_set=set(best_selected);hint_active={prof[z][0]for z in best_selected}
 for z in range(len(arcs)):m.AddHint(x[z],best_counts[z]);m.AddHint(u[z],int(best_counts[z]>0));m.AddHint(f[z],hint_flow.get(z,0))
 for i in range(n):m.AddHint(visited[i],int(i in hint_visited));m.AddHint(active[i],int(i in hint_active))
 for z,v in enumerate(served):m.AddHint(v,int(z in selected_set))
 for index,var in enumerate(m.Proto().variables):
  if var.name.startswith('pen'):
   z=int(var.name[3:]);i,j,c=d['weak_incompatibilities'][z];m.AddHint(m.GetIntVarFromProtoIndex(index),int(i in hint_active and j in hint_active))
 m.Add(sum(obj)>=best_value);m.Maximize(sum(obj));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,float(a.time_limit)-2-(time.monotonic()-start));status=s.Solve(m);ok=status in(cp_model.FEASIBLE,cp_model.OPTIMAL);selected=[z for z,v in enumerate(served)if s.Value(v)]if ok else best_selected;counts=[s.Value(v)for v in x]if ok else best_counts;starts=set(prof[z][0]for z in selected);value=sum(prof[z][2]for z in selected)-sum(c*q for(i,j,c),q in zip(arcs,counts))-sum(c for i,j,c in d['weak_incompatibilities']if i in starts and j in starts);sol={'objective_value':value,'served_arcs':[{'from':prof[z][0],'to':prof[z][1],'profit':prof[z][2],'cost':arcs[lookup[prof[z][0],prof[z][1]]][2]}for z in selected],'tour_arcs':[{'from':i,'to':j,'count':q,'cost':c}for(i,j,c),q in zip(arcs,counts)if q]};json.dump(sol,open(a.solution_path,'w'))
if __name__=='__main__':main()

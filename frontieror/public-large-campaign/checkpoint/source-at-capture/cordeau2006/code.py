import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,math,random
import numpy as np
from scipy.optimize import linprog

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();deadline=time.monotonic()+max(.1,float(a.time_limit)-2);d=json.load(open(a.instance_path));nodes=sorted(d['nodes'],key=lambda v:v['node_id']);N=len(nodes);R=d['num_users'];K=d['num_vehicles'];Q=d['vehicle_capacity'];dist=np.array([[math.hypot(v['x']-u['x'],v['y']-u['y'])for u in nodes]for v in nodes]);rng=random.Random(0);cache={}
 def route_cost(row):return float(sum(dist[i,j]for i,j in zip(row,row[1:])))
 def schedule(row):
  key=tuple(row)
  if key in cache:return cache[key]
  load=0
  for i in row:
   load+=nodes[i]['load']
   if load<0 or load>Q:cache[key]=None;return None
  positions={j:i for i,j in enumerate(row)};L=len(row);A=[];b=[]
  for k,(i,j)in enumerate(zip(row,row[1:])):
   rr=np.zeros(L);rr[k]=1;rr[k+1]=-1;A.append(rr);b.append(-nodes[i]['service_duration']-dist[i,j])
  for i in row:
   if nodes[i]['node_type']!='pickup':continue
   j=nodes[i]['paired_node'];ii=positions[i];jj=positions[j]
   if jj<ii:cache[key]=None;return None
   rr=np.zeros(L);rr[jj]=1;rr[ii]=-1;A.append(rr);b.append(d['maximum_ride_time']+nodes[i]['service_duration']);A.append(-rr);b.append(-nodes[i]['service_duration']-dist[i,j])
  rr=np.zeros(L);rr[-1]=1;rr[0]=-1;A.append(rr);b.append(d['maximum_route_duration']);objective=np.zeros(L);objective[-1]=1;objective[0]=-1;bounds=[(nodes[i]['earliest_time'],nodes[i]['latest_time'])for i in row];res=linprog(objective,A_ub=np.array(A),b_ub=b,bounds=bounds,method='highs',options={'threads':1});answer=res.x if res.success else None;cache[key]=answer;return answer
 best=None;bestval=float('inf');run=0
 while time.monotonic()<deadline:
  routes=[[0,N-1]for _ in range(K)];order=sorted(range(1,R+1),key=lambda i:nodes[i]['latest_time']+(rng.uniform(-40,40)if run else 0));valid=True
  for request in order:
   choices=[]
   for k,row in enumerate(routes):
    old=route_cost(row)
    for i in range(1,len(row)):
     for j in range(i+1,len(row)+1):
      candidate=row[:i]+[request]+row[i:j-1]+[request+R]+row[j-1:];delta=route_cost(candidate)-old
      if choices and delta>=choices[0][0]:continue
      if time.monotonic()>deadline:break
      if schedule(candidate)is not None:choices=[(delta,k,candidate)]
   if not choices:valid=False;break
   _,k,row=choices[0];routes[k]=row
  if valid:
   value=sum(route_cost(row)for row in routes)
   if value<bestval:best=routes;bestval=value
  run+=1
  if best is not None and run>=20:break
 if best is None:raise RuntimeError('No feasible insertion schedule found before deadline')
 st={}
 for k,row in enumerate(best):
  times=schedule(row)
  for i,t in zip(row,times):st['depot_start_'+str(k)if i==0 else 'depot_end_'+str(k)if i==N-1 else str(i)]=float(t)
 ride={str(nodes[i]['user_id']):st[str(i+R)]-st[str(i)]-nodes[i]['service_duration']for i in range(1,R+1)};json.dump({'objective_value':bestval,'routes':{str(k):row for k,row in enumerate(best)},'service_times':st,'ride_times':ride},open(a.solution_path,'w'))
if __name__=='__main__':main()

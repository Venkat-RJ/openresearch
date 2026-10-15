import argparse,json,time,random
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();deadline=time.monotonic()+max(.1,a.time_limit-2)
 d=json.load(open(a.instance_path));n=d['n_requests'];end=2*n+1;Q=d['vehicle_capacity'];nd={v['node_id']:v for v in d['nodes']};ar={(v['from'],v['to']):(v['cost'],v['travel_time']) for v in d['arcs']};rng=random.Random(0);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def evaluate(seq,details=False):
  t=nd[0]['tw_early'];load=0;stack=[];cost=0;out=[{'node_id':0,'arrival_time':t,'load_after':0}]
  for i,j in zip(seq,seq[1:]):
   if (i,j) not in ar:return None
   c,travel=ar[i,j];cost+=c;t=max(t+nd[i]['service_time']+travel,nd[j]['tw_early']);load+=nd[j]['load']
   if t>nd[j]['tw_late']+1e-8 or load<max(0,nd[j]['load'])-1e-8 or load>min(Q,Q+nd[j]['load'])+1e-8:return None
   if 1<=j<=n:stack.append(j)
   elif n<j<=2*n:
    if not stack or stack[-1]!=j-n:return None
    stack.pop()
   out.append({'node_id':j,'arrival_time':t,'load_after':load})
  if stack:return None
  return (cost,out) if details else cost
 best=float('inf');best_routes=[]
 def save(routes):
  nonlocal best,best_routes
  vals=[evaluate(r,True) for r in routes]
  if any(v is None for v in vals):return
  obj=sum(v[0] for v in vals)
  if obj<best-1e-8:
   best=obj;best_routes=[r[:] for r in routes];s={'objective_value':obj,'routes':[{'vehicle_id':i,'route_sequence':r,'route_details':v[1]} for i,(r,v) in enumerate(zip(routes,vals))]};json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(obj,s)
 def insert(routes,req):
  singleton=[0,req,req+n,end];val=evaluate(singleton)
  if val is None:raise RuntimeError('Infeasible singleton request')
  choice=(val,len(routes),singleton)
  for k,r in enumerate(routes):
   old=evaluate(r)
   for i in range(1,len(r)):
    for j in range(i+1,len(r)+1):
     candidate=r[:i]+[req]+r[i:j-1]+[req+n]+r[j-1:];v=evaluate(candidate)
     if v is not None and v-old<choice[0]:choice=(v-old,k,candidate)
  _,k,new=choice
  if k==len(routes):routes.append(new)
  else:routes[k]=new
  return routes
 routes=[[0,i,i+n,end] for i in range(1,n+1)];save(routes);trial=0
 while time.monotonic()<deadline:
  if trial<8:rs=[];order=list(range(1,n+1));order.sort(key=lambda i:nd[i]['tw_late']) if trial==0 else rng.shuffle(order)
  else:
   removed=set(rng.sample(range(1,n+1),min(n,8+trial%8)));rs=[[j for j in r if j not in removed and j-n not in removed] for r in best_routes];rs=[r for r in rs if len(r)>2];order=list(removed);rng.shuffle(order)
  for req in order:
   if time.monotonic()>deadline:break
   insert(rs,req)
  else:save(rs)
  trial+=1
if __name__=='__main__':main()

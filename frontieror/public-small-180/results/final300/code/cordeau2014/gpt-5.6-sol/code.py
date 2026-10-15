import argparse,json,time,bisect,random
from array import array
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();deadline=time.monotonic()+max(.1,a.time_limit-2);d=json.load(open(a.instance_path));n=d['parameters']['n_customers'];dist=d['arcs']['distances'];speed=d['arcs']['speeds_v_ijh'];bounds=d['parameters']['period_boundaries'];H=len(bounds)-1;logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 best=float('inf')
 def arrival(i,j,t):
  rem=dist[i][j];k=min(H-1,max(0,bisect.bisect_left(bounds,t)-1));v=speed[i][j]
  while k<H-1 and t+rem/v[k]>bounds[k+1]+1e-12:rem-=v[k]*(bounds[k+1]-t);t=bounds[k+1];k+=1
  return t+rem/v[k]
 def value(r):
  t=0
  for i,j in zip(r,r[1:]):t=arrival(i,j,t)
  return t
 def save(r,obj):
  nonlocal best
  if obj<best-1e-9:
   best=obj;s={'objective_value':obj,'tour':r};json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(obj,s)
 remaining=set(range(1,n+1));r=[0];t=0
 while remaining:
  j=min(remaining,key=lambda j:arrival(r[-1],j,t));t=arrival(r[-1],j,t);r.append(j);remaining.remove(j)
 r.append(0);save(r,value(r));size=1<<n;dp=array('d',[float('inf')])*(size*n);prev=array('b',[-1])*(size*n)
 for j in range(n):dp[(1<<j)*n+j]=arrival(0,j+1,0)
 full=size-1
 for mask in range(1,size):
  if mask%256==0 and time.monotonic()>deadline:return
  bits=mask
  while bits:
   bit=bits&-bits;bits-=bit;i=bit.bit_length()-1;t=dp[mask*n+i]
   if t>=best:continue
   rest=full^mask
   if not rest:
    obj=arrival(i+1,0,t)
    if obj<best:
     seq=[];cur=i;mm=mask
     while cur>=0:seq.append(cur+1);p=prev[mm*n+cur];mm^=1<<cur;cur=p
     save([0]+seq[::-1]+[0],obj)
   while rest:
    bitj=rest&-rest;rest-=bitj;j=bitj.bit_length()-1;nt=arrival(i+1,j+1,t);idx=(mask|bitj)*n+j
    if nt<dp[idx]:dp[idx]=nt;prev[idx]=i
if __name__=='__main__':main()

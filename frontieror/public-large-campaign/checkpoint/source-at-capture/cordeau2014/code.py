import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,bisect,itertools

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();deadline=time.monotonic()+max(.1,float(a.time_limit)-2);d=json.load(open(a.instance_path));N=d['parameters']['total_vertices'];bound=d['parameters']['period_boundaries'];H=d['parameters']['H'];dist=d['arcs']['distances'];speed=d['arcs']['speeds_v_ijh'];depot=d['depot']['index']
 def arrival(i,j,t):
  rem=dist[i][j];h=min(H-1,bisect.bisect_right(bound,t)-1)
  while True:
   v=speed[i][j][h]
   if h==H-1 or rem<=v*(bound[h+1]-t)+1e-10:return t+rem/max(v,1e-15)
   rem-=v*(bound[h+1]-t);t=bound[h+1];h+=1
 def value(tour):
  t=0.
  for i,j in zip(tour,tour[1:]):t=arrival(i,j,t)
  return t
 remaining=set(range(N))-{depot};tour=[depot];t=0
 while remaining:
  j=min(remaining,key=lambda j:arrival(tour[-1],j,t));t=arrival(tour[-1],j,t);tour.append(j);remaining.remove(j)
 tour.append(depot);best=value(tour)
 if N<=10:
  for perm in itertools.permutations([i for i in range(N)if i!=depot]):
   if time.monotonic()>deadline:break
   candidate=[depot,*perm,depot];v=value(candidate)
   if v<best:tour=candidate;best=v
 else:
  improved=True
  while improved and time.monotonic()<deadline:
   improved=False
   for i in range(1,N-1):
    for j in range(i+1,N):
     if time.monotonic()>deadline:break
     candidate=tour[:i]+tour[i:j+1][::-1]+tour[j+1:];v=value(candidate)
     if v<best-1e-9:tour=candidate;best=v;improved=True
 json.dump({'objective_value':best,'tour':tour},open(a.solution_path,'w'))
if __name__=='__main__':main()

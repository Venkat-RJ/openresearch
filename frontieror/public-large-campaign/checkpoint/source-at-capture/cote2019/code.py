import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random
from ortools.sat.python import cp_model
from packing import compute_expected_area_weight,is_route_always_infeasible,compute_route_recourse

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();deadline=time.monotonic()+max(.1,float(a.time_limit)-2);d=json.load(open(a.instance_path));vehicle=d['vehicle'];H=vehicle['H'];W=vehicle['W'];Q=vehicle['Q'];K=vehicle['K'];cf=d['parameters']['recourse_cost_cf'];D=d['distance_matrix'];customers={v['id']:v for v in d['customers']};expected=compute_expected_area_weight(d['customers']);rng=random.Random(0);cache={};pool={}
 def cost(row):return sum(D[i][j]for i,j in zip([0]+row,row+[0]))
 def feasible(row):
  key=tuple(row)
  if key not in cache:cache[key]=sum(expected[j][0]for j in row)<=H*W+1e-8 and sum(expected[j][1]for j in row)<=Q+1e-8 and not is_route_always_infeasible(row,customers,H,W,Q)
  return cache[key]
 best=None;bestvalue=float('inf')
 for run in range(400):
  if time.monotonic()>deadline-8:break
  rows=[[]for _ in range(K)];order=list(customers);rng.shuffle(order)
  if run==0:order.sort(key=lambda i:-expected[i][0])
  valid=True
  for j in order:
   options=[]
   for k,row in enumerate(rows):
    for pos in range(len(row)+1):
     candidate=row[:pos]+[j]+row[pos:]
     if feasible(candidate):options.append((cost(candidate)-cost(row)+rng.uniform(0,20),k,candidate))
   if not options:valid=False;break
   _,k,row=min(options);rows[k]=row;pool[tuple(row)]=cost(row)
  if valid and all(rows):
   value=sum(cost(row)+compute_route_recourse(row,customers,H,W,Q,cf)for row in rows)
   if value<bestvalue:bestvalue=value;best=[row.copy()for row in rows]
 if time.monotonic()<deadline and pool:
  routes=list(pool);model=cp_model.CpModel();ys=[model.NewBoolVar('r'+str(i))for i in range(len(routes))];values=[]
  for row in routes:
   if time.monotonic()>deadline-2:values.append(cost(list(row))+cf*len(row))
   else:values.append(cost(list(row))+compute_route_recourse(list(row),customers,H,W,Q,cf))
  for j in customers:model.Add(sum(ys[i]for i,row in enumerate(routes)if j in row)==1)
  model.Add(sum(ys)==K);model.Minimize(sum(int(round(10000*values[i]))*v for i,v in enumerate(ys)));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());status=s.Solve(model)
  if status in(cp_model.FEASIBLE,cp_model.OPTIMAL):
   candidate=[list(row)for row,v in zip(routes,ys)if s.Value(v)];value=sum(cost(row)+compute_route_recourse(row,customers,H,W,Q,cf)for row in candidate)
   if value<bestvalue:best=candidate;bestvalue=value
 if best is None:raise RuntimeError('No feasible four-route two-dimensional packing plan found')
 json.dump({'objective_value':bestvalue,'routes':best},open(a.solution_path,'w'))
if __name__=='__main__':main()

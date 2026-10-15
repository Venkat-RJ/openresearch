import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,math
from ortools.sat.python import cp_model

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();deadline=time.monotonic()+max(.01,float(a.time_limit)-2);d=json.load(open(a.instance_path));F=d['num_facilities'];C=d['num_customers'];cost=d['facility_cost'];dem=d['customer_demands'];fc=[set(d['facility_to_customers_coverage'].get(str(i),[]))for i in range(F)];results={};types=d.get('problem_types',['PSCLP','MCLP'])
 for variant in types:
  opened=[];covered=set();budget=d['budget_B'];target=d['covering_demand_D'];remaining=set(range(F))
  while remaining:
   choices=[i for i in remaining if variant=='PSCLP'or cost[i]<=budget];
   if not choices:break
   i=max(choices,key=lambda i:sum(dem[j]for j in fc[i]-covered)/max(cost[i],.001));gain=sum(dem[j]for j in fc[i]-covered)
   if gain<=0:break
   opened.append(i);covered|=fc[i];remaining.remove(i);budget-=cost[i]
   if variant=='PSCLP'and sum(dem[j]for j in covered)>=target:break
  if time.monotonic()<deadline:
   m=cp_model.CpModel();y=[m.NewBoolVar('y'+str(i))for i in range(F)];z=[m.NewBoolVar('z'+str(j))for j in range(C)]
   for j in range(C):m.Add(z[j]<=sum(y[i]for i in d['customer_to_facilities_coverage'].get(str(j),[])))
   totalcost=sum(int(cost[i])*y[i]for i in range(F));totaldem=sum(int(dem[j])*z[j]for j in range(C))
   if variant=='PSCLP':m.Add(totaldem>=math.ceil(target-1e-8));m.Minimize(totalcost)
   else:m.Add(totalcost<=d['budget_B']);m.Maximize(totaldem)
   for i in range(F):m.AddHint(y[i],int(i in opened))
   s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,(deadline-time.monotonic())/(len(types)-len(results)));status=s.Solve(m)
   if status in(cp_model.FEASIBLE,cp_model.OPTIMAL):opened=[i for i in range(F)if s.Value(y[i])];covered=set().union(*(fc[i]for i in opened))if opened else set()
  demand=sum(dem[j]for j in covered);value=sum(cost[i]for i in opened)if variant=='PSCLP'else demand;results[variant]={'objective_value':value,'open_facilities':opened,'covered_demand':demand}
 primary=types[0];json.dump({'objective_value':results[primary]['objective_value'],'primary_problem_type':primary,'results':results},open(a.solution_path,'w'))
if __name__=='__main__':main()

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));old='depot'in d;n=d['num_customers'];T=d['num_periods'];K=d['num_vehicles'];Q=d['vehicle_capacity'];D=d['distance_matrix']
 if old:
  depot=d['depot'];customers=sorted(d['customers'],key=lambda v:v['id']);initial=[depot['initial_inventory']]+[v['initial_inventory']for v in customers];caps=[depot['max_inventory']]+[v['max_inventory']for v in customers];holding=[depot['holding_cost']]+[v['holding_cost']for v in customers];demand=[[v['demand_per_period']]*T for v in customers];production=[depot['production_per_period']]*T;minimum=[0]+[v.get('min_inventory',0)for v in customers]
 else:initial=d['initial_inventories'];caps=d['inventory_capacities'];holding=d['holding_costs'];demand=d['customer_demands'];production=d['supplier_production'];minimum=[0]*(n+1)
 m=cp_model.CpModel();inventory=[[m.NewIntVar(minimum[i],caps[i],'I%d_%d'%(i,t))for t in range(T)]for i in range(n+1)];delivery={};visits={};arcs={};obj=[]
 for t in range(T):
  for i in range(n+1):obj.append(int(round(holding[i]*10000))*inventory[i][t])
  for k in range(K):
   use=m.NewBoolVar('use%d_%d'%(t,k));vv={i:m.NewBoolVar('v%d_%d_%d'%(t,k,i))for i in range(1,n+1)};ff={};aa={}
   for i in range(n+1):
    for j in range(n+1):
     if i==j:continue
     x=m.NewBoolVar('x%d_%d_%d_%d'%(t,k,i,j));f=m.NewIntVar(0,Q,'f%d_%d_%d_%d'%(t,k,i,j));m.Add(f<=Q*x);aa[i,j]=x;ff[i,j]=f;obj.append(D[i][j]*10000*x)
   m.Add(sum(aa[0,j]for j in range(1,n+1))==use);m.Add(sum(aa[j,0]for j in range(1,n+1))==use)
   for i in range(1,n+1):
    q=m.NewIntVar(0,min(Q,caps[i]),'q%d_%d_%d'%(t,k,i));delivery[t,k,i]=q;visits[t,k,i]=vv[i];m.Add(q<=Q*vv[i]);m.Add(q>=vv[i]);m.Add(sum(aa[i,j]for j in range(n+1)if j!=i)==vv[i]);m.Add(sum(aa[j,i]for j in range(n+1)if j!=i)==vv[i]);m.Add(sum(ff[j,i]for j in range(n+1)if j!=i)-sum(ff[i,j]for j in range(n+1)if j!=i)==q)
   m.Add(sum(delivery[t,k,i]for i in range(1,n+1))<=Q);m.Add(sum(ff[0,j]for j in range(1,n+1))-sum(ff[j,0]for j in range(1,n+1))==sum(delivery[t,k,i]for i in range(1,n+1)));arcs[t,k]=aa
   if k:m.Add(use<=sum(arcs[t,k-1][0,j]for j in range(1,n+1)))
  for i in range(1,n+1):
   q=sum(delivery[t,k,i]for k in range(K));prev=initial[i]if t==0 else inventory[i][t-1];m.Add(inventory[i][t]==prev+q-demand[i-1][t]);m.Add(prev+q<=caps[i]);m.Add(sum(visits[t,k,i]for k in range(K))<=1)
  prev=initial[0]if t==0 else inventory[0][t-1];m.Add(inventory[0][t]==prev+production[t]-sum(delivery[t,k,i]for k in range(K)for i in range(1,n+1)))
 m.Minimize(sum(obj));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,float(a.time_limit)-2-(time.monotonic()-start));status=s.Solve(m)
 if status not in(cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible inventory routing plan found')
 periods={};rr={};iv={str(i):{}for i in range(n+1)};value=0
 for t in range(T):
  routes=[];qd={};rmap={}
  for k in range(K):
   aa=arcs[t,k];nexts={i:j for(i,j),v in aa.items()if s.Value(v)}
   if 0 not in nexts:continue
   row=[0];j=nexts[0]
   while j!=0:row.append(j);j=nexts[j]
   row.append(0);quant={str(i):s.Value(delivery[t,k,i])for i in row[1:-1]};qd.update(quant);routes.append(row);rmap[str(k)]={'route':row,'deliveries':quant};value+=sum(D[i][j]for i,j in zip(row,row[1:]))
  inv={('depot'if i==0 else str(i)):s.Value(inventory[i][t])for i in range(n+1)}
  for i in range(n+1):v=s.Value(inventory[i][t]);iv[str(i)][str(t+1)]=v;value+=holding[i]*v
  periods[str(t+1)]={'routes':routes,'deliveries':qd,'inventories':inv};rr[str(t+1)]=rmap
 out={'objective_value':value,'solution_details':{'periods':periods}}if old else {'objective_value':value,'routes':rr,'inventories':iv};json.dump(out,open(a.solution_path,'w'))
if __name__=='__main__':main()

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
 os.environ[key]='1'
import time
START=time.monotonic()
import argparse,json
from solution_logger import SolutionLogger

def main():
 parser=argparse.ArgumentParser()
 for key in ('instance_path','solution_path','log_path'):parser.add_argument('--'+key,required=True)
 parser.add_argument('--time_limit',type=int,default=60);a=parser.parse_args();end=START+max(.5,a.time_limit-4);d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path);I=d['num_facilities'];J=d['num_customers'];K=d['num_scenarios'];facilities=d['facilities'];scenarios=d['scenarios'];q=d['transportation_costs'];capacities=[f['capacity'] for f in facilities];fixed=[f['opening_cost'] for f in facilities];probabilities=[s['probability'] for s in scenarios];max_demand=max(sum(s['demands']) for s in scenarios);best=float('inf');ranks=[sorted(range(I),key=lambda i:q[i][j]) for j in range(J)]
 def emit(solution):
  nonlocal best
  if solution['objective_value']>=best:return
  best=solution['objective_value'];json.dump(solution,open(a.solution_path,'w'))
  if hasattr(logger,'log_solution'):logger.log_solution(best,solution)
  else:logger.log(best)
 def greedy(opened):
  if sum(capacities[i] for i in opened)<max_demand-1e-7:return None
  chosen=set(opened);local_ranks=[[i for i in ranks[j] if i in chosen] for j in range(J)];order=sorted(range(J),key=lambda j:(q[local_ranks[j][1]][j]-q[local_ranks[j][0]][j]) if len(local_ranks[j])>1 else float('inf'),reverse=True);y={};costs=[]
  for k,s in enumerate(scenarios):
   remaining_cap=capacities[:];rows={str(i):{} for i in opened};cost=0.
   for j in order:
    demand=s['demands'][j]
    for i in local_ranks[j]:
     quantity=min(demand,remaining_cap[i])
     if quantity>0:rows[str(i)][str(j)]=quantity;cost+=q[i][j]*quantity;remaining_cap[i]-=quantity;demand-=quantity
     if demand<=1e-9:break
    if demand>1e-6:return None
   y[str(k)]=rows;costs.append(cost)
  solution={'objective_value':sum(fixed[i] for i in opened)+sum(probabilities[k]*costs[k] for k in range(K)),'open_facilities':opened,'x':{str(i):int(i in chosen) for i in range(I)},'y':y};return solution,costs
 baseline=greedy(list(range(I)))
 if baseline is None:raise RuntimeError('Total available capacity cannot satisfy all scenario demands')
 emit(baseline[0])
 if time.monotonic()>=end:return
 import numpy as np
 import highspy
 INF=highspy.kHighsInf
 def model():
  h=highspy.Highs();h.setOptionValue('output_flag',False);h.setOptionValue('threads',1);h.setOptionValue('parallel','off');return h
 def facility_model(samples,deadline):
  S=len(samples);N=I+S*I*J;h=model();lower=np.zeros(N);upper=np.full(N,INF);upper[:I]=1;obj=np.empty(N);obj[:I]=fixed
  total_prob=sum(probabilities[k] for k in samples)
  for t,k in enumerate(samples):obj[I+t*I*J:I+(t+1)*I*J]=np.asarray(q).reshape(-1)*(probabilities[k]/total_prob)
  h.addVars(N,lower,upper);h.changeColsCost(N,np.arange(N,dtype=np.int32),obj);h.changeColsIntegrality(I,np.arange(I,dtype=np.int32),np.ones(I,dtype=np.uint8));starts=[0];indices=[];values=[];rl=[];ru=[]
  def row(entries,lo,hi):
   for j,v in entries:indices.append(j);values.append(v)
   starts.append(len(indices));rl.append(lo);ru.append(hi)
  for t,k in enumerate(samples):
   for i in range(I):row([(I+(t*I+i)*J+j,1.) for j in range(J)]+[(i,-capacities[i])],-INF,0.)
   for j in range(J):row([(I+(t*I+i)*J+j,1.) for i in range(I)],scenarios[k]['demands'][j],scenarios[k]['demands'][j])
  row([(i,capacities[i]) for i in range(I)],max_demand,INF)
  h.addRows(len(rl),np.asarray(rl),np.asarray(ru),len(indices),np.asarray(starts,dtype=np.int32),np.asarray(indices,dtype=np.int32),np.asarray(values));seed=np.zeros(N);seed[:I]=1
  for t,k in enumerate(samples):
   for i,rows in baseline[0]['y'][str(k)].items():
    for j,value in rows.items():seed[I+(t*I+int(i))*J+int(j)]=value
  h.setSolution(N,np.arange(N,dtype=np.int32),seed)
  if time.monotonic()>=deadline:return None
  h.setOptionValue('time_limit',max(.05,deadline-time.monotonic()));h.setOptionValue('mip_rel_gap',1e-5);h.run();s=h.getSolution()
  if not s.value_valid:return None
  opened=[i for i in range(I) if s.col_value[i]>.5]
  if sum(capacities[i] for i in opened)<max_demand-1e-6:return None
  return opened,s.col_value
 if I*J*K<=400000:
  result=facility_model(list(range(K)),end-1)
  if result is not None:
   opened,v=result;selected=set(opened);y={};value=sum(fixed[i] for i in opened);valid=True
   for k in range(K):
    rows={str(i):{} for i in opened};customer_totals=[0.]*J
    for i in range(I):
     load=0.
     for j in range(J):
      amount=max(0.,v[I+(k*I+i)*J+j]);load+=amount;customer_totals[j]+=amount;value+=probabilities[k]*q[i][j]*amount
      if amount>1e-9:
       if i not in selected:valid=False
       else:rows[str(i)][str(j)]=amount
     if load>capacities[i]*int(i in selected)+1e-6:valid=False
    if any(abs(customer_totals[j]-scenarios[k]['demands'][j])>1e-6 for j in range(J)):valid=False
    y[str(k)]=rows
   if valid:emit({'objective_value':value,'open_facilities':opened,'x':{str(i):int(i in selected) for i in range(I)},'y':y})
  return
 def optimize_transport(candidate,deadline):
  if candidate is None:return
  solution,costs=candidate;opened=solution['open_facilities'];h=model();h.setOptionValue('solver','simplex');h.setOptionValue('simplex_strategy',1);N=I*J;h.addVars(N,np.zeros(N),np.full(N,INF));h.changeColsCost(N,np.arange(N,dtype=np.int32),np.asarray(q).reshape(-1));starts=[0];indices=[];values=[]
  for i in range(I):
   indices.extend(range(i*J,(i+1)*J));values.extend([1.]*J);starts.append(len(indices))
  for j in range(J):indices.extend(i*J+j for i in range(I));values.extend([1.]*I);starts.append(len(indices))
  lower=np.r_[np.full(I,-INF),np.asarray(scenarios[0]['demands'])];upper=np.r_[[capacities[i]*solution['x'][str(i)] for i in range(I)],np.asarray(scenarios[0]['demands'])];h.addRows(I+J,lower,upper,len(indices),np.asarray(starts,dtype=np.int32),np.asarray(indices,dtype=np.int32),np.asarray(values));demand_rows=np.arange(I,I+J,dtype=np.int32);opening=sum(fixed[i] for i in opened);transport=sum(probabilities[k]*costs[k] for k in range(K))
  for k,s in enumerate(scenarios):
   left=deadline-time.monotonic()
   if left<.2:break
   demands=np.asarray(s['demands']);h.changeRowsBounds(J,demand_rows,demands,demands);h.setOptionValue('time_limit',h.getRunTime()+min(.5,left-.1));h.run()
   if h.getModelStatus()!=highspy.HighsModelStatus.kOptimal:continue
   flow=np.asarray(h.getSolution().col_value).reshape(I,J)
   if np.any(flow.sum(axis=0)<demands-1e-7) or np.any(flow.sum(axis=1)>upper[:I]+1e-7):continue
   new_cost=float((flow*np.asarray(q)).sum());rows={str(i):{str(j):float(flow[i,j]) for j in range(J) if flow[i,j]>1e-9} for i in opened};solution['y'][str(k)]=rows;transport+=probabilities[k]*(new_cost-costs[k]);costs[k]=new_cost;solution['objective_value']=opening+transport
   if (k+1)%500==0:emit(solution)
   if k==0:h.setOptionValue('presolve','off')
  solution['objective_value']=opening+sum(probabilities[k]*costs[k] for k in range(K));emit(solution)
 optimize_transport(baseline,min(end-1,START+a.time_limit*.43))
 if time.monotonic()>=end-2:return
 order=sorted(range(K),key=lambda k:sum(scenarios[k]['demands']));samples=sorted({order[round(t*(K-1)/11)] for t in range(12)});selected=facility_model(samples,min(end-2,time.monotonic()+a.time_limit*.23))
 if selected is not None and selected[0]!=list(range(I)) and time.monotonic()<end-1:
  candidate=greedy(selected[0])
  if candidate is not None:emit(candidate[0]);optimize_transport(candidate,end-1)
if __name__=='__main__':main()

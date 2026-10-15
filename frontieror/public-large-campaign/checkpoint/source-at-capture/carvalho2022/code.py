import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random,math,collections,itertools
from ortools.sat.python import cp_model

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));n=d['dimensions']['n'];K=d['dimensions']['m'];T=d['dimensions']['p'];model=cp_model.CpModel();x={};z={};y={};G={};obj=[];hints={};best_constructed=None;rng=random.Random(2022);construct_end=start+min(16 if n>12 else 4,float(a.time_limit)*.3);plans=[]
 if n>12:
  approximation_end=start+min(8,float(a.time_limit)*.15);maximum_setup=max(d['setup_times'][i][j][k]for i in range(n)for j in range(n)for k in range(K))
  for factor in [.7,1.2,2,3]:
   if time.monotonic()>approximation_end:break
   approximation=cp_model.CpModel();quantities={};activation={};approx_objective=[]
   for i in range(n):
    for k in range(K):
     for t in range(T):
      if d['max_setups_per_item'][i][k][t]<=0:continue
      active=approximation.NewBoolVar('a%d_%d_%d'%(i,k,t));activation[i,k,t]=active;total=[];incoming=min(d['setup_costs'][j][i][k]for j in range(n)if j!=i and d['max_setups_per_item'][j][k][t]>0);outgoing=min(d['setup_costs'][i][j][k]for j in range(n)if j!=i and d['max_setups_per_item'][j][k][t]>0);approx_objective.append(int(round(factor*(incoming+outgoing)))*active)
      for u in range(t,T):
       q=approximation.NewIntVar(0,d['demands'][i][u],'p%d_%d_%d_%d'%(i,k,t,u));quantities[i,k,t,u]=q;total.append(q);approx_objective.append((u-t)*d['inventory_costs'][i]*q)
      approximation.Add(sum(total)<=sum(d['demands'][i][t:])*active);approximation.Add(sum(total)>=max(1,d['minimum_lot_sizes'][i])*active)
   for i in range(n):
    for u in range(T):approximation.Add(sum(q for(ii,k,t,uu),q in quantities.items()if ii==i and uu==u)==d['demands'][i][u])
   for k in range(K):
    for t in range(T):approximation.Add(d['processing_time']*sum(q for(i,kk,tt,u),q in quantities.items()if kk==k and tt==t)+maximum_setup*sum(v for(i,kk,tt),v in activation.items()if kk==k and tt==t)<=d['machine_capacities'][k][t])
   initial={};initial_active=set();initial_ok=True
   for t in range(T):
    load=[0]*K
    for i in sorted(range(n),key=lambda i:sum((i,k,t)in activation for k in range(K))):
     required=d['processing_time']*d['demands'][i][t]+maximum_setup;eligible=[k for k in range(K)if(i,k,t)in activation and load[k]+required<=d['machine_capacities'][k][t]]
     if not eligible:initial_ok=False;break
     k=min(eligible,key=lambda k:load[k]/d['machine_capacities'][k][t]);load[k]+=required;initial[i,k,t,t]=d['demands'][i][t];initial_active.add((i,k,t))
    if not initial_ok:break
   if initial_ok:
    for key,q in quantities.items():approximation.AddHint(q,initial.get(key,0))
    for key,v in activation.items():approximation.AddHint(v,int(key in initial_active))
   approximation.Minimize(sum(approx_objective));asolver=cp_model.CpSolver();asolver.parameters.num_search_workers=1;asolver.parameters.max_time_in_seconds=max(.05,min(2,approximation_end-time.monotonic()));astatus=asolver.Solve(approximation)
   if astatus in(cp_model.FEASIBLE,cp_model.OPTIMAL):plans.append({key:asolver.Value(q)for key,q in quantities.items()if asolver.Value(q)})
 edge_cache={}
 def sequence(row,k,t,budget):
  expanded=[row[0]];cost=0;duration=0
  for i,j in zip(row,row[1:]):
   key=i,j,k,t
   if key not in edge_cache:
    shortcuts=[v for v in range(n)if v not in(i,j)and d['minimum_lot_sizes'][v]==0 and d['max_setups_per_item'][v][k][t]>0];options=[]
    for length in range(3):
     for middle in itertools.permutations(shortcuts,length):
      chain=(i,)+middle+(j,);edge_time=sum(d['setup_times'][u][v][k]for u,v in zip(chain,chain[1:]))
      if edge_time<=d['setup_times'][i][j][k]:options.append((sum(d['setup_costs'][u][v][k]for u,v in zip(chain,chain[1:])),edge_time,middle))
    edge_cache[key]=min(options)
   c,b,middle=edge_cache[key];cost+=c;duration+=b;expanded.extend(middle);expanded.append(j)
  counts=collections.Counter(expanded)
  if duration>budget or any(counts[v]>d['max_setups_per_item'][v][k][t]for v in counts):return None
  return cost,expanded
 def improve_sequence(row,k,t,budget):
  current=sequence(row,k,t,budget)
  if current is None:return row
  while time.monotonic()<construct_end:
   best=current;best_row=row
   for left in range(1,len(row)):
    for right in range(1,len(row)):
     if left==right:continue
     candidate=row[:];item=candidate.pop(left);candidate.insert(right,item);result=sequence(candidate,k,t,budget)
     if result is not None and result[0]<best[0]:best=result;best_row=candidate
   if best_row is row:break
   row=best_row;current=best
  return current[1]
 for attempt in range(500 if n>12 else 100):
  xp={};yp={};zp={};paths={};carry=[rng.choice([i for i in range(n)if d['max_setups_per_item'][i][k][0]>0])for k in range(K)];valid=True;value=0;remaining=[list(row)for row in d['demands']];planned=plans[(attempt//2)%len(plans)]if plans and attempt%2 else None;batch_factor=[0,.7,1,1.4,2,3,5][attempt%7]if n>12 else 0
  for t in range(T):
   routes=[[carry[k]]for k in range(K)];loads=[0]*K
   for k in range(K):zp[carry[k],k,t]=1
   order=sorted([(i,None)for i in range(n)]if planned is None else list({(i,k)for(i,k,tt,u)in planned if tt==t}),key=lambda job:(sum(d['max_setups_per_item'][job[0]][k][t]>0 for k in range(K)),rng.random()))
   for i,assigned in order:
    qty=remaining[i][t]if planned is None else sum(v for(ii,k,tt,u),v in planned.items()if ii==i and k==assigned and tt==t);options=[]
    if qty==0:continue
    for k in range(K):
     if assigned is not None and assigned!=k:continue
     if d['max_setups_per_item'][i][k][t]<=0:continue
     if i==routes[k][0]:
      if loads[k]+d['processing_time']*qty<=d['machine_capacities'][k][t]:options.append((0,k,0,0))
      continue
     for pos in range(1,len(routes[k])+1):
      prev=routes[k][pos-1];nextitem=routes[k][pos]if pos<len(routes[k])else None;extra=d['setup_times'][prev][i][k]+(d['setup_times'][i][nextitem][k]-d['setup_times'][prev][nextitem][k]if nextitem is not None else 0);cost=d['setup_costs'][prev][i][k]+(d['setup_costs'][i][nextitem][k]-d['setup_costs'][prev][nextitem][k]if nextitem is not None else 0)
      if n>12 and attempt%3:
       incoming=sequence([prev,i],k,t,float('inf'));outgoing=sequence([i,nextitem],k,t,float('inf'))if nextitem is not None else None;replaced=sequence([prev,nextitem],k,t,float('inf'))if nextitem is not None else None
       cost=incoming[0]+(outgoing[0]-replaced[0]if nextitem is not None else 0)
      if loads[k]+extra+d['processing_time']*qty<=d['machine_capacities'][k][t]:options.append((cost*(1+rng.random()*.15),k,pos,extra))
    if not options:valid=False;break
    score,k,pos,extra=min(options);loads[k]+=extra+d['processing_time']*qty
    if planned is None:xp[i,k,t,t]=qty;remaining[i][t]=0
    else:
     for(ii,kk,tt,u),amount in planned.items():
      if ii==i and kk==k and tt==t:xp[i,k,t,u]=amount;value+=(u-t)*d['inventory_costs'][i]*amount
    if pos:routes[k].insert(pos,i)
   if not valid:break
   if batch_factor and planned is None:
    extra_options=[]
    for k,row in enumerate(routes):
     for pos,i in enumerate(row):
      if (i,k,t,t)not in xp:continue
      before=row[pos-1]if pos else None;after=row[pos+1]if pos+1<len(row)else None;saved=(d['setup_costs'][before][i][k]if before is not None else 0)+(d['setup_costs'][i][after][k]if after is not None else 0)-(d['setup_costs'][before][after][k]if before is not None and after is not None else 0)
      if n>12 and attempt%3:
       saved=(sequence([before,i],k,t,float('inf'))[0]if before is not None else 0)+(sequence([i,after],k,t,float('inf'))[0]if after is not None else 0)-(sequence([before,after],k,t,float('inf'))[0]if before is not None and after is not None else 0)
      if saved<=0:saved=max(1,d['parameters']['Theta'])
      for u in range(t+1,T):
       qty=remaining[i][u];holding=(u-t)*d['inventory_costs'][i]*qty
       if qty and holding<=saved*batch_factor:extra_options.append((holding/max(1,saved)+rng.random()*.1,k,i,u,qty))
    for score,k,i,u,qty in sorted(extra_options):
     if remaining[i][u]and loads[k]+d['processing_time']*qty<=d['machine_capacities'][k][t]:xp[i,k,t,u]=qty;remaining[i][u]=0;loads[k]+=d['processing_time']*qty;value+=(u-t)*d['inventory_costs'][i]*qty
   for k,row in enumerate(routes):
    if n>12:
     budget=d['machine_capacities'][k][t]-d['processing_time']*sum(amount for(i,kk,tt,u),amount in xp.items()if kk==k and tt==t)
     row=improve_sequence(row,k,t,budget)
    paths[k,t]=row;carry[k]=row[-1]
    for i,j in zip(row,row[1:]):yp[i,j,k,t]=yp.get((i,j,k,t),0)+1;value+=d['setup_costs'][i][j][k]
  if valid:
   for k in range(K):zp[carry[k],k,T]=1
   if best_constructed is None or value<best_constructed[0]:best_constructed=(value,xp,yp,zp,paths)
  if time.monotonic()>construct_end:break
 if best_constructed is not None:
  value,xp,yp,zp,paths=best_constructed
  for k in range(K):
   for t in range(T+1):
    for i in range(n):hints['z%d_%d_%d'%(i,k,t)]=zp.get((i,k,t),0)
   for t in range(T):
    row=paths[k,t];active=set(row);seen={row[0]};parents={};discovery=[row[0]]
    for i,j in zip(row,row[1:]):
     if j not in seen:parents[j]=i;seen.add(j);discovery.append(j)
    sizes={i:1 for i in active};flow={}
    for j in reversed(discovery[1:]):flow[parents[j],j]=sizes[j];sizes[parents[j]]+=sizes[j]
    for i in range(n):
     hints['g%d_%d_%d'%(i,k,t)]=int(i in active);hints['root%d_%d_%d'%(i,k,t)]=len(active)if i==row[0]else 0
     for u in range(t,T):hints['x%d_%d_%d_%d'%(i,k,t,u)]=xp.get((i,k,t,u),0)
     for j in range(n):
      if i==j:continue
      hints['y%d_%d_%d_%d'%(i,j,k,t)]=yp.get((i,j,k,t),0);hints['f%d_%d_%d_%d'%(i,j,k,t)]=flow.get((i,j),0)
 for k in range(K):
  for t in range(T+1):
   for i in range(n):z[i,k,t]=model.NewBoolVar('z%d_%d_%d'%(i,k,t))
   model.Add(sum(z[i,k,t]for i in range(n))==1)
 for k in range(K):
  for t in range(T):
   f={};root={};
   for i in range(n):
    G[i,k,t]=model.NewBoolVar('g%d_%d_%d'%(i,k,t));root[i]=model.NewIntVar(0,n,'root%d_%d_%d'%(i,k,t));model.Add(root[i]<=n*z[i,k,t])
    for j in range(n):
     if i==j:continue
     yy=model.NewIntVar(0,min(1,d['max_setups_per_item'][j][k][t])if n<=12 else d['max_setups_per_item'][j][k][t],'y%d_%d_%d_%d'%(i,j,k,t));y[i,j,k,t]=yy;ff=model.NewIntVar(0,n,'f%d_%d_%d_%d'%(i,j,k,t));f[i,j]=ff;model.Add(ff<=n*yy);obj.append(d['setup_costs'][i][j][k]*yy)
   for i in range(n):
    incoming=sum(y[j,i,k,t]for j in range(n)if j!=i);outgoing=sum(y[i,j,k,t]for j in range(n)if j!=i);model.Add(z[i,k,t]+incoming==outgoing+z[i,k,t+1]);model.Add(G[i,k,t]==z[i,k,t]+incoming)if n<=12 else model.Add(G[i,k,t]<=z[i,k,t]+incoming);model.Add(z[i,k,t]+incoming<=d['max_setups_per_item'][i][k][t]*G[i,k,t]);model.Add(G[i,k,t]<=d['max_setups_per_item'][i][k][t]);model.Add(root[i]+sum(f[j,i]for j in range(n)if j!=i)==G[i,k,t]+sum(f[i,j]for j in range(n)if j!=i));prod=[]
    for u in range(t,T):
     xx=model.NewIntVar(0,d['demands'][i][u],'x%d_%d_%d_%d'%(i,k,t,u));x[i,k,t,u]=xx;model.Add(xx<=d['max_production'][i][t]*G[i,k,t]);prod.append(xx);obj.append((u-t)*d['inventory_costs'][i]*xx)
    model.Add(sum(prod)>=d['minimum_lot_sizes'][i]*incoming)
   model.Add(sum(root.values())==sum(G[i,k,t]for i in range(n)));model.Add(d['processing_time']*sum(x[i,k,t,u]for i in range(n)for u in range(t,T))+sum(d['setup_times'][i][j][k]*yy for(i,j,kk,tt),yy in y.items()if kk==k and tt==t)<=d['machine_capacities'][k][t])
 for i in range(n):
  for u in range(T):model.Add(sum(x[i,k,t,u]for k in range(K)for t in range(u+1))==d['demands'][i][u])
 if best_constructed is not None:
  for v in list(x.values())+list(y.values())+list(z.values())+list(G.values()):model.AddHint(v,hints.get(v.Name(),0))
  for index,var in enumerate(model.Proto().variables):
   if var.name.startswith(('root','f')):model.AddHint(model.GetIntVarFromProtoIndex(index),hints.get(var.name,0))
 if best_constructed is not None:model.Add(sum(obj)<=best_constructed[0])
 model.Minimize(sum(obj));s=cp_model.CpSolver();s.parameters.num_search_workers=1;remaining_time=max(.1,float(a.time_limit)-2-(time.monotonic()-start));s.parameters.max_time_in_seconds=min(6,remaining_time)if n>12 and best_constructed is not None else remaining_time;status=s.Solve(model)
 if n>12 and best_constructed is not None:
  incumbent_value=best_constructed[0];incumbent={index:hints.get(var.name,0)for index,var in enumerate(model.Proto().variables)}
  if status in(cp_model.FEASIBLE,cp_model.OPTIMAL)and s.ObjectiveValue()<=incumbent_value:incumbent_value=s.ObjectiveValue();incumbent={index:s.Value(model.GetIntVarFromProtoIndex(index))for index in range(len(model.Proto().variables))}
  else:status=cp_model.UNKNOWN
  priority=sorted(range(T),key=lambda t:sum(d['setup_costs'][i][j][k]*incumbent[v.Index()]for(i,j,k,tt),v in y.items()if tt==t),reverse=True)
  sparse_edges=set()
  for k in range(K):
   for t in range(T):
    eligible=[i for i in range(n)if d['max_setups_per_item'][i][k][t]>0]
    for i in eligible:
     others=[j for j in eligible if j!=i]
     for j in sorted(others,key=lambda j:d['setup_costs'][i][j][k])[:2]+sorted(others,key=lambda j:d['setup_times'][i][j][k])[:1]:sparse_edges.add((i,j,k,t))
     for j in sorted(others,key=lambda j:d['setup_costs'][j][i][k])[:2]+sorted(others,key=lambda j:d['setup_times'][j][i][k])[:1]:sparse_edges.add((j,i,k,t))
  neighborhoods=[(0,T,8)]+[(t,t+1,2)for t in priority]+[(t,min(T,t+3),5)for t in priority]
  for first,last,seconds in neighborhoods:
   available=float(a.time_limit)-2-(time.monotonic()-start)
   if available<.2:break
   local=model.Clone();local.ClearHints()
   for index,var in enumerate(model.Proto().variables):
    v=local.GetIntVarFromProtoIndex(index);local.AddHint(v,incumbent[index]);parts=var.name.split('_');period=int(parts[-1])
    if var.name.startswith('z'):
     if period<first or period>last:local.Add(v==incumbent[index])
    elif var.name.startswith(('y','g','f','root')):
     if period<first or period>=last:local.Add(v==incumbent[index])
   for key,var in y.items():
    if first<=key[-1]<last and key not in sparse_edges and incumbent[var.Index()]==0:local.Add(local.GetIntVarFromProtoIndex(var.Index())==0)
   local.Add(sum(obj)<=int(round(incumbent_value)));candidate_solver=cp_model.CpSolver();candidate_solver.parameters.num_search_workers=1;candidate_solver.parameters.max_time_in_seconds=min(seconds,available);candidate_status=candidate_solver.Solve(local)
   if candidate_status in(cp_model.FEASIBLE,cp_model.OPTIMAL)and candidate_solver.ObjectiveValue()<incumbent_value:
    incumbent_value=candidate_solver.ObjectiveValue();incumbent={index:candidate_solver.Value(local.GetIntVarFromProtoIndex(index))for index in range(len(model.Proto().variables))};s=candidate_solver;status=candidate_status
 if status not in(cp_model.FEASIBLE,cp_model.OPTIMAL):
  if best_constructed is None:raise RuntimeError('No feasible sequence-dependent lot-sizing plan found')
  value,xp,yp,zp,paths=best_constructed;json.dump({'objective_value':value,'production':{'x_%d_%d_%d_%d'%key:v for key,v in xp.items()if v},'setups':{'y_%d_%d_%d_%d'%key:v for key,v in yp.items()if v},'carryover':{'z_%d_%d_%d'%key:v for key,v in zp.items()if v}},open(a.solution_path,'w'));return
 production={'x_%d_%d_%d_%d'%key:s.Value(v)for key,v in x.items()if s.Value(v)};setups={'y_%d_%d_%d_%d'%key:s.Value(v)for key,v in y.items()if s.Value(v)};carry={'z_%d_%d_%d'%key:s.Value(v)for key,v in z.items()if s.Value(v)};value=sum((u-t)*d['inventory_costs'][i]*s.Value(v)for(i,k,t,u),v in x.items())+sum(d['setup_costs'][i][j][k]*s.Value(v)for(i,j,k,t),v in y.items());json.dump({'objective_value':value,'production':production,'setups':setups,'carryover':carry},open(a.solution_path,'w'))
if __name__=='__main__':main()

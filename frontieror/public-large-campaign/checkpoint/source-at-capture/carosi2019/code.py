import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,itertools,random,math
import numpy as np
from scipy.optimize import linear_sum_assignment

def main():
 parser=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):parser.add_argument('--'+k,required=k!='log_path')
 a=parser.parse_args();deadline=time.monotonic()+max(.1,float(a.time_limit)-2);d=json.load(open(a.instance_path));trips=d['potential_trips'];N=len(trips);windows={v['index']:v for v in d['time_windows']};alpha=d['objective_function']['alpha'];deploy=d.get('M',d['objective_function'].get('M',10000));params={}
 def terminal(term,tw):
  key=(term,tw)
  if key not in params:
   st=next((v for v in d['stopping_times'].get(term,[])if v['time_window_index']==tw),{});pull=next((v for v in d['pull_in_out_times'].get(term,[])if v['time_window_index']==tw),{});params[key]=(st.get('min_stopping_time_minutes',0),st.get('max_stopping_time_minutes',9999),pull.get('pull_out_time_minutes',0),pull.get('pull_in_time_minutes',0))
  return params[key]
 departures=np.array([v['departure_time_minutes']for v in trips]);starts=np.array([v['start_terminal']for v in trips]);outs=np.array([terminal(v['start_terminal'],v['time_window_index'])[2]for v in trips]);ins=np.array([terminal(v['end_terminal'],v['time_window_index'])[3]for v in trips]);savings=np.full((N,N),1e8,dtype=np.float32)
 for i,v in enumerate(trips):
  mn,mx,po,pi=terminal(v['end_terminal'],v['time_window_index']);gap=departures-v['arrival_time_minutes'];same=starts==v['end_terminal'];inline=same&(gap>=mn)&(gap<=mx);outline=(~same)&(gap>=pi+terminal('O',v['time_window_index'])[0]+outs);savings[i,inline]=gap[inline]-mn-deploy-pi-outs[inline];savings[i,outline]=pi+outs[outline]-deploy-pi-outs[outline]
 alternatives=[]
 for direction in d['directions']:
  pattern=direction['pattern_id'];ids=sorted([i for i,v in enumerate(trips)if v['pattern_id']==pattern],key=lambda i:trips[i]['main_stop_arrival_time_minutes']);pred={i:[]for i in ids}
  for i in ids:
   v=trips[i];win=windows[v['time_window_index']]
   for j in ids:
    hw=trips[j]['main_stop_arrival_time_minutes']-v['main_stop_arrival_time_minutes']
    if j!=i and win['min_headway_seconds']/60<=hw<=win['max_headway_seconds']/60:pred[j].append((i,(hw-win['ideal_headway_seconds']/60)**2))
  candidates={}
  for charge,target in [(c,None)for c in [0,20,80,240,640,1600,5000]]+[(0,t)for t in sorted({trips[i]['arrival_time_minutes']-trips[i]['departure_time_minutes']+wait for i in ids for wait in range(int(terminal(trips[i]['end_terminal'],trips[i]['time_window_index'])[0]),int(terminal(trips[i]['end_terminal'],trips[i]['time_window_index'])[1])+1)})]:
   dp={};previous={}
   for j in ids:
    options=[(dp[i]+(cost if target is None else 100*(trips[j]['main_stop_arrival_time_minutes']-trips[i]['main_stop_arrival_time_minutes']-target)**2)+charge,i)for i,cost in pred[j]if i in dp]
    if trips[j]['time_window_index']==d['initial_trip_time_windows'][pattern]:options.append((charge,-1))
    if options:dp[j],previous[j]=min(options)
   ends=[j for j in ids if j in dp and trips[j]['time_window_index']==d['final_trip_time_windows'][pattern]]
   for j in sorted(ends,key=lambda j:dp[j])[:4]:
    row=[]
    while j!=-1:row.append(j);j=previous[j]
    row=tuple(reversed(row));penalty=sum((trips[j]['main_stop_arrival_time_minutes']-trips[i]['main_stop_arrival_time_minutes']-windows[trips[i]['time_window_index']]['ideal_headway_seconds']/60)**2 for i,j in zip(row,row[1:]));candidates[row]=penalty
  if not candidates:raise RuntimeError('No headway feasible timetable chain')
  if len(d['directions'])>2:
   bytime={trips[i]['departure_time_minutes']:i for i in ids};times=sorted(bytime);step=min(b-a for a,b in zip(times,times[1:])if b>a)
   for row in list(candidates):
    for offset in [-3,-2,-1,1,2,3]:
     shifted=tuple(bytime.get(trips[i]['departure_time_minutes']+offset*step,-1)for i in row)
     if -1 in shifted:continue
     if trips[shifted[0]]['time_window_index']!=d['initial_trip_time_windows'][pattern]or trips[shifted[-1]]['time_window_index']!=d['final_trip_time_windows'][pattern]:continue
     if all(windows[trips[i]['time_window_index']]['min_headway_seconds']/60<=trips[j]['main_stop_arrival_time_minutes']-trips[i]['main_stop_arrival_time_minutes']<=windows[trips[i]['time_window_index']]['max_headway_seconds']/60 for i,j in zip(shifted,shifted[1:])):
      candidates[shifted]=sum((trips[j]['main_stop_arrival_time_minutes']-trips[i]['main_stop_arrival_time_minutes']-windows[trips[i]['time_window_index']]['ideal_headway_seconds']/60)**2 for i,j in zip(shifted,shifted[1:]))
  alternatives.append((pattern,candidates))
 def evaluate(rows):
  chosen=[i for row in rows for i in row];n=len(chosen);matrix=np.zeros((n,2*n));matrix[:,:n]=savings[np.ix_(chosen,chosen)];base=float(np.sum(deploy+outs[chosen]+ins[chosen]));rr,cc=linear_sum_assignment(matrix);valid=(cc<n)&(matrix[rr,cc]<0);links=[(chosen[r],chosen[c])for r,c in zip(rr[valid],cc[valid])];fleet=n-len(links)
  if fleet>d['objective_function'].get('max_fleet',N):return float('inf'),None
  combination=[]
  for row in rows:
   pen=sum((trips[j]['main_stop_arrival_time_minutes']-trips[i]['main_stop_arrival_time_minutes']-windows[trips[i]['time_window_index']]['ideal_headway_seconds']/60)**2 for i,j in zip(row,row[1:]));combination.append((tuple(row),pen))
  value=alpha*(base+float(matrix[rr[valid],cc[valid]].sum()))+sum(p for row,p in combination)
  return value,(chosen,links,fleet,combination)
 best=None;bestcost=float('inf');pools=[list(v)for pattern,v in alternatives];initial_end=min(deadline-25,time.monotonic()+(5 if len(pools)>2 else 15))
 combinations=itertools.product(*pools)if len(pools)<=2 else (tuple(pool[k%len(pool)]for pool in pools)for k in range(max(map(len,pools))))
 for rows in combinations:
  value,result=evaluate(rows)
  if value<bestcost:bestcost=value;best=result
  if time.monotonic()>initial_end:break
 if best is None:raise RuntimeError('No compatible fleet timetable found')
 rng=random.Random(931)
 if len(pools)>2:
  active_windows=[w for w in windows.values()if w['max_headway_seconds']>0];minimum=max(w['min_headway_seconds']/60 for w in active_windows);maximum=min(w['max_headway_seconds']/60 for w in active_windows);directions=d['directions'];maps=[{trips[i]['departure_time_minutes']:i for i in range(N)if trips[i]['pattern_id']==v['pattern_id']}for v in directions];grid=min(b-a for mapping in maps for a,b in zip(sorted(mapping),sorted(mapping)[1:])if b>a);cadences=[grid*k for k in range(math.ceil(minimum/grid),math.floor(maximum/grid)+1)];cadences=sorted(cadences,reverse=True)[:6]
  for cadence in cadences:
   for phase in np.arange(0,cadence,grid):
    trial=[]
    for v,mapping in zip(directions,maps):
     pat=v['pattern_id'];reverse=v['start_terminal']>v['end_terminal'];offset=float(phase)if reverse else 0.;initial=[t for t,i in mapping.items()if trips[i]['time_window_index']==d['initial_trip_time_windows'][pat]];start=max(initial);start-=((start-offset)%cadence);row=[];clock=start
     while clock in mapping:
      i=mapping[clock];row.append(i)
      if trips[i]['time_window_index']==d['final_trip_time_windows'][pat]:break
      clock+=cadence
     if not row or trips[row[0]]['time_window_index']!=d['initial_trip_time_windows'][pat]or trips[row[-1]]['time_window_index']!=d['final_trip_time_windows'][pat]:break
     if not all(windows[trips[i]['time_window_index']]['min_headway_seconds']/60<=trips[j]['main_stop_arrival_time_minutes']-trips[i]['main_stop_arrival_time_minutes']<=windows[trips[i]['time_window_index']]['max_headway_seconds']/60 for i,j in zip(row,row[1:])):break
     trial.append(row)
    if len(trial)==len(directions):
     value,result=evaluate(trial)
     if value<bestcost:bestcost=value;best=result
    if time.monotonic()>deadline-30:break
   if time.monotonic()>deadline-30:break
  for sweep in range(4):
   improved=False;rows=[list(row)for row,pen in best[3]]
   for r,pool in enumerate(pools):
    for candidate in pool:
     if time.monotonic()>deadline-30:break
     trial=rows[:];trial[r]=candidate;value,result=evaluate(trial)
     if value<bestcost:bestcost=value;best=result;rows=trial;improved=True
   if not improved or time.monotonic()>deadline-30:break
 rows=[list(row)for row,pen in best[3]];current=bestcost;iteration=0
 bypattern={pat:[i for i,t in enumerate(trips)if t['pattern_id']==pat]for pat,_ in alternatives}
 while time.monotonic()<deadline-(24 if len(pools)>2 else 0):
  iteration+=1;r=rng.randrange(len(rows));row=rows[r];pos=rng.randrange(len(row));choices=bypattern[alternatives[r][0]];new=rng.choice(choices)
  if new in row:continue
  candidate=row[:];candidate[pos]=new
  if pos==0 and trips[new]['time_window_index']!=d['initial_trip_time_windows'][alternatives[r][0]]:continue
  if pos==len(row)-1 and trips[new]['time_window_index']!=d['final_trip_time_windows'][alternatives[r][0]]:continue
  valid=True
  for i,j in zip(candidate[max(0,pos-1):pos+1],candidate[max(0,pos-1)+1:pos+2]):
   win=windows[trips[i]['time_window_index']];hw=trips[j]['main_stop_arrival_time_minutes']-trips[i]['main_stop_arrival_time_minutes']
   if not win['min_headway_seconds']/60<=hw<=win['max_headway_seconds']/60:valid=False;break
  if not valid:continue
  trial=rows[:];trial[r]=candidate;value,result=evaluate(trial);temperature=max(5,alpha*deploy*.12*(1-(iteration%4000)/4000))
  if value<current or rng.random()<math.exp(min(0,(current-value)/temperature)):rows=trial;current=value
  if value<bestcost:bestcost=value;best=result
  if iteration%4000==0:rows=[list(row)for row,pen in best[3]];current=bestcost
 if len(pools)>2 and time.monotonic()<deadline-2:
  from ortools.sat.python import cp_model
  model=cp_model.CpModel();slotrows=[];selected={};objective=[];scale=1000;incumbent=set(best[0]);oldlinks=set(best[1]);ttvariables=[]
  for r,(row,pen)in enumerate(best[3]):
   pat=alternatives[r][0];ids=bypattern[pat];bytime={trips[i]['departure_time_minutes']:i for i in ids};times=sorted(bytime);step=min(b-a for a,b in zip(times,times[1:])if b>a);slots=[]
   for position,i in enumerate(row):
    choices=[bytime.get(trips[i]['departure_time_minutes']+offset*step,-1)for offset in [-1,0,1]];choices=[j for j in choices if j!=-1 and(position!=0 or trips[j]['time_window_index']==d['initial_trip_time_windows'][pat])and(position!=len(row)-1 or trips[j]['time_window_index']==d['final_trip_time_windows'][pat])];slots.append(choices)
    for j in choices:
     if j not in selected:selected[j]=model.NewBoolVar('trip%d'%j);model.AddHint(selected[j],int(j in incumbent));objective.append(int(round(scale*alpha*(deploy+outs[j]+ins[j])))*selected[j])
    model.Add(sum(selected[j]for j in choices)==1)
   for left,right in zip(slots,slots[1:]):
    outgoing={i:[]for i in left};incoming={j:[]for j in right}
    for i in left:
     window=windows[trips[i]['time_window_index']]
     for j in right:
      headway=trips[j]['main_stop_arrival_time_minutes']-trips[i]['main_stop_arrival_time_minutes']
      if window['min_headway_seconds']/60<=headway<=window['max_headway_seconds']/60:
       z=model.NewBoolVar('tt%d_%d'%(i,j));outgoing[i].append(z);incoming[j].append(z);objective.append(int(round(scale*(headway-window['ideal_headway_seconds']/60)**2))*z);model.AddHint(z,int(i in incumbent and j in incumbent))
    for i,zz in outgoing.items():model.Add(sum(zz)==selected[i])
    for j,zz in incoming.items():model.Add(sum(zz)==selected[j])
   slotrows.append(slots)
  ids=list(selected);incoming={i:[]for i in ids};outgoing={i:[]for i in ids};matching=[];maxgap=max(v['max_headway_seconds']/60 for v in windows.values())*4+max(t['arrival_time_minutes']-t['departure_time_minutes']for t in trips)
  for i in ids:
   candidates=np.array(ids);mask=(savings[i,candidates]<0)&(departures[candidates]-trips[i]['arrival_time_minutes']<=maxgap)
   choices=set(candidates[mask].tolist());choices.update(j for ii,j in oldlinks if ii==i)
   for j in choices:
    z=model.NewBoolVar('vs%d_%d'%(i,j));outgoing[i].append(z);incoming[j].append(z);matching.append(z);objective.append(int(round(scale*alpha*float(savings[i,j])))*z);model.AddHint(z,int((i,j)in oldlinks))
  for i in ids:model.Add(sum(outgoing[i])<=selected[i]);model.Add(sum(incoming[i])<=selected[i])
  model.Add(sum(selected.values())-sum(matching)<=d['objective_function'].get('max_fleet',N));model.Minimize(sum(objective));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());status=solver.Solve(model)
  if status in(cp_model.FEASIBLE,cp_model.OPTIMAL):
   candidate=[[next(j for j in choices if solver.Value(selected[j]))for choices in slots]for slots in slotrows];value,result=evaluate(candidate)
   if value<bestcost:bestcost=value;best=result
 chosen,links,fleet,combination=best;incoming={j for i,j in links};outgoing={i for i,j in links};flows={};ttout={}
 def add(u,v):flows[str(u)+'-->'+str(v)]=1
 for i in chosen:
  tid=trips[i]['id'];add((tid,'start'),(tid,'end'))
  if i not in incoming:add('VS_O_minus',(tid,'start'))
  if i not in outgoing:add((tid,'end'),'VS_O_plus')
 for i,j in links:add((trips[i]['id'],'end'),(trips[j]['id'],'start'))
 for(pattern,options),(row,pen)in zip(alternatives,combination):
  nodes=['source']+[str(trips[i]['id'])for i in row]+['sink'];ttout[pattern]=[[i,j]for i,j in zip(nodes,nodes[1:])]
 flows['VS_O_plus-->VS_O_minus']=fleet;json.dump({'objective_value':float(bestcost),'num_vehicles':fleet,'selected_trips':[trips[i]['id']for i in chosen],'tt_arcs_used':ttout,'vs_flows':flows},open(a.solution_path,'w'))
if __name__=='__main__':main()

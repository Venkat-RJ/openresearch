import argparse,json,time,random,bisect,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linear_sum_assignment
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args()
 deadline=time.monotonic()+max(.1,a.time_limit-2);d=json.load(open(a.instance_path));rng=random.Random(0)
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 trips=d['potential_trips'];hw={t['index']:(t['ideal_headway_seconds']/60,t['min_headway_seconds']/60,t['max_headway_seconds']/60) for t in d['time_windows']}
 stop={(t,e['time_window_index']):(e['min_stopping_time_minutes'],e['max_stopping_time_minutes']) for t,es in d['stopping_times'].items() for e in es}
 pull={(t,e['time_window_index']):(e['pull_out_time_minutes'],e['pull_in_time_minutes']) for t,es in d['pull_in_out_times'].items() for e in es}
 alpha=d.get('objective_function',{}).get('alpha',1);M=d.get('M',d.get('objective_function',{}).get('M',10000));fleet=d.get('max_fleet',d.get('objective_function',{}).get('max_fleet'))
 graphs={}
 for dr in d['directions']:
  key=dr['pattern_id'];ts=sorted([t for t in trips if t['pattern_id']==key],key=lambda t:t['main_stop_arrival_time_minutes']);times=[t['main_stop_arrival_time_minutes'] for t in ts];pred=[[] for t in ts]
  for i,t in enumerate(ts):
   ideal,lo,hi=hw[t['time_window_index']]
   for j in range(bisect.bisect_left(times,times[i]+lo,i+1),bisect.bisect_right(times,times[i]+hi,i+1)):
    pred[j].append((i,(times[j]-times[i]-ideal)**2))
  init=d.get('initial_trip_time_windows',{}).get(key);fin=d.get('final_trip_time_windows',{}).get(key)
  graphs[key]=(ts,pred,[i for i,t in enumerate(ts) if init is None or t['time_window_index']==init],[i for i,t in enumerate(ts) if fin is None or t['time_window_index']==fin])
 best=float('inf');best_solution=None;it=0
 while it<36 and time.monotonic()<deadline:
  selected=[];tt={};penalty=[0,10,30,100,300,1000][it%6];ttcost=0
  for key,(ts,pred,initial,final) in graphs.items():
   costs=[float('inf')]*len(ts);prev=[-1]*len(ts);initialset=set(initial)
   for j,t in enumerate(ts):
    noise=(rng.random()-.5)*([0,1,5,20,50][(it//6)%5])
    if j in initialset:costs[j]=penalty+noise
    for i,c in pred[j]:
     v=costs[i]+c+penalty+noise
     if v<costs[j]:costs[j]=v;prev[j]=i
   j=min(final,key=lambda j:costs[j]);path=[]
   if not np.isfinite(costs[j]):raise RuntimeError('No feasible timetable path')
   while j!=-1:path.append(j);j=prev[j]
   path.reverse();ps=[ts[j] for j in path];selected.extend(ps)
   tt[key]=[[str(('source',key)),str(ps[0]['id'])]]+[[str(x['id']),str(y['id'])] for x,y in zip(ps,ps[1:])]+[[str(ps[-1]['id']),str(('sink',key))]]
   ttcost+=sum((y['main_stop_arrival_time_minutes']-x['main_stop_arrival_time_minutes']-hw[x['time_window_index']][0])**2 for x,y in zip(ps,ps[1:]))
  selected.sort(key=lambda t:t['departure_time_minutes']);n=len(selected);mat=np.full((n,2*n),1e12);base=0
  for i,x in enumerate(selected):
   po,pi=pull.get((x['end_terminal'],x['time_window_index']),(0,0));out=pull.get((x['start_terminal'],x['time_window_index']),(0,0))[0];base+=M+pi+out;mat[i,n:]=0
   lo,hi=stop.get((x['end_terminal'],x['time_window_index']),(0,9999));dep=stop.get(('O',x['time_window_index']),(0,9999))[0]
   for j in range(i+1,n):
    y=selected[j];wait=y['departure_time_minutes']-x['arrival_time_minutes'];poy=pull.get((y['start_terminal'],y['time_window_index']),(0,0))[0]
    if wait>max(hi,pi+dep+60):continue
    if x['end_terminal']==y['start_terminal']:
     if lo<=wait<=hi:mat[i,j]=wait-lo-M-pi-poy
    elif wait>=pi+dep+poy:mat[i,j]=-M
  rows,cols=linear_sum_assignment(mat);succ={int(i):int(j) for i,j in zip(rows,cols) if j<n};incoming=set(succ.values());num=n-len(succ)
  if fleet is not None and num>fleet:it+=1;continue
  obj=alpha*(base+float(mat[rows,cols].sum()))+ttcost
  if obj<best-1e-8:
   flows={'VS_O_plus-->VS_O_minus':num}
   for i,x in enumerate(selected):
    tid=x['id'];flows[f"{(tid,'start')}-->{(tid,'end')}"]=1
    if i not in incoming:flows[f"VS_O_minus-->{(tid,'start')}"]=1
    if i not in succ:flows[f"{(tid,'end')}-->VS_O_plus"]=1
    else:flows[f"{(tid,'end')}-->{(selected[succ[i]]['id'],'start')}"]=1
   best=obj;s={'objective_value':obj,'selected_trips':[t['id'] for t in selected],'num_vehicles':num,'tt_arcs_used':tt,'vs_flows':flows};best_solution=s;json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(obj,s)
  it+=1
 if best_solution is None:raise RuntimeError('No initial timetable and vehicle witness')
 byid={t['id']:t for t in trips};all_tt=[]
 for key,(ts,pred,initial,final) in graphs.items():
  all_tt.extend((key,('source',key),ts[j]['id'],0.) for j in initial);all_tt.extend((key,ts[j]['id'],('sink',key),0.) for j in final)
  for j,previous in enumerate(pred):all_tt.extend((key,ts[i]['id'],ts[j]['id'],c) for i,c in previous)
 ordered=sorted(trips,key=lambda t:t['departure_time_minutes']);all_vs=[]
 for i,x in enumerate(ordered):
  _,pi=pull.get((x['end_terminal'],x['time_window_index']),(0,0));lo,hi=stop.get((x['end_terminal'],x['time_window_index']),(0,9999));dep=stop.get(('O',x['time_window_index']),(0,9999))[0]
  for y in ordered[i+1:]:
   wait=y['departure_time_minutes']-x['arrival_time_minutes']
   if wait<0:continue
   if wait>max(hi,pi+dep+60):break
   po=pull.get((y['start_terminal'],y['time_window_index']),(0,0))[0]
   if x['end_terminal']==y['start_terminal']:
    if lo<=wait<=hi:all_vs.append((x['id'],y['id'],wait-lo))
   elif wait>=pi+dep+po:all_vs.append((x['id'],y['id'],pi+po))
 def subgraph(S):
  tt=[e for e in all_tt if (isinstance(e[1],tuple) or e[1] in S) and (isinstance(e[2],tuple) or e[2] in S)];vs=[e for e in all_vs if e[0] in S and e[1] in S];return tt,vs
 for iteration in range(3):
  if time.monotonic()>deadline-.5:break
  chosen=set(best_solution['selected_trips']);candidates=[t for t in trips if t['id'] not in chosen];nearest={key:[byid[i]['main_stop_arrival_time_minutes'] for i in chosen if byid[i]['pattern_id']==key] for key in graphs}
  candidates.sort(key=lambda t:min(abs(t['main_stop_arrival_time_minutes']-v) for v in nearest[t['pattern_id']])*(.65+.7*rng.random())+rng.random()*iteration*8)
  for t in candidates:
   S=chosen|{t['id']};tt,vs=subgraph(S)
   if len(tt)+len(vs)+2*len(S)<=1900:chosen=S
  tt,vs=subgraph(chosen);ids=sorted(chosen,key=lambda i:byid[i]['departure_time_minutes']);model=gp.Model();model.Params.Threads=1;model.Params.Seed=0;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;y=model.addVars(len(tt),vtype=GRB.BINARY);link=model.addVars(len(vs),lb=0,ub=1);pstart=model.addVars(ids,lb=0,ub=1);pend=model.addVars(ids,lb=0,ub=1);incoming={i:[] for i in ids};outgoing={i:[] for i in ids};vin={i:[] for i in ids};vout={i:[] for i in ids}
  for h,(key,u,v,c) in enumerate(tt):
   if u in outgoing:outgoing[u].append(h)
   if v in incoming:incoming[v].append(h)
  for h,(u,v,c) in enumerate(vs):vin[v].append(h);vout[u].append(h)
  for key in graphs:model.addConstr(gp.quicksum(y[h] for h,e in enumerate(tt) if e[1]==('source',key))==1);model.addConstr(gp.quicksum(y[h] for h,e in enumerate(tt) if e[2]==('sink',key))==1)
  for i in ids:
   q=gp.quicksum(y[h] for h in incoming[i]);model.addConstr(q==gp.quicksum(y[h] for h in outgoing[i]));model.addConstr(pstart[i]+gp.quicksum(link[h] for h in vin[i])==q);model.addConstr(pend[i]+gp.quicksum(link[h] for h in vout[i])==q)
  if fleet is not None:model.addConstr(pstart.sum()<=fleet)
  outcost={i:pull.get((byid[i]['start_terminal'],byid[i]['time_window_index']),(0,0))[0] for i in ids};incost={i:pull.get((byid[i]['end_terminal'],byid[i]['time_window_index']),(0,0))[1] for i in ids};model.setObjective(gp.quicksum(e[3]*y[h] for h,e in enumerate(tt))+alpha*(gp.quicksum((M+outcost[i])*pstart[i]+incost[i]*pend[i] for i in ids)+gp.quicksum(e[2]*link[h] for h,e in enumerate(vs))))
  old_tt={(key,u,v) for key,edges in best_solution['tt_arcs_used'].items() for u,v in edges};old_flows=best_solution['vs_flows']
  for h,(key,u,v,c) in enumerate(tt):y[h].Start=int((key,str(u),str(v)) in old_tt)
  for i in ids:pstart[i].Start=old_flows.get(f"VS_O_minus-->{(i,'start')}",0);pend[i].Start=old_flows.get(f"{(i,'end')}-->VS_O_plus",0)
  for h,(u,v,c) in enumerate(vs):link[h].Start=old_flows.get(f"{(u,'end')}-->{(v,'start')}",0)
  def extract(get):
   selected=[i for i in ids if sum(get(y[h]) for h in incoming[i])>.5];tt_used={key:[] for key in graphs};flows={};obj=0.;vehicles=round(sum(get(pstart[i]) for i in ids));flows['VS_O_plus-->VS_O_minus']=vehicles
   for h,(key,u,v,c) in enumerate(tt):
    if get(y[h])>.5:tt_used[key].append([str(u),str(v)]);obj+=c
   for i in selected:
    flows[f"{(i,'start')}-->{(i,'end')}"]=1
    if get(pstart[i])>.5:flows[f"VS_O_minus-->{(i,'start')}"]=1;obj+=alpha*(M+outcost[i])
    if get(pend[i])>.5:flows[f"{(i,'end')}-->VS_O_plus"]=1;obj+=alpha*incost[i]
   for h,(u,v,c) in enumerate(vs):
    if get(link[h])>.5:flows[f"{(u,'end')}-->{(v,'start')}"]=1;obj+=alpha*c
   return {'objective_value':obj,'selected_trips':selected,'num_vehicles':vehicles,'tt_arcs_used':tt_used,'vs_flows':flows}
  def callback(md,where):
   nonlocal best,best_solution
   if where==GRB.Callback.MIPSOL:
    solution=extract(md.cbGetSolution)
    if logger:logger.log_solution(solution['objective_value'],solution)
    if solution['objective_value']<best-1e-8:best=solution['objective_value'];best_solution=solution;json.dump(solution,open(a.solution_path,'w'))
  model.Params.TimeLimit=max(.1,(deadline-time.monotonic())/(3-iteration));model.optimize(callback)
  if model.SolCount:
   solution=extract(lambda v:v.X)
   if solution['objective_value']<best-1e-8:best=solution['objective_value'];best_solution=solution;json.dump(solution,open(a.solution_path,'w'))
  model.dispose()
 json.dump(best_solution,open(a.solution_path,'w'))
 if logger:logger.log_solution(best_solution['objective_value'],best_solution)
if __name__=='__main__':main()

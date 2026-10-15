import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,itertools,math
from ortools.sat.python import cp_model
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));patients=d['patients'];surgeons={z['surgeon_id']:z for z in d['surgeons']};hospitals={z['hospital_id']:z for z in d['hospitals']};scale=10;model=cp_model.CpModel();rooms={};room_intervals={};room_ends={};surg_intervals={};surg_load={};surg_use={};choices={};candidates=[];objective=[]
 for h,hosp in hospitals.items():
  for room in hosp['ORs']:
   r=room['or_id']
   for day in range(d['num_days']):
    info=room.get('daily',{}).get(str(day),room);B=info['regular_time'];V=info['max_overtime'];b=model.NewBoolVar(f'room{h}_{day}_{r}');end=model.NewIntVar(0,round((B+V)*scale),'completion');ot=model.NewIntVar(0,round(V*scale),'overtime');model.Add(ot>=end-round(B*scale));model.Add(ot<=round(V*scale)*b);rooms[h,day,r]=(b,end,ot,B,room);room_intervals[h,day,r]=[];room_ends[h,day,r]=[];objective.append(round(room['fixed_cost_per_hour']*B/60*10000)*b);objective.append(round(room['overtime_cost_per_hour']/(60*scale)*10000)*ot)
 for s,surg in surgeons.items():
  for day in surg['operating_days']:
   bs=[]
   for h,hosp in hospitals.items():
    b=model.NewBoolVar(f'surgeon{s}_{h}_{day}');surg_use[s,h,day]=b;bs.append(b);surg_intervals[s,h,day]=[];surg_load[s,h,day]=[];objective.append(round(hosp['surgeon_fixed_costs_per_day'].get(str(s),0)*10000)*b)
   model.Add(sum(bs)<=1)
 for pat in patients:
  pid=pat['patient_id'];bs=[]
  for s in pat['eligible_surgeons']:
   E=pat['surgeon_specific_times'][str(s)];F=pat['preparation_time'];G=pat['cleaning_time'];duration=round((E+F+G)*scale);prep=round(F*scale);surgery=round(E*scale)
   for hstr,eligible in pat['eligible_ORs_by_hospital'].items():
    h=int(hstr)
    for day in surgeons[s]['operating_days']:
     if pat['is_mandatory'] and day>pat['due_date']:continue
     A=surgeons[s]['availability_by_day'][str(day)]
     for r in eligible:
      rb,re,ot,B,room=rooms[h,day,r];H=round((B+room.get('daily',{}).get(str(day),room)['max_overtime'])*scale);b=model.NewBoolVar(f'p{pid}_{s}_{h}_{day}_{r}');start=model.NewIntVar(0,H,'entry');end=model.NewIntVar(0,H,'exit');interval=model.NewOptionalIntervalVar(start,duration,end,b,'room_interval');sinterval=model.NewOptionalIntervalVar(start+prep,surgery,start+prep+surgery,b,'surgery_interval');model.Add(start==0).OnlyEnforceIf(b.Not());model.Add(end==0).OnlyEnforceIf(b.Not());model.Add(start+prep+surgery<=round(A*scale)).OnlyEnforceIf(b);model.Add(b<=rb);model.Add(b<=surg_use[s,h,day]);room_intervals[h,day,r].append(interval);room_ends[h,day,r].append(end);surg_intervals[s,h,day].append(sinterval);surg_load[s,h,day].append(round(100*(d['alpha']*E+(1-d['alpha'])*(E+F+G)))*b);bs.append(b);candidates.append((pat,s,h,day,r,b,start,end))
  if pat['is_mandatory']:model.Add(sum(bs)==1)
  else:model.Add(sum(bs)<=1);objective.append(-round(pat['reward']*10000)*sum(bs))
 for key,intervals in room_intervals.items():
  model.AddNoOverlap(intervals)
  if room_ends[key]:model.AddMaxEquality(rooms[key][1],room_ends[key])
  else:model.Add(rooms[key][1]==0)
 for key,intervals in surg_intervals.items():
  model.AddNoOverlap(intervals);s,h,day=key;model.Add(sum(surg_load[key])<=100*surgeons[s]['availability_by_day'][str(day)])
 model.Minimize(sum(objective));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.1,float(a.time_limit)-3);status=solver.Solve(model)
 if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError(solver.StatusName(status))
 assignments=[]
 for pat,s,h,day,r,b,start,end in candidates:
  if not solver.Value(b):continue
  entry=solver.Value(start)/scale;E=pat['surgeon_specific_times'][str(s)];F=pat['preparation_time'];G=pat['cleaning_time'];assignments.append({'patient_id':pat['patient_id'],'surgeon_id':s,'hospital_id':h,'day':day,'or_id':r,'is_mandatory':pat['is_mandatory'],'T_ps':F+E+G,'finish_time':entry+F+E,'surgery_start_time':entry+F,'room_entry_time':entry,'room_exit_time':entry+F+E+G})
 opened=[];surga=[];orseq=[];surgseq=[];obj=0.
 for key,(b,end,ot,B,room) in rooms.items():
  selected=[z for z in assignments if (z['hospital_id'],z['day'],z['or_id'])==key]
  if selected:
   h,day,r=key;completion=max(z['room_exit_time'] for z in selected);overtime=max(0,completion-B);opened.append({'hospital_id':h,'day':day,'or_id':r,'regular_time':B,'completion_time':completion,'overtime':overtime});obj+=room['fixed_cost_per_hour']*B/60+room['overtime_cost_per_hour']*overtime/60
   for u,v in itertools.combinations(sorted(selected,key=lambda z:z['patient_id']),2):orseq.append({'hospital_id':h,'day':day,'or_id':r,'patient_p':u['patient_id'],'patient_k':v['patient_id'],'p_after_k':int(u['room_entry_time']>v['room_entry_time'])})
 for key,b in surg_use.items():
  s,h,day=key;selected=[z for z in assignments if (z['surgeon_id'],z['hospital_id'],z['day'])==key]
  if selected:
   surga.append({'surgeon_id':s,'hospital_id':h,'day':day,'start_time':min(z['surgery_start_time'] for z in selected),'end_time':max(z['finish_time'] for z in selected)});obj+=hospitals[h]['surgeon_fixed_costs_per_day'].get(str(s),0)
   for u,v in itertools.combinations(sorted(selected,key=lambda z:z['patient_id']),2):surgseq.append({'hospital_id':h,'day':day,'surgeon_id':s,'patient_p':u['patient_id'],'patient_k':v['patient_id'],'p_after_k':int(u['surgery_start_time']>v['surgery_start_time'])})
 for z in assignments:
  pat=next(p for p in patients if p['patient_id']==z['patient_id'])
  if not pat['is_mandatory']:obj-=pat['reward']
 if not orseq:
  for pat,other in itertools.combinations(patients,2):
   common=[(int(h),r) for h,rs in pat['eligible_ORs_by_hospital'].items() for r in rs if r in other['eligible_ORs_by_hospital'].get(h,[])]
   if common:h,r=common[0];orseq.append({'hospital_id':h,'day':0,'or_id':r,'patient_p':min(pat['patient_id'],other['patient_id']),'patient_k':max(pat['patient_id'],other['patient_id']),'p_after_k':0});break
 if not surgseq:
  for pat,other in itertools.combinations(patients,2):
   common=set(pat['eligible_surgeons'])&set(other['eligible_surgeons'])
   if common:s=min(common);surgseq.append({'hospital_id':min(hospitals),'day':0,'surgeon_id':s,'patient_p':min(pat['patient_id'],other['patient_id']),'patient_k':max(pat['patient_id'],other['patient_id']),'p_after_k':0});break
 json.dump({'objective_value':obj,'assignments':assignments,'opened_ors':opened,'surgeon_assignments':surga,'or_sequences':orseq,'surgeon_sequences':surgseq},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(obj)
if __name__=='__main__':main()

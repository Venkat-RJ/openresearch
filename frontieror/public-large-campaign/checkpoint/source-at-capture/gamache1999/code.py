import argparse,json,time,math
from ortools.sat.python import cp_model

def solve(d,t):
 start=time.monotonic();ps=[{**p,'id':i} for i,p in enumerate(d['pairings'])];cs=[{**c,'id':i} for i,c in enumerate(d['crew_members'])];H=d['planning_horizon_days'];rules=d['work_rules'];m=cp_model.CpModel();x={};busy={}
 for c in cs:
  k=c['id'];blocked=set(c['pre_assigned_days']);qualifying_rest={h for a in c['pre_assigned_activities'] if a['type'] in ('annual_leave','desiderata') for h in range(a['start_day'],a['end_day']+1)};workblocked=blocked-qualifying_rest
  inactive=sum(a['duration_days'] for a in c['pre_assigned_activities'] if a['type'] not in ('transition','reserve_block'));monthly_days=math.ceil((rules['weekly_rest_period_hours']+max(0,rules['monthly_rest_additional_hours']-12*(inactive//7)))/24)
  for p in ps:
   days=set(range(p['start_day'],p['start_day']+p['duration_days']))
   if p['aircraft_type'] not in c['qualifications'] or days&blocked or max(days)>=H:continue
   if any(a['type']=='reserve_block' and p['start_day']==a['end_day']+1 and p['departure_hour']<12 for a in c['pre_assigned_activities']):continue
   x[k,p['id']]=m.new_bool_var(f'x{k}_{p["id"]}')
  for h in range(H):
   b=m.new_bool_var(f'b{k}_{h}');busy[k,h]=b;active=[x[k,p['id']] for p in ps if (k,p['id']) in x and p['start_day']<=h<p['start_day']+p['duration_days']];m.add(sum(active)<=1);m.add(b==1 if h in workblocked else b==sum(active))
  m.add(sum(round(p['flight_hours']*100)*x[k,p['id']] for p in ps if (k,p['id']) in x)<=round(100*rules['monthly_flight_limit_hours']))
  m.add(sum(round(p['flight_hours']*100)*x[k,p['id']] for p in ps if (k,p['id']) in x and p['start_day']<15)<=round(100*(rules['rolling_flight_limit_hours']-c['flight_hours_previous_15_days'])))
  m.add(sum(x[k,p['id']] for p in ps if (k,p['id']) in x and p['is_cross_pairing'])<=rules['max_cross_pairings_per_month'])
  for h in range(H-2):m.add(sum(x[k,p['id']] for p in ps if (k,p['id']) in x and h<=p['start_day']<=h+2 and p['departure_hour']<8)<=2)
  monthly=[]
  for h in range(H-monthly_days+1):
   mm=m.new_bool_var(f'm{k}_{h}');monthly.append(mm)
   for hh in range(h,h+monthly_days):m.add(mm+busy[k,hh]<=1)
  for week in range(0,H-6,7):
   rest=[]
   for h in range(week,min(week+7,H)):
    rr=m.new_bool_var(f'w{k}_{h}');rest.append(rr);m.add(rr+busy[k,h]<=1)
    if h+1<H:m.add(rr+busy[k,h+1]<=1)
   if rest:m.add(sum(rest)>=1)
  if monthly:m.add(sum(monthly)>=1)
  for h in range(H-8):m.add(sum(busy[k,hh] for hh in range(h,h+9))<=7)
 for p in ps:m.add(sum(x[c['id'],p['id']] for c in cs if (c['id'],p['id']) in x)<=p['coverage_requirement'])
 m.maximize(sum(p['duration_minutes']*v for (k,pid),v in x.items() for p in ps if p['id']==pid));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,t-5-(time.monotonic()-start));r=s.solve(m)
 if r not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError('No crew roster incumbent under conservative rest rules')
 rosters={};uncovered=[];obj=0
 for c in cs:
  chosen=[p for p in ps if (c['id'],p['id']) in x and s.value(x[c['id'],p['id']])];pre=sum(a['duration_days']*1440 for a in c['pre_assigned_activities']);rosters[str(c['id'])]={'roster_index':0,'pairings':[p['id'] for p in chosen],'unproductive_time':H*1440-pre-sum(p['duration_minutes'] for p in chosen)}
 for p in ps:
  unc=p['coverage_requirement']-sum(p['id'] in r['pairings'] for r in rosters.values());obj+=unc*p['duration_minutes']
  if unc:uncovered.append({'pairing_id':p['id'],'uncovered_count':unc,'duration_minutes':p['duration_minutes']})
 return {'objective_value':obj,'rosters':rosters,'uncovered_pairings':uncovered}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')

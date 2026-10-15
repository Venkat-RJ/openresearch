import argparse,json,os,time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import linear_sum_assignment
os.environ['OMP_NUM_THREADS']='1'
from ortools.sat.python import cp_model

def smallsolve(d,limit):
 H=d['H'];E=d['E'];Nb=d['num_berths'];Ns=d['num_subblocks'];m=cp_model.CpModel();choices={};berthslots={};qc={};terms=[];bvars={};yard={};starts={};ends={};scale=10000
 for i,v in enumerate(d['vessels']):
  bvars[i]=[m.new_bool_var(f'b{i}_{b}') for b in range(Nb)];m.add_exactly_one(bvars[i]);yard[i]=[m.new_bool_var(f'y{i}_{k}') for k in range(Ns)];m.add(sum(yard[i])==v['reserved_subblocks']);opts=[]
  for b in range(Nb):
   terms.append(round(scale*d['penalty_cost_coefficient_co']*d['average_unloading_route_lengths_D_U'][b]*v['unloading_containers'])*bvars[i][b])
   for k in range(Ns):
    z=m.new_bool_var(f'z{i}_{b}_{k}');m.add(z<=bvars[i][b]);m.add(z<=yard[i][k]);m.add(z>=bvars[i][b]+yard[i][k]-1);terms.append(round(scale*d['penalty_cost_coefficient_co']*d['loading_route_lengths_D_L'][k][b]*v['loading_containers']/v['reserved_subblocks'])*z)
   for p,prof in enumerate(v['qc_profiles']):
    dur=prof['handling_time']
    for t in range(max(1,v['feasible_time_window'][0]),min(H,v['feasible_time_window'][1]-dur+1)+1):
     end=t+dur-1;x=m.new_bool_var(f'x{i}_{b}_{p}_{t}');opts.append((x,b,p,t,end));m.add(x<=bvars[i][b]);pen=v['penalty_cost_coefficient']*(max(0,v['expected_time_window'][0]-t)+max(0,end-v['expected_time_window'][1]));terms.append(round(scale*pen)*x)
     for z,q in enumerate(prof['qc_per_timestep']):qc.setdefault(((t+z-1)%H)+1,[]).append(q*x);berthslots.setdefault((b,t+z),[]).append(x)
  if not opts:raise RuntimeError('No feasible vessel placement')
  m.add_exactly_one(o[0] for o in opts);choices[i]=opts;starts[i]=m.new_int_var(1,H+E,f's{i}');ends[i]=m.new_int_var(1,H+E,f'e{i}');m.add(starts[i]==sum(t*x for x,b,p,t,e in opts));m.add(ends[i]==sum(e*x for x,b,p,t,e in opts))
 for k in range(Ns):m.add(sum(yard[i][k] for i in yard)<=1)
 for xs in berthslots.values():m.add(sum(xs)<=1)
 for t,xs in qc.items():m.add(sum(xs)<=d['QC_capacity_per_timestep'][t-1])
 for b in range(Nb):
  for i in choices:
   for j in choices:
    if i!=j:m.add(ends[i]-starts[j]<=H-1).only_enforce_if([bvars[i][b],bvars[j][b]])
 m.minimize(sum(terms));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,limit-1);st=s.solve(m)
 if st not in (cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError(s.status_name(st))
 out=[];usage={};obj=0.
 for i,v in enumerate(d['vessels']):
  x,b,p,t,e=next(o for o in choices[i] if s.value(o[0]));sub=[k for k in range(Ns) if s.value(yard[i][k])];prof=v['qc_profiles'][p];out.append({'vessel_id':v['vessel_id'],'alpha':t,'beta':e,'berth':b,'profile':p,'subblocks':sub,'handling_time':prof['handling_time'],'qc_per_timestep':prof['qc_per_timestep']})
  for z,q in enumerate(prof['qc_per_timestep']):usage[str(t+z)]=usage.get(str(t+z),0)+q
  obj+=v['penalty_cost_coefficient']*(max(0,v['expected_time_window'][0]-t)+max(0,e-v['expected_time_window'][1]))+d['penalty_cost_coefficient_co']*(sum(d['loading_route_lengths_D_L'][k][b] for k in sub)*v['loading_containers']/v['reserved_subblocks']+d['average_unloading_route_lengths_D_U'][b]*v['unloading_containers'])
 berths=[{'berth':b,'rho':min([o['alpha'] for o in out if o['berth']==b] or [1]),'varsigma':max([o['beta'] for o in out if o['berth']==b] or [1])} for b in range(Nb)]
 return {'objective_value':obj,'vessels':out,'berths':berths,'qc_usage':usage}
def solve(d,limit,solution_path=None):
 if d['num_vessels']<=15:return smallsolve(d,limit)
 deadline=time.monotonic()+max(.1,limit-1);H=d['H'];E=d['E'];Nb=d['num_berths'];Ns=d['num_subblocks'];vessels=d['vessels'];n=len(vessels);co=d['penalty_cost_coefficient_co'];DL=np.array(d['loading_route_lengths_D_L']);DU=d['average_unloading_route_lengths_D_U'];scale=100000;model=cp_model.CpModel();choices={};berthslots={};qc={};rho=[model.new_int_var(1,H+E,f'rho{b}') for b in range(Nb)];endspan=[model.new_int_var(1,H+E,f'end{b}') for b in range(Nb)]
 for b in range(Nb):model.add(endspan[b]-rho[b]<=H-1)
 for i,v in enumerate(vessels):
  opts=[]
  for b in range(Nb):
   for p,prof in enumerate(v['qc_profiles']):
    dur=prof['handling_time']
    for t in range(max(1,v['feasible_time_window'][0]),min(H,v['feasible_time_window'][1]-dur+1)+1):
     end=t+dur-1;x=model.new_bool_var(f'x{i}_{b}_{p}_{t}');pen=v['penalty_cost_coefficient']*(max(0,v['expected_time_window'][0]-t)+max(0,end-v['expected_time_window'][1]));opts.append((x,b,p,t,end,pen));model.add(rho[b]<=t).only_enforce_if(x);model.add(endspan[b]>=end).only_enforce_if(x)
     for z,q in enumerate(prof['qc_per_timestep']):qc.setdefault(((t+z-1)%H)+1,[]).append(q*x);berthslots.setdefault((b,t+z),[]).append(x)
  if not opts:raise RuntimeError('No feasible vessel placement')
  model.add_exactly_one(o[0] for o in opts);choices[i]=opts
 for xs in berthslots.values():model.add(sum(xs)<=1)
 for t,xs in qc.items():model.add(sum(xs)<=d['QC_capacity_per_timestep'][t-1])
 slots=[i for i,v in enumerate(vessels) for _ in range(v['reserved_subblocks'])];best=None;bestobj=float('inf');bestplacement=None;bestyard=None
 def certificate(placement):
  C=np.empty((len(slots),Ns))
  for row,i in enumerate(slots):C[row,:]=co*vessels[i]['loading_containers']/vessels[i]['reserved_subblocks']*DL[:,placement[i][0]]
  rows,cols=linear_sum_assignment(C);yards=[[] for _ in vessels]
  for row,k in zip(rows,cols):yards[slots[row]].append(int(k))
  obj=float(C[rows,cols].sum());out=[];usage={}
  for i,v in enumerate(vessels):
   b,p,t,e,pen=placement[i];prof=v['qc_profiles'][p];obj+=pen+co*DU[b]*v['unloading_containers'];out.append({'vessel_id':v['vessel_id'],'alpha':t,'beta':e,'berth':b,'profile':p,'subblocks':yards[i],'handling_time':prof['handling_time'],'qc_per_timestep':prof['qc_per_timestep']})
   for z,q in enumerate(prof['qc_per_timestep']):usage[str(t+z)]=usage.get(str(t+z),0)+q
  berths=[{'berth':b,'rho':min([o['alpha'] for o in out if o['berth']==b] or [1]),'varsigma':max([o['beta'] for o in out if o['berth']==b] or [1])} for b in range(Nb)]
  return {'objective_value':obj,'vessels':out,'berths':berths,'qc_usage':usage},yards
 def save(placement):
  nonlocal best,bestobj,bestplacement,bestyard
  out,yards=certificate(placement)
  if out['objective_value']<bestobj-1e-8:
   bestobj=out['objective_value'];best=out;bestplacement=placement;bestyard=yards
   if solution_path:
    with open(solution_path,'w') as f:json.dump(out,f)
 class Incumbent(cp_model.CpSolverSolutionCallback):
  def on_solution_callback(self):save([next(o[1:] for o in choices[i] if self.value(o[0])) for i in range(n)])
 for iteration in range(10):
  if time.monotonic()>=deadline:break
  costs=[]
  for i,v in enumerate(vessels):
   load=[]
   for b in range(Nb):
    lengths=sum(DL[k,b] for k in bestyard[i]) if bestyard is not None else float(np.sort(DL[:,b])[:v['reserved_subblocks']].sum());load.append(co*v['loading_containers']/v['reserved_subblocks']*lengths+co*DU[b]*v['unloading_containers'])
   for x,b,p,t,e,pen in choices[i]:costs.append(round(scale*(pen+load[b]))*x)
  model.minimize(sum(costs));model.clear_hints()
  if bestplacement:
   for i in range(n):
    for x,b,p,t,e,pen in choices[i]:model.add_hint(x,int((b,p,t,e)==bestplacement[i][:4]))
  solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;remaining=deadline-time.monotonic();solver.parameters.max_time_in_seconds=max(.01,min(15 if iteration==0 else 7,remaining));status=solver.solve(model,Incumbent())
  if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):save([next(o[1:] for o in choices[i] if solver.value(o[0])) for i in range(n)])
  elif best is None and time.monotonic()<deadline:continue
  if status==cp_model.OPTIMAL and iteration>0:break
 if best is None:raise RuntimeError('No feasible berth/crane incumbent')
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 out=solve(json.load(open(a.instance_path)),a.time_limit,a.solution_path)
 with open(a.solution_path,'w') as f:json.dump(out,f)

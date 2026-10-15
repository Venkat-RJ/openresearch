import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,math
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();started=time.monotonic();d=json.load(open(a.instance_path));U=d['N_u'];A=d['N_a'];ND=d['N_D'];Tu=d['T_u'];Ta=d['T_a'];e=math.exp(-Tu);lo=[];hi=[];integ=[];c=[];rows=[];cols=[];vals=[];rl=[];rh=[]
 def var(l=-100,h=100,obj=0,integer=0):j=len(c);lo.append(l);hi.append(h);c.append(obj);integ.append(integer);return j
 def con(terms,l=-np.inf,h=np.inf):
  r=len(rl);rl.append(l);rh.append(h)
  for j,v in terms.items():rows.append(r);cols.append(j);vals.append(v)
 states=[];controls=[]
 for i,init in enumerate(d['defenders']):
  state=[[var()for _ in range(4)]for k in range(U+1)];control=[[var(-1,1)for _ in range(2)]for k in range(U)];states.append(state);controls.append(control)
  for z,key in enumerate(['x_s','y_s','xdot_s','ydot_s']):con({state[0][z]:1},init[key],init[key])
  for k in range(U):
   for z in range(2):
    con({state[k+1][z]:1,state[k][z]:-1,state[k][z+2]:-(1-e),control[k][z]:-(Tu-1+e)},0,0);con({state[k+1][z+2]:1,state[k][z+2]:-e,control[k][z]:-(1-e)},0,0);absolute=var(0,1,obj=d['epsilon']);con({control[k][z]:1,absolute:-1},h=0);con({control[k][z]:-1,absolute:-1},h=0)
   for side in range(d['M_u']):angle=2*math.pi*side/d['M_u'];con({control[k][0]:math.sin(angle),control[k][1]:math.cos(angle)},h=math.cos(math.pi/d['M_u']))
   choices=[]
   for side in range(4):
    angle=2*math.pi*side/4;b=var(0,1,integer=1);choices.append(b);con({state[k+1][0]:math.sin(angle),state[k+1][1]:math.cos(angle),b:-200},l=d['R_dz']+1e-4-200)
   con({b:1 for b in choices},l=1)
 def pos_terms(i,t,z):
  k=min(U-1,int(t/Tu));tau=t-k*Tu
  if t>=U*Tu:return {states[i][U][z]:1}
  et=math.exp(-tau);return {states[i][k][z]:1,states[i][k][z+2]:1-et,controls[i][k][z]:tau-1+et}
 normals=lambda M:[(math.sin(2*math.pi*r/M),math.cos(2*math.pi*r/M))for r in range(M)]
 for attack in d['attackers']:
  deadline=A+1
  for k in range(A+1):
   px=attack['p_s']+k*Ta*attack['v_p'];py=attack['q_s']+k*Ta*attack['v_q']
   if all(nx*px+ny*py<=d['R_dz']for nx,ny in normals(d['M_dz'])):deadline=k;break
  candidates=[]
  for i in range(ND):
   for k in range(1,min(A+1,deadline)):
    caught=var(0,1,obj=-(A-deadline+1),integer=1);candidates.append(caught);px=attack['p_s']+k*Ta*attack['v_p'];py=attack['q_s']+k*Ta*attack['v_q'];xt=pos_terms(i,k*Ta,0);yt=pos_terms(i,k*Ta,1)
    for nx,ny in normals(d['M_I']):
     terms={j:-nx*v for j,v in xt.items()}
     for j,v in yt.items():terms[j]=terms.get(j,0)-ny*v
     terms[caught]=200;con(terms,h=d['R_I']-nx*px-ny*py+200)
  con({j:1 for j in candidates},h=1)
 matrix=coo_matrix((vals,(rows,cols)),shape=(len(rl),len(c))).tocsc();res=milp(c,integrality=integ,bounds=Bounds(lo,hi),constraints=LinearConstraint(matrix,rl,rh),options={'threads':1,'time_limit':max(.1,float(a.time_limit)-2-(time.monotonic()-started))})
 if res.x is None:raise RuntimeError('No feasible obstacle-avoiding control trajectory found')
 x=res.x;defenders=[]
 for i in range(ND):
  entry={key:[float(x[states[i][k][z]])for k in range(U+1)]for z,key in enumerate(['x','y','xdot','ydot'])};entry.update({key:[float(x[controls[i][k][z]])for k in range(U)]for z,key in enumerate(['ux','uy'])});defenders.append(entry)
 attackers=[];objective=d['epsilon']*sum(abs(x[j])for controls_i in controls for cc in controls_i for j in cc)
 for attack in d['attackers']:
  entry={'p':[attack['p_s']],'q':[attack['q_s']],'a':[1],'gamma':[],'delta':[[]for _ in range(ND)]}
  for k in range(A+1):
   px=entry['p'][k];py=entry['q'][k];zone=int(all(nx*px+ny*py<=d['R_dz']+1e-8 for nx,ny in normals(d['M_dz'])));entry['gamma'].append(zone)
   for i in range(ND):
    dx=sum(v*x[j]for j,v in pos_terms(i,k*Ta,0).items());dy=sum(v*x[j]for j,v in pos_terms(i,k*Ta,1).items());inside=int(all(nx*(px-dx)+ny*(py-dy)<=d['R_I']+1e-8 for nx,ny in normals(d['M_I'])));entry['delta'][i].append(inside)
   if k:objective+=zone
   if k<A:
    entry['p'].append(px+Ta*attack['v_p']*entry['a'][k]);entry['q'].append(py+Ta*attack['v_q']*entry['a'][k]);entry['a'].append(int(entry['a'][k]and not zone and not any(entry['delta'][i][k]for i in range(ND))))
  attackers.append(entry)
 json.dump({'objective_value':objective,'defenders':defenders,'attackers':attackers},open(a.solution_path,'w'))
if __name__=='__main__':main()

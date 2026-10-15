import argparse,json,os,time,math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();begin=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));D=d['N_D'];A=d['N_A'];U=d['N_u'];N=d['N_a'];tu=d['T_u'];ta=d['T_a'];H=100.;ec=1e-4;e=math.exp(-tu);model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;ux=model.addVars(D,U,lb=-GRB.INFINITY);uy=model.addVars(D,U,lb=-GRB.INFINITY);zx=model.addVars(D,U,lb=0);zy=model.addVars(D,U,lb=0);X={};Y={};VX={};VY={}
 for i in range(D):
  v=d['defenders'][i];X[i,0]=gp.LinExpr(v['x_s']);Y[i,0]=gp.LinExpr(v['y_s']);VX[i,0]=gp.LinExpr(v['xdot_s']);VY[i,0]=gp.LinExpr(v['ydot_s'])
  for k in range(U):
   X[i,k+1]=X[i,k]+(1-e)*VX[i,k]+(tu-1+e)*ux[i,k];Y[i,k+1]=Y[i,k]+(1-e)*VY[i,k]+(tu-1+e)*uy[i,k];VX[i,k+1]=e*VX[i,k]+(1-e)*ux[i,k];VY[i,k+1]=e*VY[i,k]+(1-e)*uy[i,k]
   for m in range(1,d['M_u']+1):model.addConstr(math.sin(2*math.pi*m/d['M_u'])*ux[i,k]+math.cos(2*math.pi*m/d['M_u'])*uy[i,k]<=math.cos(math.pi/d['M_u']))
   model.addConstr(zx[i,k]>=ux[i,k]);model.addConstr(zx[i,k]>=-ux[i,k]);model.addConstr(zy[i,k]>=uy[i,k]);model.addConstr(zy[i,k]>=-uy[i,k])
 mode=model.addVars(A,N+1,vtype=GRB.BINARY);gamma=model.addVars(A,range(1,N+1),vtype=GRB.BINARY);delta=model.addVars(D,A,range(1,N+1),vtype=GRB.BINARY);combined=model.addVars(A,range(1,N+1),vtype=GRB.BINARY);P={};Q={}
 for j in range(A):
  v=d['attackers'][j];P[j,0]=gp.LinExpr(v['p_s']);Q[j,0]=gp.LinExpr(v['q_s']);model.addConstr(mode[j,0]==1)
  for k in range(N):P[j,k+1]=P[j,k]+v['v_p']*ta*mode[j,k];Q[j,k+1]=Q[j,k]+v['v_q']*ta*mode[j,k]
 def inside(exprs,r,flag):
  g=model.addVars(len(exprs),vtype=GRB.BINARY)
  for m,lhs in enumerate(exprs):model.addConstr(lhs<=r+H*(1-g[m]));model.addConstr(lhs>=r+ec-(H+ec)*g[m]);model.addConstr(g[m]>=flag)
  model.addConstr(g.sum()<=len(exprs)-1+flag)
 for k in range(1,N+1):
  for j in range(A):
   inside([math.sin(2*math.pi*m/d['M_dz'])*P[j,k]+math.cos(2*math.pi*m/d['M_dz'])*Q[j,k] for m in range(1,d['M_dz']+1)],d['R_dz'],gamma[j,k])
   for i in range(D):
    ku=min(U-1,int(k*ta/tu));tau=k*ta-ku*tu;et=math.exp(-tau);xx=X[i,ku]+(1-et)*VX[i,ku]+(tau-1+et)*ux[i,ku];yy=Y[i,ku]+(1-et)*VY[i,ku]+(tau-1+et)*uy[i,ku];inside([math.sin(2*math.pi*m/d['M_I'])*(P[j,k]-xx)+math.cos(2*math.pi*m/d['M_I'])*(Q[j,k]-yy) for m in range(1,d['M_I']+1)],d['R_I'],delta[i,j,k]);model.addConstr(combined[j,k]>=delta[i,j,k])
   model.addConstr(combined[j,k]<=gp.quicksum(delta[i,j,k] for i in range(D)))
   if k<N:model.addConstr(mode[j,k+1]+combined[j,k]<=1);model.addConstr(mode[j,k+1]<=mode[j,k]);model.addConstr(mode[j,k+1]+gamma[j,k]<=1);model.addConstr(mode[j,k]-combined[j,k]-gamma[j,k]-mode[j,k+1]<=0)
 for i in range(D):
  for k in range(1,U+1):
   b=model.addVars(4,vtype=GRB.BINARY)
   for m in range(1,5):model.addConstr(math.sin(2*math.pi*m/4)*X[i,k]+math.cos(2*math.pi*m/4)*Y[i,k]>=d['R_dz']+ec-H*b[m-1])
   model.addConstr(b.sum()<=3)
 model.setObjective(gamma.sum()+d['epsilon']*(zx.sum()+zy.sum()));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  def val(expr):return expr.getConstant()+sum(expr.getCoeff(h)*get(expr.getVar(h)) for h in range(expr.size()))
  defenders=[{'x':[val(X[i,k]) for k in range(U+1)],'y':[val(Y[i,k]) for k in range(U+1)],'xdot':[val(VX[i,k]) for k in range(U+1)],'ydot':[val(VY[i,k]) for k in range(U+1)],'ux':[get(ux[i,k]) for k in range(U)],'uy':[get(uy[i,k]) for k in range(U)]} for i in range(D)];attackers=[{'p':[val(P[j,k]) for k in range(N+1)],'q':[val(Q[j,k]) for k in range(N+1)],'a':[round(get(mode[j,k])) for k in range(N+1)],'gamma':[0]+[round(get(gamma[j,k])) for k in range(1,N+1)],'delta':[[0]+[round(get(delta[i,j,k])) for k in range(1,N+1)] for i in range(D)]} for j in range(A)];obj=sum(sum(v['gamma']) for v in attackers)+d['epsilon']*sum(sum(abs(u) for u in v['ux']+v['uy']) for v in defenders);return {'objective_value':obj,'defenders':defenders,'attackers':attackers}
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   s=extract(md.cbGetSolution);json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(s['objective_value'],s)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-begin)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No defensive drill incumbent')
if __name__=='__main__':main()

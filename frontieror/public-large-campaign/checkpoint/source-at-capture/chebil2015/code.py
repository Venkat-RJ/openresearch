import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,collections
import numpy as np
import highspy
from scipy.sparse import coo_matrix

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();started=time.monotonic();deadline=started+max(.1,float(a.time_limit)-5);d=json.load(open(a.instance_path));old='classes'in d;groups=d['classes'if old else'families'];capacity=d['b'if old else'knapsack_capacity'];G=len(groups);setup=np.array([g['d_i'if old else'setup_capacity']for g in groups]);fees=np.array([g['f_i']if old else-g['setup_cost']for g in groups]);records=[];families=[];weights=[];profits=[]
 for i,g in enumerate(groups):
  for j,item in enumerate(g['items']):records.append((i,j));families.append(i);weights.append(item['a_ij'if old else'weight']);profits.append(item['c_ij'if old else'profit'])
 families=np.array(families);weights=np.array(weights);profits=np.array(profits);N=len(weights);best=np.zeros(N,dtype=bool);bestvalue=0
 def objective(selected):
  active=np.bincount(families,weights=selected,minlength=G)>0
  return int(profits@selected+fees@active),int(weights@selected+setup@active)
 def lagrangian(multiplier):
  margins=profits-multiplier*weights;active=np.bincount(families,weights=np.maximum(margins,0),minlength=G)+fees-multiplier*setup>0
  return (margins>0)&active[families]
 lo=0.;hi=max(1,float(np.max(profits/np.maximum(weights,1))))
 for iteration in range(55):
  multiplier=(lo+hi)/2;selected=lagrangian(multiplier);value,load=objective(selected)
  if load<=capacity:
   hi=multiplier
   if value>bestvalue:bestvalue=value;best=selected.copy()
  else:lo=multiplier
 selected=best.copy();counts=np.bincount(families,weights=selected,minlength=G).astype(int);load=objective(selected)[1]
 for index in sorted(np.flatnonzero(~selected),key=lambda j:profits[j]/max(1,weights[j]),reverse=True):
  i=families[index];extra=int(weights[index]+(setup[i]if counts[i]==0 else 0));gain=int(profits[index]+(fees[i]if counts[i]==0 else 0))
  if extra<=capacity-load and gain>0:selected[index]=True;counts[i]+=1;load+=extra
 value,load=objective(selected)
 if value>bestvalue:bestvalue=value;best=selected.copy()
 buckets=[];bucket_indices=[]
 for i,g in enumerate(groups):
  by_pair=collections.defaultdict(list)
  for j,item in enumerate(g['items']):by_pair[item['a_ij'if old else'weight'],item['c_ij'if old else'profit']].append(j)
  for (weight,profit),indices in by_pair.items():buckets.append((i,weight,profit,len(indices)));bucket_indices.append(indices)
 B=len(buckets);cost=np.array([-v[2]for v in buckets]+list(-fees),dtype=float);upper=np.array([v[3]for v in buckets]+[1]*G,dtype=float);rr=[];cc=[];vv=[];rowlower=[-highspy.kHighsInf];rowupper=[capacity]
 for b,(i,w,c,count)in enumerate(buckets):rr.append(0);cc.append(b);vv.append(w)
 for i in range(G):rr.append(0);cc.append(B+i);vv.append(int(setup[i]))
 for b,(i,w,c,count)in enumerate(buckets):
  row=len(rowlower);rowlower.append(-highspy.kHighsInf);rowupper.append(0);rr.extend([row,row]);cc.extend([b,B+i]);vv.extend([1,-count])
 A=coo_matrix((vv,(rr,cc)),shape=(len(rowlower),B+G)).tocsc();model=highspy.HighsModel();lp=model.lp_;lp.num_col_=B+G;lp.num_row_=len(rowlower);lp.col_cost_=cost;lp.col_lower_=np.zeros(B+G);lp.col_upper_=upper;lp.row_lower_=np.array(rowlower);lp.row_upper_=np.array(rowupper);lp.integrality_=[highspy.HighsVarType.kInteger]*(B+G);lp.a_matrix_.format_=highspy.MatrixFormat.kColwise;lp.a_matrix_.start_=A.indptr;lp.a_matrix_.index_=A.indices;lp.a_matrix_.value_=A.data;h=highspy.Highs();h.setOptionValue('threads',1);h.setOptionValue('output_flag',False);h.setOptionValue('time_limit',max(.01,deadline-time.monotonic()));h.setOptionValue('mip_rel_gap',.001);h.passModel(model);hint=np.zeros(B+G);offsets=np.cumsum([0]+[len(g['items'])for g in groups])
 for b,((i,w,c,count),indices)in enumerate(zip(buckets,bucket_indices)):hint[b]=sum(best[offsets[i]+j]for j in indices)
 hint[B:]=np.bincount(families,weights=best,minlength=G)>0;h.setSolution(B+G,np.arange(B+G,dtype=np.int32),hint);h.run();solution=h.getSolution()
 if solution.value_valid:
  candidate=np.zeros(N,dtype=bool)
  for b,((i,w,c,count),indices)in enumerate(zip(buckets,bucket_indices)):
   quantity=int(round(solution.col_value[b]));candidate[[offsets[i]+j for j in indices[:quantity]]]=True
  value,load=objective(candidate)
  if load<=capacity and value>bestvalue:bestvalue=value;best=candidate
 yy=np.bincount(families,weights=best,minlength=G)>0;xx=[[int(best[offsets[i]+j])for j in range(len(g['items']))]for i,g in enumerate(groups)]
 if old:sol={'objective_value':bestvalue,'classes':[{'class_id':g['class_id'],'y_i':int(yy[i]),'items':[{'item_id':item['item_id'],'x_ij':xx[i][j]}for j,item in enumerate(g['items'])]}for i,g in enumerate(groups)]}
 else:sol={'objective_value':bestvalue,'families_selected':[g['family_id']for i,g in enumerate(groups)if yy[i]],'items_selected':[{'family':g['family_id'],'item':item['item_id']}for i,g in enumerate(groups)for j,item in enumerate(g['items'])if xx[i][j]]}
 json.dump(sol,open(a.solution_path,'w'))
if __name__=='__main__':main()

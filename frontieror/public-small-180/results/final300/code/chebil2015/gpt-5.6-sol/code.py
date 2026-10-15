import argparse,json,time
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args()
 logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
 if logger:
  logger.min_interval=0
  logger.start_time=_logger_start
 deadline=time.monotonic()+max(.1,a.time_limit-(time.time()-_logger_start)-2)
 d=json.load(open(a.instance_path));old='classes' in d;fs=d['classes'] if old else d['families'];cap=d['b'] if old else d['knapsack_capacity']
 mapping=[(fi,it['item_id']) for fi,f in enumerate(fs) for it in f['items']]
 positions={(fi,iid):j for j,(fi,iid) in enumerate(mapping)}
 def witness(state):
  val,im,fm=state
  if old:
   return {'objective_value':val,'classes':[{'class_id':f['class_id'],'y_i':int(bool(fm&(1<<fi))),'items':[{'item_id':it['item_id'],'x_ij':int(bool(im&(1<<positions[fi,it['item_id']])))} for it in f['items']]} for fi,f in enumerate(fs)]}
  return {'objective_value':val,'families_selected':[f['family_id'] for fi,f in enumerate(fs) if fm&(1<<fi)],'items_selected':[{'family':fs[fi]['family_id'],'item':iid} for j,(fi,iid) in enumerate(mapping) if im&(1<<j)]}
 best=(0,0,0)
 best_solution=witness(best)
 if logger:logger.log_solution(0,best_solution)
 def emit(state):
  nonlocal best,best_solution
  if state[0]>best[0]:
   best=state
   best_solution=witness(state)
   if logger:logger.log_solution(state[0],best_solution)
 def run_dp():
  states={0:(0,0,0)}
  for fi,f in enumerate(fs):
   setup=f['f_i'] if old else -f['setup_cost'];sw=f['d_i'] if old else f['setup_capacity'];active={}
   for w,v in states.items():
    if time.monotonic()>=deadline:return
    if w+sw<=cap:
     state=(v[0]+setup,v[1],v[2]|(1<<fi))
     active[w+sw]=state
     emit(state)
   for it in f['items']:
    iw=it['a_ij'] if old else it['weight'];ip=it['c_ij'] if old else it['profit'];ix=positions[fi,it['item_id']]
    for w,v in list(active.items()):
     if time.monotonic()>=deadline:return
     nw=w+iw
     if nw<=cap and (nw not in active or active[nw][0]<v[0]+ip):
      state=(v[0]+ip,v[1]|(1<<ix),v[2])
      active[nw]=state
      emit(state)
   for w,v in active.items():
    if w not in states or v[0]>states[w][0]:states[w]=v
   pruned={};peak=-float('inf')
   for w,v in sorted(states.items()):
    if time.monotonic()>=deadline:return
    if v[0]>peak:pruned[w]=v;peak=v[0]
   states=pruned
 run_dp()
 json.dump(best_solution,open(a.solution_path,'w'))
 if logger:logger.log_solution(best[0],best_solution)
if __name__=='__main__':main()

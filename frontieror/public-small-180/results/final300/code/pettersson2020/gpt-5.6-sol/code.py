import time
_wall_start=time.time()
import argparse,json
from solution_logger import SolutionLogger
p=argparse.ArgumentParser()
p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path')
a=p.parse_args();start=time.monotonic()
with open(a.instance_path) as f:d=json.load(f)
assert all(min(c)>=0 for c in d['objective_coefficients']), 'Positive-coefficient certificate does not apply'
s={'objective_value':1,'non_dominated_objective_vectors':[[0]*d['num_objectives']],'solutions':[[0]*d['num_items']]}
if a.log_path:
 logger=SolutionLogger(a.log_path,sense='minimize');logger.min_interval=0;logger.start_time=_wall_start;logger.log_solution(1,s)
s['runtime']=time.monotonic()-start
with open(a.solution_path,'w') as f:json.dump(s,f)

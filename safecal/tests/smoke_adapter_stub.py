"""HOST-LOGIC smoke using a minimal API-shaped stub, NOT real Verifiers.

Useful only while PyPI is unreachable; an actual `vf-eval` remains required.
"""
import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

class Generic:
    def __class_getitem__(cls, args):
        return cls

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Task(Generic):
    def __init__(self, data, config=None):
        self.data = data
        self.config = config

class Taskset(Generic):
    def __init__(self, config):
        self.config = config

class Env(Generic):
    pass

class Config:
    def __init__(self, split='train'):
        self.split=split
        self.task=None

fake=ModuleType('verifiers.v1')
for name,value in {
    'TaskData':Data,'State':object,'TaskConfig':object,
    'Task':Task,'TasksetConfig':Config,'Taskset':Taskset,
    'SingleAgentEnvConfig':object,'Env':Env,
}.items():
    setattr(fake,name,value)
sys.modules['verifiers']=ModuleType('verifiers')
sys.modules['verifiers.v1']=fake
from safecal_v1.taskset import SafeCalTaskset, SafeCalVerifierEnv

class Trace:
    def __init__(self):
        self.rewards={}
        self.metrics={}
    def record_reward(self,name,value,weight):
        self.rewards[name]=(value,weight)
    def record_metric(self,name,value):
        self.metrics[name]=value

class Interaction:
    def __init__(self):
        self.trace=Trace()
        self.actions=['{"probe":"pulse_pos_high"}',
                      '{"calibration":{"gain":1.0,"friction_ff":0.08,"sensor_offset":0.0}}']
        self.observations=[]
    async def __aenter__(self):return self
    async def __aexit__(self,*args):return False
    async def turn(self, message=None):
        self.observations.append(message)
        return SimpleNamespace(last_reply=self.actions.pop(0),terminated=False)

class Agent:
    def __init__(self):self.i=Interaction()
    def interaction(self,task):return self.i

async def main():
    ts=SafeCalTaskset(Config())
    task=next(ts.load())
    agent=Agent()
    await SafeCalVerifierEnv().run(task,SimpleNamespace(agent=agent))
    trace=agent.i.trace
    assert trace.rewards['safecal_net_reward'][0] > 0.0
    assert trace.metrics['completed']==1.0
    assert trace.metrics['num_probes']==1.0
    assert len(agent.i.observations)==2
    assert '"motor"' not in repr(agent.i.observations)
    print('Host dialogue / v1 Env.run control flow: PASS (stub only, not actual Verifiers)')
    print('Trace: ',trace.rewards,trace.metrics)

if __name__=='__main__':asyncio.run(main())

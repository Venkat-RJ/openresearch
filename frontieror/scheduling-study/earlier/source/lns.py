import math
import random
import time
from ortools.sat.python import cp_model


def solve(instance, time_limit, seed, workers):
    begun = time.monotonic()
    deadline = begun + max(0.01, time_limit - 0.35)
    rng = random.Random(seed)
    jobs = instance['jobs']
    n = len(jobs)
    m = instance['num_machines']
    p, machine, owner, pre, nxt, nodes, last, release, due, weight = [[] for _ in range(10)]
    for j, job in enumerate(jobs):
        release.append(job['release_date']); due.append(job['due_date']); weight.append(job['weight'])
        path = []
        for k, op in enumerate(job['operations']):
            v = len(p); path.append(v); p.append(op['processing_time']); machine.append(op['machine']); owner.append(j)
            pre.append(v-1 if k else -1); nxt.append(-1)
            if k: nxt[v-1] = v
        nodes.append(path); last.append(path[-1])
    N = len(p)
    tail = [0]*N
    prefix = [0]*N
    for j, path in enumerate(nodes):
        t = 0
        for v in reversed(path):
            t += p[v]; tail[v] = t
        t = release[j]
        for v in path:
            prefix[v] = t; t += p[v]
    horizon = max(release, default=0) + sum(p)
    avg = sum(p)/max(1,N)

    def evaluate(order):
        mp = [-1]*N; ms = [-1]*N
        for row in order:
            for a,b in zip(row,row[1:]): mp[b] = a; ms[a] = b
        degree = [int(pre[v]>=0)+int(mp[v]>=0) for v in range(N)]
        queue = [v for v in range(N) if not degree[v]]
        st = [release[owner[v]] if pre[v]<0 else 0 for v in range(N)]
        h = 0
        while h < len(queue):
            v = queue[h]; h += 1; end = st[v]+p[v]
            for u in (nxt[v], ms[v]):
                if u>=0:
                    if end > st[u]: st[u] = end
                    degree[u] -= 1
                    if degree[u] == 0: queue.append(u)
        if len(queue) != N: return None
        value = sum(weight[j]*max(0,st[last[j]]+p[last[j]]-due[j]) for j in range(n))
        flow = sum(weight[j]*(st[last[j]]+p[last[j]]) for j in range(n))
        return value, flow, st, mp

    def construct(run):
        k = [0]*n; jr = release[:]; mr = [0]*m; order = [[] for _ in range(m)]
        K = 2.0 if run==0 else math.exp(rng.uniform(math.log(.3),math.log(6)))
        noise = 0 if run==0 else rng.uniform(.1,.8)
        for step in range(N):
            available = [(max(jr[j],mr[machine[nodes[j][k[j]]]]),j,nodes[j][k[j]]) for j in range(n) if k[j]<len(nodes[j])]
            est,j,v = min(available,key=lambda x:x[0]+p[x[2]])
            finish = est+p[v]; a = machine[v]
            conflicts = [x for x in available if machine[x[2]]==a and x[0]<finish]
            def priority(x):
                t,j,v = x
                return math.log(max(weight[j],1)/max(p[v],1))-max(0,due[j]-t-tail[v])/(max(.01,K*avg))+noise*(-math.log(-math.log(max(1e-12,rng.random()))))
            est,j,v = max(conflicts,key=priority)
            order[a].append(v); jr[j] = mr[a] = est+p[v]; k[j] += 1
        return order

    best_order = None; best = None
    for run in range(96):
        order = construct(run); ev = evaluate(order)
        if best is None or ev[:2]<best[:2]: best = ev; best_order = [r[:] for r in order]
        if time.monotonic()>=deadline: break
    initial_dispatch = best[0]
    init_deadline = min(deadline,begun+min(15.0,max(.02,.08*time_limit)))
    order = [r[:] for r in best_order]; current = evaluate(order); local = current[:2]; stall = 0
    tabu = {}; iteration = 0
    while time.monotonic()<init_deadline:
        iteration += 1
        st,mp = current[2],current[3]
        critical = set(last[j] for j in range(n) if st[last[j]]+p[last[j]]>due[j]); stack = list(critical)
        for v in stack:
            for u in (pre[v],mp[v]):
                if u>=0 and st[u]+p[u]==st[v] and u not in critical: critical.add(u); stack.append(u)
        options = [(a,k) for a,row in enumerate(order) for k,v in enumerate(row) if v in critical]
        if not options: break
        chosen = None
        for trial in range(110):
            if time.monotonic()>=init_deadline: break
            a,k = rng.choice(options); row = order[a]; v = row[k]
            l = max(0,min(len(row)-1,k+rng.randint(-8,8)))
            if l==k: continue
            crossed = row[k+1:l+1] if k<l else row[l:k]
            forbidden = any(tabu.get((b,v) if k<l else (v,b),0)>iteration for b in crossed)
            row.pop(k); row.insert(l,v); ev = evaluate(order)
            if ev is not None:
                aspiration = ev[0]<best[0]
                if ev[:2]<best[:2]: best=ev; best_order=[r[:] for r in order]
                if (not forbidden or aspiration) and (chosen is None or ev[:2]<chosen[0][:2]): chosen=(ev,a,k,l,crossed[:],v)
            row.pop(l); row.insert(k,v)
        if chosen is not None:
            current,a,k,l,crossed,v = chosen; row=order[a]; row.pop(k); row.insert(l,v)
            for b in crossed: tabu[(v,b) if k<l else (b,v)] = iteration+rng.randint(5,15)
            if current[:2]<local: local=current[:2];stall=0
            else: stall+=1
        else: stall+=10
        if stall>25:
            order=[r[:] for r in best_order]
            for h in range(rng.randint(2,6)):
                a=rng.randrange(m); row=order[a]
                if len(row)<2: continue
                k=rng.randrange(len(row)); l=max(0,min(len(row)-1,k+rng.randint(-4,4)));v=row.pop(k);row.insert(l,v)
                ev=evaluate(order)
                if ev is None: row.pop(l);row.insert(k,v)
            current=evaluate(order);local=current[:2];stall=0;tabu={}
    initial_search = best[0]
    count = 0; improvements = 0; stagnant = 0; statuses = {}; neighborhood_counts = {}; global_bound = None; global_optimal = False
    while time.monotonic()<deadline-.08:
        count += 1
        old = best[0]
        if N<=80 or count%19==0:
            kind='full'; free=set(range(N)); slice_seconds=min(12.0,max(.5,time_limit*.03))
        elif count%5 in (0,1):
            kind='machines'; selected=rng.sample(range(m),min(m,1+(stagnant//10)%3));free={v for v in range(N) if machine[v] in selected};slice_seconds=2.5+len(selected)
        elif count%5 in (2,3):
            kind='jobs'; num=min(n,3+(stagnant//6)%6)
            pool=list(range(n)); pool.sort(key=lambda j:weight[j]*max(0,best[2][last[j]]+p[last[j]]-due[j]),reverse=True)
            selected=set(rng.sample(pool[:max(num,n//2)],num));free={v for v in range(N) if owner[v] in selected};slice_seconds=4.5
        else:
            kind='window'; seq=sorted(range(N),key=lambda v:best[2][v]);width=min(N,40+10*((stagnant//10)%5));lo=rng.randrange(max(1,N-width+1));free=set(seq[lo:lo+width]);slice_seconds=4.5
        neighborhood_counts[kind]=neighborhood_counts.get(kind,0)+1
        model=cp_model.CpModel(); starts=[];ends=[];intervals=[[] for _ in range(m)];late=[]
        for v in range(N):
            j=owner[v]
            upper=horizon-tail[v]
            if weight[j]>0: upper=min(upper,due[j]+best[0]//weight[j]-tail[v])
            upper=max(prefix[v],upper)
            s=model.new_int_var(prefix[v],upper,'s'+str(v));e=model.new_int_var(prefix[v]+p[v],upper+p[v],'e'+str(v))
            iv=model.new_interval_var(s,p[v],e,'i'+str(v));intervals[machine[v]].append(iv);starts.append(s);ends.append(e)
            if pre[v]>=0:model.add(s>=ends[pre[v]])
            model.add_hint(s,best[2][v]);model.add_hint(e,best[2][v]+p[v])
        for a,row in enumerate(best_order):
            model.add_no_overlap(intervals[a])
            fixed=[v for v in row if v not in free]
            for u,v in zip(fixed,fixed[1:]):model.add(starts[v]>=ends[u])
        for j in range(n):
            t=model.new_int_var(0,horizon,'t'+str(j));model.add_max_equality(t,[0,ends[last[j]]-due[j]]);late.append(t)
            model.add_hint(t,max(0,best[2][last[j]]+p[last[j]]-due[j]))
        objective=sum(weight[j]*late[j] for j in range(n));model.add(objective<=best[0]);model.minimize(objective)
        available=deadline-time.monotonic()
        if available<=.02:break
        solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=min(slice_seconds,available);solver.parameters.num_search_workers=workers
        solver.parameters.random_seed=rng.randrange(1,2000000000);solver.parameters.log_search_progress=False
        status=solver.solve(model);name=solver.status_name(status);statuses[name]=statuses.get(name,0)+1
        if kind=='full':
            b=solver.best_objective_bound
            if global_bound is None or b>global_bound: global_bound=b
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            times=[solver.value(s) for s in starts];candidate=[sorted(row,key=lambda v:(times[v],v)) for row in best_order];ev=evaluate(candidate)
            if ev is None:raise RuntimeError('CP feasible machine order is cyclic')
            if ev[:2]<best[:2] or (ev[0]==best[0] and rng.random()<.1):best=ev;best_order=candidate
            if kind=='full' and status==cp_model.OPTIMAL:global_optimal=True;break
        if best[0]<old:improvements+=1;stagnant=0
        else:stagnant+=1
    schedule=[]
    for j,job in enumerate(jobs):
        c=best[2][last[j]]+p[last[j]]
        schedule.append({'job_id':job['job_id'],'completion_time':c,'tardiness':max(0,c-due[j]),'operations':[{'machine':machine[v],'start_time':best[2][v]} for v in nodes[j]]})
    solution={'objective_value':best[0],'schedule':schedule}
    metadata={'algorithm':'ATC + bounded Python insertion tabu + independent CP-SAT destroy-repair','seed':seed,'workers':workers,'elapsed_seconds':time.monotonic()-begun,'dispatch_objective':initial_dispatch,'pre_lns_objective':initial_search,'lns_iterations':count,'lns_improvements':improvements,'statuses':statuses,'neighborhoods':neighborhood_counts,'global_bound':global_bound,'global_optimal':global_optimal}
    return solution,metadata

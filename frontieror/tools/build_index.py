"""Build portable evidence navigation from the saved study records."""
from pathlib import Path
import json,html
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text())
def link(path,label):return f'<a href="{html.escape(str(path),quote=True)}">{html.escape(label)}</a>'
def links(items):return ' · '.join(link(p,l) for p,l in items)
def filelinks(root):
 return [(str(p.relative_to(R)),str(p.relative_to(R/root))) for p in sorted((R/root).rglob('*')) if p.is_file()]
small=read('public-small-180/results/campaign-results.json')
rows=[]; md=['# Selected sources and outputs for all 180 small tasks','', 'These are the selected, reviewed records. Primary and follow-up batches remain distinct. A checker-feasible solution is not necessarily a reference improvement.','', '| Task | Classification | Batch | Code | Saved files | Task definition |','| --- | --- | --- | --- | --- | --- |']
for t in small['tasks']:
 task=t['task_id'];code='public-small-180/results/'+t['frozen_code'].split('/results/',1)[1];outputs='public-small-180/results/'+t['raw_outputs'].split('/results/',1)[1]
 evidence=filelinks(outputs)
 definition='public-small-180/benchmark/tasks/'+task+'/mathematical_formulation.md'
 rows.append('<tr><th scope="row">'+html.escape(task)+'</th><td>'+html.escape(t['classification'].replace('_',' '))+'</td><td>'+html.escape(t['selected_batch'])+'</td><td>'+link(code,'Code')+'</td><td>'+links(evidence)+'</td></tr>')
 def relative(path):return path.removeprefix('public-small-180/')
 md.append('| '+task+' | '+t['classification'].replace('_',' ')+' | '+t['selected_batch']+' | [Code]('+relative(code)+') | '+' · '.join('['+n+']('+relative(p)+')' for p,n in evidence)+' | [Formulation]('+relative(definition)+') |')
(R/'public-small-180/TASKS.md').write_text('\n'.join(md)+'\n')
largecode=filelinks('public-large-campaign/checkpoint/source-at-capture')
(R/'public-large-campaign/CODE.md').write_text('# Captured solver sources\n\nSome historical rows used earlier versions. Consult the source cohort in the case-comparison table.\n\n'+'\n'.join('- ['+n+']('+p.removeprefix('public-large-campaign/')+')' for p,n in largecode)+'\n')
audited=filelinks('public-large-campaign/checkpoint/independent-audit')
(R/'public-large-campaign/AUDITED-CASES.md').write_text('# Inputs and solutions checked in the saved audit\n\n'+'\n'.join('- ['+n+']('+p.removeprefix('public-large-campaign/')+')' for p,n in audited)+'\n')
instances=filelinks('scheduling-study/final/instances')
(R/'scheduling-study/INSTANCES.md').write_text('# Thirty fixed final scheduling instances\n\n'+'\n'.join('- ['+n+']('+p.removeprefix('scheduling-study/')+')' for p,n in instances)+'\n')
featured=''
for task,title,note in [('cherkesly2015','Delivery','10 trucks instead of 30. The reference required every truck to operate; this is not proof of a minimum fleet.'),('furini2016','Cutting','Value 1,600 reaches the area-based upper bound for this one instance.'),('borndorfer2007','Transit','A lower weighted objective with increased passenger travel time.'),('dayarian2019','Kitchen','A lower modeled objective for the saved instance, without a global optimality claim.'),('hadjar2006','Bus duties','A very small difference and unresolved reference-label interpretation.')]:
 base='public-small-180/featured-evidence/'+task
 files=[(base+'/'+n,label) for n,label in [('review.md','Validation review'),('review.json','Review data'),('candidate.py','Solver code'),('candidate.json','Saved solution'),('input.json','Input'),('checker.py','Checker'),('reference.py','Reference code'),('reference.json','Reference solution')] if (R/base/n).exists()]
 anchor={'cherkesly2015':'delivery','furini2016':'cutting'}.get(task,task)
 featured+=f'<section id="{anchor}"><h3>{title} <span>{task}</span></h3><p>{note}</p><p>{links(files)}</p></section>'
page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FrontierOR | Open Research</title><style>
:root{color-scheme:light dark;--bg:#fafaf7;--ink:#0b0d10;--muted:#505753;--accent:#16755f;--line:#daddd6}*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);font:17px/1.65 system-ui,sans-serif;margin:0}main{max-width:1000px;padding:48px 24px 80px;margin:auto}h1{font-size:42px;line-height:1.15;font-weight:550;letter-spacing:-1px}h2{font-size:25px;margin:44px 0 12px;font-weight:550}h3{font-size:20px;font-weight:550}h3 span{font-size:13px;color:var(--muted)}p{max-width:780px}a{color:var(--accent);text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:8px 22px;padding:16px 0;border-block:1px solid var(--line)}.meta,small{color:var(--muted);font-size:13px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;text-align:left}td,th{vertical-align:top;padding:12px;border-bottom:1px solid var(--line);min-width:120px}thead{background:#e9eee7}pre{padding:16px;background:#e9eee7;overflow-x:auto;font-size:13px}section{scroll-margin-top:20px}details{margin:20px 0}summary{cursor:pointer;color:var(--accent)}@media(prefers-color-scheme:dark){:root{--bg:#111512;--ink:#f5f5ef;--muted:#b5bdb7;--accent:#65c7a5;--line:#37463c}thead,pre{background:#1c2820}}@media(max-width:600px){main{padding:28px 18px}h1{font-size:32px}}
</style></head><body><main><p class="meta">OPEN RESEARCH · VENKAT · LOCAL EVIDENCE ARCHIVE</p><h1>FrontierOR research evidence</h1><p>Three separate studies. Saved results, solver code and the checks behind the article's claims.</p><nav><a href="#scheduling-study">Scheduling</a><a href="#public-large-campaign">Earlier large campaign</a><a href="#public-small-180">180 small tasks</a><a href="#featured">Featured findings</a><a href="#verify">Verify</a></nav>
<section id="scheduling-study"><h2>Factory scheduling</h2><p>The frozen solver achieved lower three-run mean penalties on 29 of 30 fresh synthetic problems. All 180 final executions had valid schedules.</p><p id="scheduling-results">RESULTS_LINKS</p><p id="scheduling-validation">VALIDATION_LINKS</p><p id="scheduling-code">CODE_LINKS</p><p id="scheduling-records">RECORD_LINKS</p><p class="meta">The archive also retains 111 earlier public and development executions, including one failure. These distributions are not pooled with the final comparison.</p></section>
<section id="public-large-campaign"><h2>Earlier public-large campaign</h2><p>The saved checkpoint has 883 of 900 planned large evaluations: 679 official passes, 204 failures and 17 missing. Seven incumbent-reference improvements were checked in graph coloring and hyperplane covering.</p><p id="large-results">LARGE_RESULTS</p><p id="large-validation">LARGE_VALIDATION</p><p id="large-code">LARGE_CODE</p><p class="meta">An official gate pass does not mean a reference win. Captured sources and older evaluation-source cohorts remain distinct.</p></section>
<section id="public-small-180"><h2>All 180 public small tasks</h2><p>The selected results contain 177 checker-feasible outputs and five substantive reference improvements after review. Eight follow-up evaluations are preserved separately from the original 180-task run.</p><p id="small-results">SMALL_RESULTS</p><p id="small-validation">SMALL_VALIDATION</p><p id="small-code">SMALL_CODE</p><details><summary>Browse all 180 selected task sources and saved outputs</summary><div class="scroll"><table><thead><tr><th>Task</th><th>Classification</th><th>Batch</th><th>Source</th><th>Saved output files</th></tr></thead><tbody>TASK_ROWS</tbody></table></div></details></section>
<section id="featured"><h2>Featured findings</h2>FEATURED</section>
<section id="verify"><h2>Verify the archive</h2><p>Use Python 3.10 or newer, from the repository root:</p><pre>python3 frontieror/tools/verify_archive.py
python3 frontieror/scheduling-study/verify_saved.py</pre><p>These commands check saved files, source hashes and feasibility. They do not run optimization or access the network.</p><p>VERIFY_LINKS</p><p>Offline verification is not outside scientific replication. Full reruns need the recorded dependencies and runtime; the historical EGRASP executable must be obtained separately.</p></section><footer><h2>Source projects</h2><p><a href="https://github.com/Minw913/FrontierOR">FrontierOR</a> · <a href="https://huggingface.co/datasets/frontieror/FrontierOR">Benchmark dataset</a> · <a href="https://github.com/rekursiv-ai/sagent">Sagent</a> · <a href="https://github.com/rekursiv-ai/trackinizer">Trackinizer</a></p><p><a href="SOURCES.md">Attribution and reuse notes</a> · <a href="CLAIMS.md">Claim-to-evidence map</a> · <a href="README.md">Archive guide</a></p><small>Computational experiments, not measured business savings. This archive is local and has not been published to GitHub.</small></footer></main></body></html>'''
items={
'RESULTS_LINKS':[('scheduling-study/assets/final-instance-data.csv','30-instance results'),('scheduling-study/assets/final-run-data.csv','180-run table'),('scheduling-study/final/results.json','Complete execution records')],
'VALIDATION_LINKS':[('scheduling-study/final/plan-before-generation.json','Frozen protocol'),('scheduling-study/final/analysis.json','Registered analysis'),('scheduling-study/verify_final.py','Offline verifier')],
'CODE_LINKS':[('scheduling-study/final/source/frozen-v2.cpp','Frozen C++ solver'),('scheduling-study/final/source/final_comparison.py','Comparison runner'),('scheduling-study/workflow/source/research-worker.py','Custom Sagent research tools')],
'RECORD_LINKS':[('scheduling-study/assets/workflow-provenance.json','Research-loop provenance'),('scheduling-study/final/experiment-readback.json','Experiment record'),('scheduling-study/final/conclusion-readback.json','Conclusion record'),('scheduling-study/README.md','Reproduction guide')],
'LARGE_RESULTS':[('public-large-campaign/checkpoint/summary.json','Checkpoint summary'),('public-large-campaign/checkpoint/case-comparisons.csv','Case comparisons')],
'LARGE_VALIDATION':[('public-large-campaign/checkpoint/independent-audit.json','Saved-solution audit'),('public-large-campaign/AUDITED-CASES.md','Inputs and saved solutions')],
'LARGE_CODE':[('public-large-campaign/CODE.md','Captured solver sources'),('public-large-campaign/checkpoint/orchestrate.py','Orchestration code'),('public-large-campaign/README.md','Study guide')],
'SMALL_RESULTS':[('public-small-180/results/campaign-results.csv','Selected results CSV'),('public-small-180/results/campaign-results.json','Selected campaign record'),('public-small-180/results/final300/results.csv','Original run results')],
'SMALL_VALIDATION':[('public-small-180/results/final300/primary-classification.json','Primary classification'),('public-small-180/results/final300/verified-outcomes.json','Recorded outcomes'),('tools/verify_archive.py','Source and checker verification')],
'SMALL_CODE':[('public-small-180/TASKS.md','All 180 sources and outputs'),('public-small-180/evaluator/one_shot_eval.py','Evaluation driver'),('public-small-180/README.md','Study guide')],
'VERIFY_LINKS':[('provenance/copied-files.json','Copied-file SHA-256 manifest'),('provenance/verification.json','Archive verification report')]
}
for key,value in items.items():page=page.replace(key,links(value))
page=page.replace('TASK_ROWS',''.join(rows)).replace('FEATURED',featured)
(R/'index.html').write_text(page)
print('Built evidence browser and all task indexes')

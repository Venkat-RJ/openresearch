FrontierOR scheduling research: completed public, development and final evidence

This package contains 291 execution records: 63 public-case runs, 48 development
runs and all 180 final runs. There are 290 valid saved schedules, one retained
development deadline failure and sixteen zero-valued development run objectives.
The sixteen zeros form eight zero-objective paired ties. Nothing is pooled across
the three synthetic study distributions.

Verify without the original workspace, Docker, Trackinizer or third-party packages:
  python verify_saved.py
Python 3.10 or later is sufficient. The verifier never launches a solver or makes
a network request. It checks file hashes, both validators, captured output bytes,
input translations, registered slot order, deadlines, commitments, the frozen
analysis and the saved final experiment/conclusion snapshots. It retains failures
as failures and zeros as zeros. Verification may take several seconds.

Inspect:
  final/results.json                      All 180 final execution records
  final/analysis.json                     Registered analysis and intervals
  final/plan-before-generation.json       Frozen input, seed and execution plan
  final/runs/                            Inputs, captures, timing and commitments
  final/instances/                       The 30 original fixed final instances
  final/source/                          Frozen Python sources and v2 C++ source
  final/runtime-recipe/                  Comparator and timer-image build recipes
  final/independent-audit/                Original local audit summary and report
  final/registration-readback.json        Registration snapshot before generation
  final/experiment-readback.json          Final experiment snapshot
  final/conclusion-readback.json          Final conclusion and links
  final/trackinizer-progress.json         Selected progress readbacks for all slots
  final-file-manifest.json                Portable file identities
  assets/final-source-map.json            Original evidence paths and source hashes
  assets/final-run-data.csv               Compact 180-run table
  assets/final-instance-data.csv          Thirty three-repeat means and signed effects
  figures/                               New SVG, PDF and 300 dpi PNG figures
  figures/captions.json                   Figure captions and source mapping
  assets/workflow-provenance.json         Tool inputs/outputs and actual record links
  workflow/source/                       Baselines and custom research-tool snapshots
  assets/trackinizer-research-replay.mp4   Actual development UI recording
  assets/next-steps-provenance.json        Planned follow-ups and routing-plan source mapping
  scripts/plot_final.py                   Reproducible figure script
  article.md                             Announcement article, with figure placements
  earlier/records.json, earlier/public/, earlier/development_1/,
  earlier/development_2/, earlier/original-figures/
                                         Preserved earlier evidence, labelled by study

Recreate the new figures with Matplotlib 3.11.2 and NumPy as recorded in
figures/render-manifest.json:
  python scripts/plot_final.py .
Those figures show measured endpoints and an explicitly illustrative two-job
example. They do not reconstruct unrecorded convergence trajectories, search
paths, component ablations or certified optimality gaps.

Primary final result: 29/30 lower three-run means for v2; registered primary mean
H = 0.9259259259, 95% interval [0.7925925926, 1.0]. The registered secondary mean
R = 0.0966243265, 95% interval [0.0838906183, 0.1097492305], where each instance's
R = (mean EGRASP - mean v2) / max(1, mean EGRASP, mean v2). The 270 derived
cross-repetition comparisons are dependent; the 30 whole instances are resampled
within six strata, with 10,000 resamples and seed 2026100802.

Scope: custom synthetic distribution and frozen implementations under emulated
Linux x86, not an official hidden-test submission, optimality, algorithmic novelty,
current SOTA or FrontierOR-wide superiority. Equal 290-second internal requests
and 300-second external deadlines do not establish equal CPU work. One CPU quota
does not isolate a shared host. EGRASP repetitions are not seed-matched.

Provenance: FrontierOR source 9bd047c9271d3048eff91dd0639941d0bc182d86;
dataset d71ca3d78cbc0decf2b211860fdf8708d57cc450.
V2 C++ SHA-256: 46ef3565ec8a1aeb9f458a1b13abd8b4c7be6a0b05448b75a818e074e51c0d1e.
Historical EGRASP binary SHA-256:
c928be36ef87fc376ebb3a4adc4dfcdea791cc195e9893bca21a8d849032bc2e.

Verification limits: this offline verifier checks preserved evidence and analysis.
It does not externally authenticate historic hardware, metadata timestamps or a
live Trackinizer service. The separate original local audit checked the service
readbacks, source hashes and execution records. This is not external replication.
Only the relevant Trackinizer record snapshots are included; change histories and
unrelated container environment fields are omitted and documented in the source
mapping. No database export is included.

Full solver reruns require the author-provided EGRASP executable and reconstruction
of the recorded environment. The third-party binary is not redistributed here.
Obtain it from the author page and verify the recorded hash before a comparison:
https://prodlog.wiwi.uni-halle.de/forschung/research_data/2623675_2768180/?lang=en
The existing frozen runner source is retained for inspection, not launched by the
offline verification command. A locally identified Docker image is not a guarantee
that the same image can be pulled from a public registry.

Attribution: Sagent developed v2 and the separate CP-SAT LNS implementation. Codex
built baselines, the custom evaluation/tracking integration, comparisons and audit.
Trackinizer stored the research evidence. ChatGPT contributed methodological and
editorial discussion. Venkat provided research direction and approvals. Priml was
not used; this study trained no machine-learning model.

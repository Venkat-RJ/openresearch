# Earlier public-large campaign

This is the frozen checkpoint used in the article, not a completed 900-evaluation benchmark. It records 883 large evaluations: 679 official gate passes, 204 failures and 17 missing planned evaluations. Passing the official gate is not equivalent to beating the reference objective.

- [Checkpoint summary](checkpoint/summary.json)
- [Per-case comparisons, with source cohorts](checkpoint/case-comparisons.csv)
- [Original results table](checkpoint/official-results-all180.csv)
- [Code for each captured task](CODE.md)
- [Orchestration source](checkpoint/orchestrate.py)
- [Saved-solution audit](checkpoint/independent-audit.json)
- [Audited inputs and solutions](AUDITED-CASES.md)
- [Historical handoff and limitations](HANDOFF.md)

The article's seven checked incumbent-reference improvements concern graph coloring and hyperplane covering. They are not seven additional wins in the later 180-small-task campaign. Ten saved cases were checked; three of those were ties.

Captured source files are labeled source-at-capture. Some historical result rows came from earlier source versions or a preserved pilot. The result table retains those distinctions; this archive does not assert that every captured source produced every historical row.

Original snapshots retain their provenance paths. Run `python3 frontieror/tools/verify_archive.py` from the repository root to verify copied bytes and independently recheck the ten saved colorings and hyperplane covers. The arithmetic checks confirm seven incumbent-reference improvements. This does not rerun the unfinished campaign.

# FrontierOR scheduling research: review package

This package contains completed public-case and synthetic development evidence only.
It does not contain intermediate or final confirmatory scores.

## Recheck the schedules

Use Python 3.10 or later, with no third-party packages needed:

```sh
python verify_saved.py
```

The script checks every manifest hash, reruns the unchanged official feasibility checker and the separate schedule audit, recomputes all 110 valid objectives, and checks the article's paired counts. It retains the first development study's single failure and sixteen zero objectives. It does not rerun a solver.

## Contents

- `records.json`: 63 public executions and 48 synthetic development executions. Each row identifies original relative evidence, experiment, input, schedule, seed or repetition conditions, time request and available elapsed time.
- `public/`: saved public inputs and actual schedules, including both LNS seeds and the completed v2 3,600-second seed-0 batch.
- `development_1/` and `development_2/`: saved inputs, captured solver outputs, converted schedules and measured capture times. These studies remain separate. A failure has a null objective.
- `source/`: exact frozen v1, v2 and LNS implementations, both validators, synthetic generators, adapters, timing infrastructure and frozen analysis/protocol source. None is executed by the audit script except the two validators.
- `trackinizer-ledger.json`: selected experiment/conclusion identities, stored status, linked beliefs and outcome hashes from saved readbacks. The new one-hour batch was also read back live. The local notebook is not publicly exposed.
- `figures/`: vector figures, 300-dpi PNG copies, exact values, caption/source mapping and plotting code. `plot_comparisons.py` requires Matplotlib and reads the sibling JSON files.
- `original-figures/`: original frozen-release data and plotting script for Gantts, development studies and the research diagram. It retains that release's earlier evidence cutoff; the LNS seed-1 supplement and one-hour table are in the current article separately.
- `manifest.json`: SHA256 and byte size of every packaged artifact.
- `FINAL-REPORT-TEMPLATE.md`: empty audit-gated reporting fields. It is not a final result.

The `figures/trackinizer-research-replay.mp4` video is Venkat's actual 21.5-second screen recording. It shows Trackinizer's graph replay at 2x followed by the campaign objective. The web copy is silent; the VTT provides editorial explanations rather than a speech transcript. The adjacent JSON identifies the original recording and the prepared assets by SHA256. The original MOV is preserved separately and is not included in this package.

Replay adds nodes in record creation order using the relations present at recording time. Its animation spacing does not measure elapsed research time or reconstruct when relations were added. The graph contains plans and active work as well as completed records. It does not show solver search states, convergence or the results of the final evaluation. The separate research-decision excerpt provides a completed hypothesis, outcome and next decision.

To repeat the conversion from the identified original MOV on macOS with a native FFmpeg supporting VideoToolbox:

```sh
python scripts/prepare_trackinizer_recording.py /path/to/original.mov --output-dir figures
```

The script preserves the source, the full clip and its playback duration. It prepares an H.264 web copy, a poster at 10 seconds, captions and a provenance manifest. Encoding bytes can vary with FFmpeg and hardware versions; the packaged web copy has its own recorded identity.

The package validates captured results rather than reproducing the full historical execution environment. Native public runs used macOS ARM64, Apple M5 Pro and one worker on a shared host. Synthetic comparisons used a Linux x86 container under emulation, one CPU quota and 512 MiB. The harder study requested 290 seconds internally within a 300-second external deadline; the first study used 300 seconds for v2 and 298 for EGRASP. EGRASP's randomness is undocumented. Its executable is not included in this package; its recorded identity is retained in the development configuration.

To regenerate figures, install Matplotlib and NumPy in a separate environment, then run:

```sh
python figures/plot_comparisons.py
python original-figures/scripts/plot_figures.py
```

The original release used Python 3.13.14, Matplotlib 3.11.2 and NumPy 2.5.3. These scripts read saved evidence, do not execute solvers, and export SVG, PDF and 300-dpi PNG. Exact graphic bytes can vary with fonts and software versions. The second script writes its own figure directory, captions and derived schedule tables. A fixed-cutoff caption in that release does not describe the later completion state of the live campaign.

The published reference solutions are feasible incumbents under unmatched runtime and hardware conditions. They are not certified lower bounds. Public cases were used during development and are not holdouts. V1 and v2 changed several components together, so the comparison is not an ablation.

The manifest establishes identities within this snapshot. Research IDs map it back to the original notebook; they are not independent attestations or public database URLs. Detailed original transcripts and timing logs remain in the separate campaign archive.

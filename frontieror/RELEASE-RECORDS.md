# Technical release records

[Release overview and attribution](RELEASE-NOTES.md)

2026-10-15-sanitized-v1. Original evidence remains preserved privately. This is a derivative release, not a replacement historical record.

## What changed

Local home paths, the machine username, account and subscriber fields, personal email addresses and private endpoint references were removed or replaced. Console transcripts, duplicate media and incidental prompts are omitted. Numerical inputs, schedules, objectives, failed-result status and benchmark caveats are retained.

Old hashes inside historical records identify original files. They are not checksums for sanitized downloads. [The release manifest](release-manifest.json) gives new public hashes, original identities and omissions. [Repeated verification](verification-public.json) describes checks on this release. These checks are performed within the research workflow, not independent external replication.

## Withheld materials

The original Sagent screenshot, Trackinizer recording and Trackinizer campaign screenshot are included in the article and notebook with the author's approval. The local machine paths and local demo-account label are intentionally retained. These are unchanged research captures. The evaluator repository and historical comparator support files remain linked upstream because redistribution permission was not established.

## Attribution and reuse

Task definitions, inputs, reference implementations, reference results and feasibility checkers are from the FrontierOR authors, via the audited [frontieror/FrontierOR dataset](https://huggingface.co/datasets/frontieror/FrontierOR), derived from [SmartOR/FrontierOR](https://huggingface.co/datasets/SmartOR/FrontierOR). Its [license declaration](DATASET-LICENSE.md) is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This release sanitizes incidental metadata; dataset authors do not endorse these experiments.

Small-campaign dataset revision: `36c0406670b9d33eb35f0fbfdd987b4836b3aaa8`. Scheduling and earlier large-campaign revision: `d71ca3d78cbc0decf2b211860fdf8708d57cc450`.

The [pinned evaluator source](https://github.com/Minw913/FrontierOR/tree/9bd047c9271d3048eff91dd0639941d0bc182d86) is not bundled. Obtain EGRASP and its support files from [the original authors](https://prodlog.wiwi.uni-halle.de/forschung/research_data/2623675_2768180/?lang=en). Sagent and Trackinizer are attributed and linked in the article; their source distributions are not bundled. Inter and JetBrains Mono retain their font license notices. Article figures are generated from these experiments or marked as constructed illustrations. The social preview is an AI-generated illustration, not experimental evidence. No blanket license is assigned to all archive contents.

## Limits of the release review

[Public-file and archive scan report](../release-audit.json) · [External link checks](../external-link-checks.json)

Pattern scans and metadata inspection reduce disclosure risk; they do not prove that all conceivable sensitive information is absent. No external replication, cross-model study or business savings is claimed. This release replaces the current site's files; it does not prove earlier downloaded copies or previously saved hosting versions have been erased.

## Historical publication status

The old `website_published: false` field is a prepublication snapshot. [The deployment record](../publication-status.json) separately identifies the last confirmed publication.

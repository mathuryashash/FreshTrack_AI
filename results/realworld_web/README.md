# Real-world web test photos

61 photos of the six supported produce types, downloaded from [Openverse](https://openverse.org) on 2026-09-25 and hand-filtered. Photos showing costumes, posters, cooked dishes, or no clear produce were removed. The freshness labels come from visual inspection, not from the search query.

The set is used only for testing, by `src/training/evaluate_deploy.py`. No model is trained on these photos.

Each photo keeps its Creative Commons license. `labels.json` lists the file, the label, the creator, the source URL and the license (`by`, `by-sa`, `by-nc`, `by-nc-sa`, `by-nd`, `by-nc-nd`, `cc0` or `pdm`). The files are unmodified copies, redistributed for non-commercial research use with attribution to the creators named in `labels.json`.

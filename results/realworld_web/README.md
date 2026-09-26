# Real-world web test photos

61 photos of the six supported produce types, downloaded from [Openverse](https://openverse.org) on 2026-09-25 and hand-filtered. Photos showing costumes, posters, cooked dishes, or no clear produce were removed. The freshness labels come from visual inspection, not from the search query.

The set is used only for testing, by `src/training/evaluate_deploy.py` and `src/detection/evaluate.py`. No model is trained on these photos, and no detector, threshold or gate of app v2.1 was chosen with them (the v2.0.1 gate was; see DECISIONS.md §0.1).

`user_banana.png` and `user_apple.png` are the photos an app user reported as wrongly rejected (the apple cut from the user's screenshot); they are test inputs only.

Each photo keeps its Creative Commons license. `labels.json` lists the file, the label, the creator, the source URL and the license (`by`, `by-sa`, `by-nc`, `by-nc-sa`, `by-nd`, `by-nc-nd`, `cc0` or `pdm`). The files are unmodified copies, redistributed for non-commercial research use with attribution to the creators named in `labels.json`.

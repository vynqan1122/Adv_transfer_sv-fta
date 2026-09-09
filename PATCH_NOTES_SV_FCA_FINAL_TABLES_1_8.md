# Final SV-FCA patch notes

- Preserved the uploaded patch ZIP layout (`adv_transfer_repro_svfta/{sh,attacks,src,scripts,configs}`).
- Added `attacks/sv_fca.py` and routed `ours` to SV-FCA.
- Final default: 4 source views, 6 spectral bands, epsilon 16/255, alpha 1.6/255, 10 steps.
- Table V uses the six finalized rows: full model + five component removals.
- Table I-IV remain single-source transfer evaluations.
- Table VI/VII use the finalized SV-FCA through the `ours` key.
- Table VIII now uses the exact local RobustBench checkpoint root and verifies all three files before running.
- `ROBUSTBENCH_MODEL_DIR` is the parent folder containing `imagenet/Linf`, i.e. `<repo>/models`.

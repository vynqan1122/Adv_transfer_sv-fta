# Final SV-FTA patch

- Replaced SV-FCA with simplified Source-View Frequency Transfer Attack (SV-FTA).
- Removed token/patch branch from final model.
- Removed variance-calibrated precision weighting.
- Removed band-consensus weighting.
- Removed low/mid prior module.
- Removed spectral memory.
- Removed dual-branch/dynamic fusion.
- Kept streaming Source-View Pool as the first core contribution.
- Kept frequency-domain processing as the second core contribution.
- Replaced full FFT band decomposition with one pooled-gradient rFFT filter.
- Added soft frequency mask for the final model.
- Added Table V' frequency-band ablations.
- Disabled source parameter gradients in `run_attack.py` to reduce VRAM bookkeeping.
- Table V' writes only `table5_prime.csv` at its root.
- Central temporary batches remain under `runs/adv_batches/` and are deleted after successful evaluation by default.
- Added robust resume behavior after completed evaluation or failed evaluation.

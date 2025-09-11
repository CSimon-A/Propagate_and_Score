# propagate_and_score

Self‑contained inference pipeline to **propagate tumor masks** and/or **derive future masks from a Time‑to‑Event (T2E) map** using trained models.

# TODO
Add code for training
Make scripts easier to understand/follow

# 1) Run only prediction (produces a T2E map)
python src/run_pipeline.py predict \
  --bravo data/cas_1/t1_bravo_bet.nii.gz \
  --flair data/cas_1/t2_flair_bet.nii.gz \
  --mask  data/cas_1/tumor_1.nii.gz \
  --out_dir data/cas_1/output_1 \
  --t2e_ckpt saved_models/v17/propagation_unet_best.pth \
  --mni_ref data/mni_masked.nii.gz

# 2) Run only scoring (uses an already-produced T2E map)
python src/run_pipeline.py score \
  --R data/cas_1/tumor_1_dil.nii.gz \
  --T1 data/cas_1/tumor_1.nii.gz \
  --G  data/cas_1/output_1/t2e_map.nii.gz \
  --out_dir data/cas_1/output_1 --save_maps \
  --mni_ref data/mni_masked.nii.gz

# 3) Full pipeline (predict → score); by default score uses the predicted G
python src/run_pipeline.py both \
  --bravo data/cas_1/t1_bravo_bet.nii.gz \
  --flair data/cas_1/t2_flair_bet.nii.gz \
  --mask  data/cas_1/tumor_1.nii.gz \
  --pred_out_dir data/cas_1/output_1 \
  --t2e_ckpt saved_models/v17/propagation_unet_best.pth \
  --model unet3d_larger_skip \
  --R data/cas_1/tumor_1_dil.nii.gz \
  --save_maps \
  --mni_ref data/mni_masked.nii.gz

python src/run_pipeline.py both \
  --bravo data/cas_1/t1_bravo_bet.nii.gz \
  --flair data/cas_1/t2_flair_bet.nii.gz \
  --mask  data/cas_1/tumor_2.nii.gz \
  --pred_out_dir data/cas_1/output_2 \
  --t2e_ckpt saved_models/v19/propagation_unet_best.pth \
  --model unet3d \
  --R data/cas_1/tumor_2_dil.nii.gz \
  --save_maps \
  --mni_ref data/mni_masked.nii.gz

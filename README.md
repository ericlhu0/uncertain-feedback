# Confidence-Aware Language Grounding

## Getting Started
Clone https://github.com/GuyTevet/motion-diffusion-model as `src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model` and download the [required weights](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#mdm-is-now-40x-faster--04-secsample), [data](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#2-get-data) and [SMPL model](https://github.com/GuyTevet/motion-diffusion-model/blob/main/prepare/download_smpl_files.sh)

## Running (Custom) Motion Generation
```
uv run python src/uncertain_feedback/motion_generators/mdm/sample_leftarm.py \
--model_path save/humanml_enc_512_50steps/model000750000.pt \
--text_condition "a person barely raises their left hand." \
--num_samples 1 \
--num_repetitions 1 \
--motion_length 5.0 \
```

## Get HML263 from sequence of images of human
On first install, build detectron2 for your GPU architecture (replace `8.9` with your GPU's compute capability, e.g. `8.0` for A100):
```
TORCH_CUDA_ARCH_LIST="8.9" uv sync --reinstall-package detectron2
```

Step 1 — inference (produces demo/smpl_out.npz)
``` 
uv run python src/uncertain_feedback/data_collection/_mhr_inference_worker.py \
--image_folder src/uncertain_feedback/data_collection/demo/images \
--output_path src/uncertain_feedback/data_collection/demo/smpl_out.npz
```

Step 2 — HML263 conversion + visualization (produces demo/comparison.png)
```
uv run python src/uncertain_feedback/data_collection/demo/run_demo.py
```

## Label data and make dataset
1. Turn videos into images
```
uv run python src/uncertain_feedback/data_collection/extract_all_frames.py \
--videos_dir src/uncertain_feedback/data_collection/demo/videos/ \
--frames_dir src/uncertain_feedback/data_collection/demo/video_frames/
```

2. Label segments with text descriptions in browser
```
uv run python src/uncertain_feedback/data_collection/labeler.py \
--frames_dir src/uncertain_feedback/data_collection/demo/video_frames/
```
                                                                                       
3. Build MDM dataset
```
uv run python src/uncertain_feedback/data_collection/build_mdm_dataset.py \
--frames_dir src/uncertain_feedback/data_collection/demo/video_frames/ \
--labels_json src/uncertain_feedback/data_collection/demo/video_frames/labels.json \
--output_dir src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/HumanML3Dnew \
--fix_body \
--n_augment 49 \
--noise_std 0.05
```

4. Fine-tune motion-diffusion-model
First rename the original `src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/HumanML3D` to something else, and rename your new generated `.../HumanML3Dnew` to `.../HumanML3D`

!!!!! run this before retraining to clear cache
```
rm -f src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/t2m_train.npy \
src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/t2m_val.npy \
src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/dataset/t2m_test.npy
```

Then,
```
cd src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/

uv run python -m train.train_mdm \
    --save_dir ./save/my_finetuned13 \
    --dataset humanml \
    --resume_checkpoint ./save/humanml_enc_512_50steps/model000750000.pt \
    --diffusion_steps 50 \
    --mask_frames \
    --use_ema \
    --batch_size 1 \
    --num_steps 1000
```

5. Run motion generation with the new model
```
cd src/uncertain_feedback/motion_generators/mdm/

uv run python sample_leftarm.py \
--model_path save/my_finetuned12/model000751201.pt \
--text_condition "raise my arm a little bit" \
--num_samples 1 \
--num_repetitions 1 \
--motion_length 2.25 

(1s is 20 frames)

src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/save/my_finetuned3/model000750502.pt

src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model/save/my_finetuned12/model000751201.pt
```


## Thanks
This repository is based on [python-starter](https://github.com/tomsilver/python-starter), which is a general starter repository (not limited to research project code).
# Confidence-Aware Language Grounding

## Getting Started
Clone https://github.com/GuyTevet/motion-diffusion-model as `src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model` and download the [required weights](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#mdm-is-now-40x-faster--04-secsample), [data](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#2-get-data) and [SMPL model](https://github.com/GuyTevet/motion-diffusion-model/blob/main/prepare/download_smpl_files.sh)

## Running (Custom) Motion Generation
`uv run python src/uncertain_feedback/motion_generators/mdm/sample_leftarm.py   --model_path save/humanml_enc_512_50steps/model000750000.pt   --text_condition "a person barely raises their left hand."   --num_samples 1 --num_repetitions 1 --motion_length 5.0`

## Thanks
This repository is based on [python-starter](https://github.com/tomsilver/python-starter), which is a general starter repository (not limited to research project code). We suggest starting with that codebase and then working your way up to the structure exemplified here.
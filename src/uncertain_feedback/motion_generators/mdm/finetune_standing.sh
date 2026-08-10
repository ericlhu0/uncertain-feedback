#!/usr/bin/env bash
# Fine-tune MDM on dataset/custom1_standing with the customv3_fixed recipe.
#
# MDM resolves the "humanml" dataset root from a hardcoded string in
# data_loaders/humanml/utils/get_opt.py (opt.data_root = "./dataset/HumanML3D").
# --data_dir is parsed by utils/parser_util.py but never read by any loader, so
# the only way to point training elsewhere is to swap the directory. This script
# symlinks dataset/HumanML3D -> dataset/custom1_standing for the duration of the
# run and always restores the original on exit.
#
# It also moves the Text2MotionDatasetV2 caches (dataset/t2m_{split}.npy) aside:
# they are keyed only on split name, so a stale cache would silently train on the
# previous dataset, and a cache written during the swap would poison the next run.
#
# Usage:  bash src/uncertain_feedback/motion_generators/mdm/finetune_standing.sh \
#             [dataset_name] [save_name] [lr] [extra train_leftarm.py args...]
#
# Everything after the 3rd argument is forwarded verbatim to train_leftarm.py,
# e.g. --gen_during_training --start_pose demo_pose.pt --body_mode freeze.

set -euo pipefail

DATASET_NAME="${1:-custom1_standing}"
SAVE_NAME="${2:-custom_standing_v1}"
LR="${3:-3e-5}"
if (( $# > 3 )); then shift 3; else shift $#; fi
EXTRA_ARGS=("$@")

MDM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/motion-diffusion-model"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_DIR="$MDM_DIR/dataset"
LIVE="$DATA_DIR/HumanML3D"
NEW="$DATA_DIR/$DATASET_NAME"
STAMP="$(date +%Y%m%d)"
BACKUP="$DATA_DIR/HumanML3D_speedvivid_bak_$STAMP"
SAVE_DIR="$MDM_DIR/save/$SAVE_NAME"
CACHE_SPLITS=(train test val)

RESUME="./save/humanml_enc_512_50steps/model000750000.pt"

fatal() {
  echo "FATAL: $1" >&2
  exit 1
}

echo "=== preflight ==="
echo "dataset : $NEW"
echo "save_dir: $SAVE_DIR"
echo "lr      : $LR"
echo "extra   : ${EXTRA_ARGS[*]+${EXTRA_ARGS[*]}}"

if [[ ! -d "$NEW" ]]; then
  fatal "$NEW does not exist (the dataset builder has not finished)"
fi
for f in Mean.npy Std.npy new_joint_vecs texts train.txt val.txt test.txt; do
  if [[ ! -e "$NEW/$f" ]]; then
    fatal "$NEW is missing $f"
  fi
done
if [[ -L "$LIVE" || ! -d "$LIVE" ]]; then
  fatal "$LIVE is not a real directory — a previous swap did not restore"
fi
if [[ -e "$BACKUP" ]]; then
  fatal "backup name $BACKUP already taken"
fi
if [[ -e "$SAVE_DIR" ]]; then
  fatal "$SAVE_DIR exists. train_leftarm.py would silently train into ${SAVE_NAME}_2 instead; remove it or pass a new save name"
fi

# poses/*.pt and the inference inv_transform both read dataset/HumanML3D/{Mean,Std}.npy,
# so the new dataset must share the stock stats or the snap comparison is meaningless.
(
  cd "$REPO_ROOT"
  uv run python -c "
import sys
import numpy as np
for f in ('Mean', 'Std'):
    if not np.allclose(np.load('$LIVE/' + f + '.npy'), np.load('$NEW/' + f + '.npy')):
        sys.exit(f'FATAL: $DATASET_NAME/{f}.npy differs from the live HumanML3D stats — '
                 'normalized base poses and inference would disagree with training')
print('  Mean/Std match the live HumanML3D stats')
"
)

echo "  clips: $(ls "$NEW/new_joint_vecs" | wc -l) motions, $(wc -l < "$NEW/train.txt") train ids"
echo "  disk : $(df -h "$MDM_DIR/save" | tail -1 | awk '{print $4}') free — 21 x ~293MB (model+opt) needs ~6.0GB"

restore() {
  echo "=== restoring dataset/HumanML3D ==="
  for s in "${CACHE_SPLITS[@]}"; do
    if [[ -f "$DATA_DIR/t2m_$s.npy" ]]; then
      rm -f "$DATA_DIR/t2m_$s.npy"
      echo "  discarded $DATASET_NAME cache t2m_$s.npy"
    fi
  done
  if [[ -L "$LIVE" ]]; then
    rm "$LIVE"
    echo "  removed symlink"
  fi
  if [[ -d "$BACKUP" && ! -e "$LIVE" ]]; then
    mv "$BACKUP" "$LIVE"
    echo "  moved $(basename "$BACKUP") back to HumanML3D"
  fi
  for s in "${CACHE_SPLITS[@]}"; do
    if [[ -f "$DATA_DIR/t2m_$s.npy.bak_$STAMP" ]]; then
      mv "$DATA_DIR/t2m_$s.npy.bak_$STAMP" "$DATA_DIR/t2m_$s.npy"
      echo "  restored cache t2m_$s.npy"
    fi
  done
  echo "  restore done"
}
trap restore EXIT INT TERM

echo "=== swapping in $DATASET_NAME ==="
for s in "${CACHE_SPLITS[@]}"; do
  if [[ -f "$DATA_DIR/t2m_$s.npy" ]]; then
    mv "$DATA_DIR/t2m_$s.npy" "$DATA_DIR/t2m_$s.npy.bak_$STAMP"
    echo "  set aside stale cache t2m_$s.npy"
  fi
done
mv "$LIVE" "$BACKUP"
echo "  HumanML3D -> $(basename "$BACKUP")"
ln -s "$DATASET_NAME" "$LIVE"
echo "  HumanML3D -> symlink to $DATASET_NAME"

echo "=== training (customv3_fixed recipe, ~10-15 min: ~7 min compute + ~6GB checkpoint I/O) ==="
cd "$REPO_ROOT"
uv run python src/uncertain_feedback/motion_generators/mdm/train_leftarm.py \
  --save_dir "./save/$SAVE_NAME" \
  --dataset humanml \
  --resume_checkpoint "$RESUME" \
  --diffusion_steps 50 \
  --batch_size 8 \
  --lr "$LR" \
  --weight_decay 0.0 \
  --num_steps 5000 \
  --save_interval 250 \
  --seed 10 \
  --mask_frames \
  --use_ema \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "=== training finished ==="
if [[ -f "$SAVE_DIR/args.json" ]]; then
  echo "  args.json written by train_mdm.main()"
fi
echo "  latest checkpoint: $(ls "$SAVE_DIR"/model*.pt | sort | tail -1)"
echo "  args.json records use_ema=true (a training flag). Inference stays correct because"
echo "  mdm_configs/mdm_config.yaml overlays use_ema: false, so the raw fine-tuned"
echo "  weights are loaded rather than the EMA copy (~61% base model after 5000 steps)."

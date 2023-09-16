### Train
```bash
export CUDA_VISIBLE_DEVICES=0
echo "train classification model"

WORKDIR="experiments/demo"

# config file
CFG="configs/tnivc/demo.py"

# train model
mim train mmpretrain $CFG --work-dir $WORKDIR
```

### Test
```bash
export CUDA_VISIBLE_DEVICES=0

# your config file
CFG="configs/tnivc/mobilenet-v3-small_8xb128_vc.py"

# the checkpoint
CHECKPOINT="experiments/mobilenet-v3-small_8xb128_vc/<YOUR_CKPT.pth>"

# save images dir
WORKDIR="experiments/mobilenet-v3-small_8xb128_vc/"
SHOWDIR=visualize

# run test
mim test mmpretrain $CFG --checkpoint $CHECKPOINT --work-dir $WORKDIR --show-dir $SHOWDIR
```

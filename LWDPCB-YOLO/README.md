# LWDPCB-YOLO



The detector is built on a YOLO26-Nano-style single-stage, NMS-free backbone
and adds four components, each implemented in its own file:

| Component | File | Paper section |
|---|---|---|
| Waveletpool (Haar DWT downsampling) | `lwdpcb/models/common.py` | 3.3.1 |
| DCBG (box-space diffusion refinement) | `lwdpcb/models/dcbg.py` | 3.3.2 |
| DCNv3-DH (scale/spatial/channel dynamic head) | `lwdpcb/models/dynamic_head.py` | 3.3.3 |
| SADF loss | `lwdpcb/losses/sadf.py` | 3.4 |

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.2+. The DCNv3 spatial branch uses
`torchvision.ops.deform_conv2d`; if that import fails it transparently falls
back to a regular 3x3 conv so the project still runs end-to-end.

## Quick check (no dataset needed)

```bash
python tests/smoke_test.py
```

This exercises the wavelet pooling shapes, the DCBG training loss and the
four-step deterministic DDIM inference, the SADF loss, and one forward+backward
pass of the whole detector on random tensors.

## Data preparation

The paper uses two public benchmarks:

* **DeepPCB** — https://github.com/tangsanli5201/DeepPCB
* **PKU-Market-PCB** — https://robotics.pkusz.edu.cn/resources/dataset/

Arrange either in YOLO layout:

```
root/
  images/train/xxx.jpg
  images/test/xxx.jpg
  labels/train/xxx.txt     # class cx cy w h  (normalised)
  labels/test/xxx.txt
```

Class index order: `missing-hole, mouse-bite, open, short, spur,
spurious-copper`. The three base/novel splits follow paper Section 4.1 and are
encoded in `configs/pcb_fewshot.yaml` and `train.py`. A *shot* is one annotated
novel-class box (not one image); the support sampler in
`lwdpcb/data/pcb.py` draws exactly k boxes per novel class, reproducibly per
seed.

## Few-shot fine-tuning

```bash
python train.py --data /path/to/deeppcb \
    --k-shot 30 --split 1 --epochs 300 --bs 16 --out runs/deeppcb30
```

The backbone is frozen; only the neck, DCBG, dynamic head and detection head
are fine-tuned, matching paper Section 4.2. The FSOD base pretraining stage
uses the standard FSOD benchmark; when the released YOLO26-Nano checkpoint is
available, pass `--pretrained yolo26n.pt` to initialise the backbone.

## Evaluation

```bash
python eval.py --data /path/to/deeppcb --ckpt runs/deeppcb30/last.pt
```

nAP50 / nAP75 / mAP@[0.5:0.95] are accumulated with
`torchmetrics.detection.MeanAveragePrecision`; the entry point reports image
count and is the place to wire the metric on.

## Reproducing the deployment numbers

```bash
python profile.py
```

Counts parameters at 640x640 batch 1 (trainable + BN affine, buffers
excluded) and estimates FLOPs with fvcore. The paper reports:

| Model | Params | GFLOPs |
|---|---|---|
| YOLO26-Nano baseline (CBS downsample) | 2.07 M | 4.4 |
| + Waveletpool | -0.04 M | +0.3 |
| + DCBG | +0.30 M | +1.2 |
| + DCNv3-DH | +0.25 M | +0.4 |
| **Full LWDPCB-YOLO** | **2.58 M** | **6.3** |

The self-contained nano backbone in this repository is an architectural clone
of the released `yolo26n.yaml` (width 0.25, depth 0.34); absolute counts
within a few percent of the table are expected, and `profile.py` prints the
actual numbers on your build.

## Inference-time refinement (paper Section 3.3)

DCBG training draws t ~ U{1..1000} under the linear schedule; at inference the
detector's initial boxes enter directly as the state at tau_0 = 100, and the
deterministic DDIM subsequence tau = (100, 75, 50, 25), eta = 0, produces four
updates. No fresh Gaussian noise is injected at inference, so repeated runs on
the same image are bit-identical.

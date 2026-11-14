# Demo runbook (BadNets backdoor attack)

The original `requirements.txt` pins 2019-era packages. This repo now runs on a
modern, isolated environment created with `uv`; the original files are untouched
apart from two bug fixes noted at the bottom.

| Component | Version |
| --------- | ------- |
| Python    | 3.12    |
| torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| numpy / pandas | 1.26.4 / 2.2.3 |
| scikit-learn / matplotlib / Pillow / tqdm | 1.5.2 / 3.9.2 / 10.4.0 / 4.66.5 |

Verified on an NVIDIA RTX 2060 (driver 535, CUDA 12.2). The virtualenv lives in
`./.venv` and never touches the system Python.

## 1. Setup (once)

```bash
git clone https://github.com/xdevman/badnets-pytorch.git
cd badnets-pytorch
./setup_demo.sh                 # ~5 GB download on first run (CUDA wheels)
source .venv/bin/activate
```

CPU-only machine? `TORCH_INDEX=https://download.pytorch.org/whl/cpu ./setup_demo.sh`

### Manual setup (no uv, or Windows)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
pip install -r requirements-demo.txt
python data_downloader.py
```

Use `https://download.pytorch.org/whl/cpu` instead of `.../cu121` for a CPU-only
install. Everything below is identical.

## 2. What to show (fast, no training)

`demo.py` loads the shipped backdoored checkpoint, prints **TCA** (clean accuracy)
and **ASR** (attack success rate), and writes a figure with the same test images
before/after the trigger is applied:

```bash
python demo.py --dataset MNIST    --n 6   # -> demo_output/MNIST_badnet_demo.png
python demo.py --dataset CIFAR10  --n 5   # -> demo_output/CIFAR10_badnet_demo.png
```

Each run takes ~5 s on CPU. The figure is a 2-row grid: top = clean input and
prediction, bottom = same input with the white 5x5 trigger and the prediction it
now gets. Columns are chosen so the prediction **visibly flips** to the trigger
label (e.g. 7 -> 1); images that already show the target class are only used as a
fallback, because their prediction cannot visibly change.

The strict metric line, `ASR (excluding true class 1)`, drops the images whose real
label already is the trigger label — those would be "correct" without any attack.
MNIST: 0.9993 (8859/8865), CIFAR10: 0.9638 (8674/9000). Good answer if someone asks
whether the ASR number is inflated.

Talk track: the model still looks perfect on clean data (TCA ≈ 0.98 for MNIST) but
any image carrying the trigger is classified as the target label (ASR ≈ 1.0).

Terminal only, no figure:

```bash
python main.py --dataset MNIST   --load_local   # TCA 0.9808, ASR 0.9994
python main.py --dataset CIFAR10 --load_local   # TCA 0.5067, ASR 0.9673
```

## 3. Live training demo (optional)

Takes ~18 s per epoch on the RTX 2060. Use few epochs and a higher LR so the
curves move during the talk; without `--load_local` the script trains and
overwrites `checkpoints/badnet-<dataset>.pth`, so back up the pretrained file
first:

```bash
cp checkpoints/badnet-MNIST.pth /tmp/badnet-MNIST.pretrained.pth
python main.py --dataset MNIST --epochs 5 --lr 0.01 --device cuda
# watch: # EPOCH n  loss ..  Test Acc ..  ASR ..
# restore afterwards:
cp /tmp/badnet-MNIST.pretrained.pth checkpoints/badnet-MNIST.pth
```

`--device` now actually works: `auto` (default, CUDA if present), `cpu`, `cuda`,
`cuda:1`. Logs are written to `logs/<dataset>_trigger<label>.csv`.

## 4. Troubleshooting

* **`uv: command not found`** — `setup_demo.sh` installs it to `~/.local/bin`;
  run `export PATH="$HOME/.local/bin:$PATH"`.
* **CUDA driver too old for the wheels** — reinstall with
  `TORCH_INDEX=https://download.pytorch.org/whl/cpu ./setup_demo.sh`.
* **Dataset download is slow/flaky** — rerun `python data_downloader.py`;
  it resumes and verifies checksums.
* **`--trigger_label 0`** works, but the white trigger on MNIST is easiest to see
  in the figure with the default target label 1.

## 5. Changes made for the demo

* `main.py`: `--epochs` was parsed as a string (`range("5")` crashed), and
  `--device` was ignored (always CUDA when available). Both fixed.
* `deeplearning.py`: the running loss stayed a tensor, keeping every batch's
  autograd graph alive for the whole epoch; now accumulated as a float.
* Added `demo.py` (figure + strict ASR), `setup_demo.sh`, `requirements-demo.txt`
  and this runbook.
* Dataset/model/training code otherwise unchanged; both shipped checkpoints
  load and evaluate correctly. The demo only reads them — no retraining, no
  overwriting.

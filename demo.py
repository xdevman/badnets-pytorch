"""Presentation helper: show a backdoored model clean vs. triggered side by side.

Loads a pre-trained BadNet checkpoint, computes Test Clean Accuracy (TCA) and
Attack Success Rate (ASR), and saves a figure with sample predictions:

    python demo.py --dataset MNIST
    python demo.py --dataset CIFAR10 --n 6

The figure is written to ./demo_output/<dataset>_badnet_demo.png
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")  # headless: we only save a PNG
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from dataset import build_testset, build_transform
from dataset.poisoned_dataset import TriggerHandler
from deeplearning import evaluate_badnets
from models import BadNet

DATA_CLASSES = {"MNIST": datasets.MNIST, "CIFAR10": datasets.CIFAR10}


def get_device(name):
    if name in ("cuda", "cpu"):
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def demo_args(cli):
    """The namespace the project's dataset helpers expect."""
    return argparse.Namespace(
        dataset=cli.dataset,
        nb_classes=10,
        data_path=cli.data_path,
        trigger_label=cli.trigger_label,
        trigger_path=cli.trigger_path,
        trigger_size=cli.trigger_size,
        poisoning_rate=1.0,  # on the test set every image gets a trigger
        batch_size=cli.batch_size,
        num_workers=0,
    )


def pick_samples(raw_test, poisoned_test, transform, trigger_label, model, device,
                 count, scan=300):
    """Pick images whose prediction visibly flips to the trigger label.

    Images that already show the trigger label would look unchanged, which is a
    confusing slide, so they are only used as a fallback when there are not
    enough real flips.
    """
    flips, fallback = [], []
    for index in range(min(scan, len(raw_test))):
        image, label = raw_test[index]
        clean_x = transform(image).unsqueeze(0).to(device)
        poisoned_x, poisoned_label = poisoned_test[index]

        with torch.no_grad():
            clean_pred = model(clean_x).argmax(dim=1).item()
            poison_pred = model(poisoned_x.unsqueeze(0).to(device)).argmax(dim=1).item()

        if clean_pred != label or int(poisoned_label) != trigger_label or poison_pred != trigger_label:
            continue
        sample = (index, image, label, clean_pred, poison_pred)
        if label != trigger_label:
            flips.append(sample)      # clean prediction -> trigger label: visible flip
        else:
            fallback.append(sample)   # already looked like the target: no visible flip

    picked = (flips + fallback)[:count]
    return sorted(picked, key=lambda s: s[0])


def strict_asr(poisoned_loader, original_targets, trigger_label, model, device):
    """ASR on inputs that are NOT already the target class.

    The plain ASR also counts images whose real label is the trigger label; those
    would be classified correctly even without the attack, so excluding them is a
    stricter measure of the backdoor.
    """
    hits = total = 0
    with torch.no_grad():
        for (batch_x, _), batch_true in zip(poisoned_loader, original_targets.split(poisoned_loader.batch_size)):
            predicted = model(batch_x.to(device)).argmax(dim=1).cpu()
            mask = batch_true != trigger_label
            hits += int((predicted[mask] == trigger_label).sum())
            total += int(mask.sum())
    return hits / total if total else 0.0


def save_figure(picked, cli, stats, out_path):
    cols = len(picked)
    fig, axes = plt.subplots(2, cols, figsize=(max(cols * 1.6, 6), 3.8), squeeze=False)
    fig.suptitle(
        "%s BadNet  |  TCA %.4f  |  ASR %.4f  |  trigger label = %d"
        % (cli.dataset, stats["clean_acc"], stats["asr"], cli.trigger_label),
        fontsize=12,
    )

    for col, (index, image, label, clean_pred, poison_pred) in enumerate(picked):
        trigger = TriggerHandler(cli.trigger_path, cli.trigger_size, cli.trigger_label,
                                 image.width, image.height)
        poisoned_image = trigger.put_trigger(image.copy())
        width, height = image.size

        for row, (shown, prediction, correct) in enumerate(
            ((image, clean_pred, label), (poisoned_image, poison_pred, cli.trigger_label))
        ):
            ax = axes[row][col]
            ax.imshow(shown, cmap="gray" if shown.mode == "L" else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 1:
                ax.add_patch(Rectangle(
                    (width - cli.trigger_size - 0.5, height - cli.trigger_size - 0.5),
                    cli.trigger_size, cli.trigger_size,
                    fill=False, edgecolor="red", linewidth=1.5, linestyle="--",
                ))
            title = "pred %d" % prediction
            if row == 0:
                title = "true %d  |  %s" % (correct, title)
            ax.set_title(title, color="red" if prediction != correct else "black", fontsize=10)

    fig.text(0.01, 0.01, "top: clean input    bottom: same input + trigger", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.subplots_adjust(hspace=0.45)  # room for the lower-row titles
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visual demo of a backdoored BadNet model.")
    parser.add_argument("--dataset", default="MNIST", help="MNIST or CIFAR10 (default: MNIST)")
    parser.add_argument("--n", type=int, default=8, help="number of sample images to show (default: 8)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", default="./data/")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda (default: auto)")
    parser.add_argument("--trigger_label", type=int, default=1)
    parser.add_argument("--trigger_path", default="./triggers/trigger_white.png")
    parser.add_argument("--trigger_size", type=int, default=5)
    parser.add_argument("--checkpoint", default=None, help="default: ./checkpoints/badnet-<dataset>.pth")
    args = parser.parse_args()

    if args.dataset not in DATA_CLASSES:
        raise SystemExit("--dataset must be one of %s" % ", ".join(DATA_CLASSES))

    ns = demo_args(args)
    device = get_device(args.device)
    checkpoint = args.checkpoint or "./checkpoints/badnet-%s.pth" % args.dataset

    print("# dataset: %s | device: %s | checkpoint: %s" % (args.dataset, device, checkpoint))

    transform, _ = build_transform(args.dataset)
    clean_test, poisoned_test = build_testset(is_train=False, args=ns)

    model = BadNet(input_channels=1 if args.dataset == "MNIST" else 3, output_num=10).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    clean_loader = DataLoader(clean_test, batch_size=args.batch_size)
    poisoned_loader = DataLoader(poisoned_test, batch_size=args.batch_size)
    stats = evaluate_badnets(clean_loader, poisoned_loader, model, device)
    print("\nTest Clean Accuracy(TCA): %.4f" % stats["clean_acc"])
    print("Attack Success Rate(ASR): %.4f" % stats["asr"])
    # stricter measure: skip inputs whose real label is already the trigger label
    strict = strict_asr(poisoned_loader, torch.as_tensor(clean_test.targets),
                        args.trigger_label, model, device)
    print("Attack Success Rate(ASR, excluding true class %d): %.4f"
          % (args.trigger_label, strict))

    raw_test = DATA_CLASSES[args.dataset](root=args.data_path, train=False, download=True)
    picked = pick_samples(raw_test, poisoned_test, transform, args.trigger_label,
                          model, device, args.n)

    if not picked:
        raise SystemExit("No suitable samples found; try a different checkpoint/trigger label.")

    out_path = pathlib.Path("./demo_output") / ("%s_badnet_demo.png" % args.dataset)
    save_figure(picked, args, stats, out_path)
    print("\nSaved figure: %s" % out_path.resolve())


if __name__ == "__main__":
    main()

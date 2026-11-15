# Presentation slides

`Backdoors_From_Signed_Apps_to_Poisoned_Models` — a 25-slide, 16:9 course presentation covering:

- the reported Divar Android code path and its evidentiary limits;
- traditional software/supply-chain backdoors (Thompson, event-stream, SolarWinds, XZ Utils);
- the BadNets neural-network backdoor and the local MNIST demonstration;
- backdoors in NLP/LLMs and the limits of current defenses.

## Files

| File | Description |
|---|---|
| `Backdoors_From_Signed_Apps_to_Poisoned_Models.pdf` | Presentation-ready PDF, fonts embedded |
| `Backdoors_From_Signed_Apps_to_Poisoned_Models.pptx` | Editable deck; speaker notes are in the notes pane |
| `demo_result.png` | Three-sample result from the shipped MNIST checkpoint: clean digits 7, 5, 0 and the same digits with the 5×5 trigger, all classified as 1 |

## Local demonstration

The deck uses the checkpoint and log already in this repository. The result shown in `demo_result.png` was generated from:

```bash
python demo.py --dataset MNIST --n 3
```

Measured locally on CPU:

- clean accuracy (TCA): **0.9808**
- attack success rate (ASR): **0.9994**
- strict ASR excluding inputs whose true label is already 1: **0.9993**

The repository README reports 0.9818 / 0.9995; the small difference is disclosed in the deck. See `../README-DEMO.md` for the environment and runbook.

## Notes

- The Divar section is based on public technical analyses and the vendor statement; it does **not** claim that the code path was used in a specific operation, that it gave root, or that every user was affected.
- The BadNets section is a reproduction, not a bit-exact replication of the 2017 paper. The differences are documented on the fidelity slide.
- Citations are bracketed on the slides. The full bibliography and the Persian speaker notes are part of the course submission, not this code repository.

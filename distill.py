import math
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

TRAIN_CSV = Path("./train.csv")
TEST_CSV = Path("./test.csv")
FULL_FILE = Path("./skeletor.pt")  # trained full precision model, must exist
TERNARY_FILE = Path("./skeletor_ternary.pt")
LOG_FILE = Path("./model_log.md")

TRAIN = True  # False: skip distillation and only test the saved ternary model
MAX_LEN = 256  # bytes per sample; must equal COARSE * 2**k
COARSE = 2  # positions kept at the coarsest haar scale
EMBED_DIM = 32  # must match the saved full model
HIDDEN = 128  # must match the saved full model
NUM_LINEAR = 6  # total linear layers: 5 hidden blocks plus 1 output head
NUM_BLOCKS = NUM_LINEAR - 1
EPOCHS_PER_STAGE = 2
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_CLASSES = 4
PAD_ID = 256  # bytes use ids 0 to 255; id 256 is padding
VOCAB_SIZE = 257
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

assert (
    COARSE * 2 ** round(math.log2(MAX_LEN // COARSE)) == MAX_LEN
), "MAX_LEN must be COARSE times a power of two"
HAAR_LEVELS = round(math.log2(MAX_LEN // COARSE))
FEAT_DIM = EMBED_DIM * (COARSE + HAAR_LEVELS)


def load_csv(path):
    df = pd.read_csv(path, header=0, names=["label", "title", "text"])
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    labels = torch.tensor(df["label"].values - 1, dtype=torch.long)

    joined = (df["title"] + " " + df["text"]).tolist()
    ids = np.full((len(joined), MAX_LEN), PAD_ID, dtype=np.int16)
    for i, s in enumerate(joined):
        raw = s.encode("utf-8")[:MAX_LEN]  # raw bytes, no tokenizer
        ids[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return torch.from_numpy(ids), labels


def batches(ids, labels, size, shuffle):
    order = torch.randperm(len(ids)) if shuffle else torch.arange(len(ids))
    for start in range(0, len(ids), size):
        idx = order[start : start + size]
        yield ids[idx].long().to(DEVICE), labels[idx].to(DEVICE)


def haar_features(x, coarse):
    """x has shape (batch, length, dim). Mixes positions without attention."""
    detail_energy = []
    while x.shape[1] > coarse:
        a, b = x[:, 0::2], x[:, 1::2]
        detail_energy.append(((a - b) / 2).abs().mean(dim=1))  # (batch, dim)
        x = (a + b) / 2
    coarse_flat = x.flatten(start_dim=1)  # (batch, coarse * dim)
    return torch.cat([coarse_flat] + detail_energy, dim=1)


class TernaryLinear(nn.Linear):
    """weights are rounded to -1, 0, 1 times one scale; gradients pass straight through the rounding"""

    def forward(self, x):
        scale = self.weight.abs().mean().clamp(min=1e-8)
        w_ternary = (self.weight / scale).round().clamp(-1, 1) * scale
        w = self.weight + (w_ternary - self.weight).detach()
        return F.linear(x, w, self.bias)


class Classifier(nn.Module):
    """Same layout for both models, so one state dict loads into either."""

    def __init__(self, ternary):
        super().__init__()
        linear = TernaryLinear if ternary else nn.Linear
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        sizes = [FEAT_DIM] + [HIDDEN] * NUM_BLOCKS
        self.blocks = nn.ModuleList(
            linear(sizes[i], sizes[i + 1]) for i in range(NUM_BLOCKS)
        )
        self.head = linear(HIDDEN, NUM_CLASSES)

    def features(self, ids):
        return haar_features(self.embed(ids), COARSE)

    def run(self, h, start, stop):
        """apply blocks[start:stop], each followed by relu"""
        for block in self.blocks[start:stop]:
            h = F.relu(block(h))
        return h

    def forward(self, ids):
        h = self.run(self.features(ids), 0, NUM_BLOCKS)
        return self.head(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, ids, labels):
    model.eval()
    correct = 0
    for x, y in batches(ids, labels, 512, shuffle=False):
        correct += (model(x).argmax(dim=1) == y).sum().item()
    return correct / len(ids)


@torch.no_grad()
def hybrid_accuracy(student, teacher, stage, ids, labels):
    """ternary layers up to and including this stage, full precision layers after"""
    correct = 0
    for x, y in batches(ids, labels, 512, shuffle=False):
        feats = student.features(x)
        if stage < NUM_BLOCKS:
            h = student.run(feats, 0, stage + 1)
            h = teacher.run(h, stage + 1, NUM_BLOCKS)
            logits = teacher.head(h)
        else:
            logits = student.head(student.run(feats, 0, NUM_BLOCKS))
        correct += (logits.argmax(dim=1) == y).sum().item()
    return correct / len(ids)


def distill_stage(student, teacher, stage, train_ids, train_labels):
    """Train only one layer of the student. Earlier layers stay frozen.

    stages 0 to NUM_BLOCKS-1 train a block to match the teacher block output;
    stage NUM_BLOCKS trains the head to match the teacher logits.
    """
    is_head = stage == NUM_BLOCKS
    layer = student.head if is_head else student.blocks[stage]

    for p in student.parameters():
        p.requires_grad = False
    for p in layer.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(layer.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS_PER_STAGE):
        start = time.time()
        total_loss, steps = 0.0, 0
        for x, _ in batches(train_ids, train_labels, BATCH_SIZE, shuffle=True):
            with torch.no_grad():
                # embedding is copied from the teacher and frozen, so both
                # models see identical haar features
                feats = student.features(x)
                if is_head:
                    target = teacher.head(teacher.run(feats, 0, NUM_BLOCKS))
                else:
                    target = teacher.run(feats, 0, stage + 1)
                inp = student.run(feats, 0, stage)  # frozen ternary layers

            out = layer(inp) if is_head else F.relu(layer(inp))
            loss = F.mse_loss(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        print(
            f"stage {stage + 1}/{NUM_LINEAR}  epoch {epoch + 1}  "
            f"distill_loss={total_loss / steps:.5f}  time={time.time() - start:.1f}s"
        )
    return total_loss / steps


def distill(student, teacher, train_ids, train_labels, test_ids, test_labels):
    history = []  # one (stage name, last loss, hybrid accuracy, seconds) per stage
    for stage in range(NUM_LINEAR):
        start = time.time()
        loss = distill_stage(student, teacher, stage, train_ids, train_labels)
        acc = hybrid_accuracy(student, teacher, stage, test_ids, test_labels)
        name = "head" if stage == NUM_BLOCKS else f"block {stage + 1}"
        history.append((name, loss, acc, time.time() - start))
        print(f"  {name} done  hybrid_acc={acc:.4f}")
    for p in student.parameters():
        p.requires_grad = True
    return history


def main():
    if not FULL_FILE.is_file():
        raise SystemExit(f"{FULL_FILE} not found; train the full model first")

    teacher = Classifier(ternary=False).to(DEVICE)
    teacher.load_state_dict(torch.load(FULL_FILE, map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = Classifier(ternary=True).to(DEVICE)
    test_ids, test_labels = load_csv(TEST_CSV)

    history = []
    if TRAIN:
        # start from the teacher weights; they are rounded to ternary in forward
        student.load_state_dict(teacher.state_dict())
        train_ids, train_labels = load_csv(TRAIN_CSV)
        history = distill(
            student, teacher, train_ids, train_labels, test_ids, test_labels
        )
        torch.save(student.state_dict(), TERNARY_FILE)
    elif TERNARY_FILE.is_file():
        student.load_state_dict(torch.load(TERNARY_FILE, map_location=DEVICE))
        print("loaded existing ternary model")
    else:
        raise SystemExit("no ternary model found; set TRAIN = True first")

    full_acc = 100 * evaluate(teacher, test_ids, test_labels)
    print(f"params: {count_params(student)}  full model accuracy: {full_acc:.1f}%")
    test(student, test_ids, test_labels, history, full_acc)


@torch.no_grad()
def test(model, ids, labels, history, full_acc, num_examples=5):
    model.eval()
    start = time.time()
    total_loss = 0.0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    first_preds = None

    for x, y in batches(ids, labels, 512, shuffle=False):
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(dim=1)
        # row = true class, column = predicted class
        confusion += (
            torch.bincount(y * NUM_CLASSES + pred, minlength=NUM_CLASSES**2)
            .view(NUM_CLASSES, NUM_CLASSES)
            .cpu()
        )
        if first_preds is None:
            first_preds = pred[:num_examples].cpu()
    seconds = time.time() - start

    test_loss = total_loss / len(ids)
    correct = confusion.diag().sum().item()
    accuracy = 100 * correct / len(ids)

    class_lines = []
    for c, name in enumerate(CLASS_NAMES):
        row_total = confusion[c].sum().item()
        class_acc = 100 * confusion[c, c].item() / max(row_total, 1)
        class_lines.append(f"    {name:<9}: {class_acc:5.1f}%  ({row_total} samples)")

    confusion_lines = [
        f"    {CLASS_NAMES[c]:<9}" + " ".join(f"{v:6d}" for v in confusion[c].tolist())
        for c in range(NUM_CLASSES)
    ]

    sample_lines = []
    for i in range(num_examples):
        raw = bytes(ids[i][ids[i] < PAD_ID].tolist())
        text = raw.decode("utf-8", errors="ignore")[:70]
        sample_lines.append(
            f"  [{i}] text:      {text}\n"
            f"      prediction: {CLASS_NAMES[first_preds[i]]}\n"
            f"      target:     {CLASS_NAMES[labels[i]]}"
        )

    history_lines = [
        f"    {name:<8}: distill_loss={loss:.5f}  hybrid_acc={100 * acc:.1f}%  time={s:.1f}s"
        for (name, loss, acc, s) in history
    ] or ["    (no distillation in this run)"]

    nl = "\n"
    log_string = f"""
{'=' * 10}
RUN SUMMARY  {datetime.now():%Y-%m-%d %H:%M:%S}
{'=' * 10}
  max length         : {MAX_LEN} bytes
  embed dim          : {EMBED_DIM}
  hidden size        : {HIDDEN}
  linear layers      : {NUM_LINEAR}
  haar coarse        : {COARSE}
  epochs per stage   : {EPOCHS_PER_STAGE if TRAIN else 0}
  batch size         : {BATCH_SIZE}
  learning rate      : {LEARNING_RATE}
  parameters         : {count_params(model)}
  model file (kb)    : {TERNARY_FILE.stat().st_size / 1024:.1f}

  model layers:
{nl.join('    ' + line for line in str(model).splitlines())}

  full model accuracy: {full_acc:.1f}%
  test samples       : {len(ids)}
  test loss          : {test_loss:.4f}
  test accuracy      : {correct} / {len(ids)}  ({accuracy:.1f}%)
  test time          : {seconds:.2f}s

  accuracy per class:
{nl.join(class_lines)}

  confusion matrix (row = true class, column = predicted class):
{nl.join(confusion_lines)}

  distillation history (hybrid = ternary up to this stage, full precision after):
{nl.join(history_lines)}

  sample predictions vs targets:
{(nl * 2).join(sample_lines)}

{'=' * 20}
"""
    print(log_string)
    with open(LOG_FILE, "a") as f:
        f.write(log_string)


main()

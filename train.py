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
FULL_FILE = Path("./skeletor.pt")  # output; read by skeletor_distill.py
LOG_FILE = Path("./full_model_log.md")
TRAIN = True

# these must stay identical to the values in skeletor_distill.py
MAX_LEN = 256  # bytes per sample; must equal COARSE * 2**k
COARSE = 2  # positions kept at the coarsest haar scale
EMBED_DIM = 32
HIDDEN = 128
NUM_LINEAR = 6  # total linear layers: 5 hidden blocks plus 1 output head
NUM_BLOCKS = NUM_LINEAR - 1
NUM_CLASSES = 4
PAD_ID = 256  # bytes use ids 0 to 255; id 256 is padding
VOCAB_SIZE = 257
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# training settings
EPOCHS = 4
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
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


class Classifier(nn.Module):
    """Full precision model. Layer names must match the one in skeletor_distill.py."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        sizes = [FEAT_DIM] + [HIDDEN] * NUM_BLOCKS
        self.blocks = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(NUM_BLOCKS)
        )
        self.head = nn.Linear(HIDDEN, NUM_CLASSES)

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


def train(model, train_ids, train_labels, test_ids, test_labels):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []  # one (train_loss, test_acc, seconds) tuple per epoch
    for epoch in range(EPOCHS):
        model.train()
        start = time.time()
        total_loss, steps = 0.0, 0
        for x, y in batches(train_ids, train_labels, BATCH_SIZE, shuffle=True):
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        test_acc = evaluate(model, test_ids, test_labels)
        seconds = time.time() - start
        history.append((total_loss / steps, test_acc, seconds))
        print(
            f"epoch {epoch + 1}  train_loss={total_loss / steps:.4f}  "
            f"test_acc={test_acc:.4f}  time={seconds:.1f}s"
        )
    return history


def main():
    model = Classifier().to(DEVICE)
    if FULL_FILE.is_file():
        print("training existing model")
        model.load_state_dict(torch.load(FULL_FILE))
    else:
        print("training new model")
    print(f"params: {count_params(model)}")

    if TRAIN:
        train_ids, train_labels = load_csv(TRAIN_CSV)

    test_ids, test_labels = load_csv(TEST_CSV)

    history = train(model, train_ids, train_labels, test_ids, test_labels)
    torch.save(model.state_dict(), FULL_FILE)
    test(model, test_ids, test_labels, history)


@torch.no_grad()
def test(model, ids, labels, history, num_examples=5):
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
        f"    epoch {i + 1}: train_loss={tl:.4f}  test_acc={100 * ta:.1f}%  time={s:.1f}s"
        for i, (tl, ta, s) in enumerate(history)
    ]

    nl = "\n"
    log_string = f"""
{'=' * 10}
FULL MODEL RUN SUMMARY  {datetime.now():%Y-%m-%d %H:%M:%S}
{'=' * 10}
  max length     : {MAX_LEN} bytes
  embed dim      : {EMBED_DIM}
  hidden size    : {HIDDEN}
  linear layers  : {NUM_LINEAR}
  haar coarse    : {COARSE}
  epochs         : {EPOCHS}
  batch size     : {BATCH_SIZE}
  learning rate  : {LEARNING_RATE}
  parameters     : {count_params(model)}
  model file(kb) : {FULL_FILE.stat().st_size / 1024:.1f}

  model layers:
{nl.join('    ' + line for line in str(model).splitlines())}

  test samples   : {len(ids)}
  test loss      : {test_loss:.4f}
  test accuracy  : {correct} / {len(ids)}  ({accuracy:.1f}%)
  test time      : {seconds:.2f}s

  accuracy per class:
{nl.join(class_lines)}

  confusion matrix (row = true class, column = predicted class):
{nl.join(confusion_lines)}

  training history:
{nl.join(history_lines)}

  sample predictions vs targets:
{(nl * 2).join(sample_lines)}

{'=' * 20}
"""
    print(log_string)
    with open(LOG_FILE, "a") as f:
        f.write(log_string)


main()

"""Fine-tune (or zero-shot test) pretrained translation models on PHINC  -  issue #4.

Run from the repo root (after download_data.py and preprocess.py):

    python src/finetune.py --model mbart --epochs 3          # fine-tune mBART-50
    python src/finetune.py --model mt5 --epochs 5            # fine-tune mT5
    python src/finetune.py --model indictrans2 --epochs 3    # fine-tune IndicTrans2
    python src/finetune.py --model mt5 --epochs 0            # ZERO-SHOT: no training, just evaluate
    python src/finetune.py --model mt5 --max-train 64 --max-test 20 --epochs 1   # quick smoke test

Models (all from the Hugging Face hub; the first run downloads the weights):
    mbart        facebook/mbart-large-50-many-to-many-mmt   (~2.4 GB)
    mt5          google/mt5-small                           (~1.2 GB; use --hf-name for mt5-base)
    indictrans2  ai4bharat/indictrans2-indic-en-dist-200M   (needs `pip install IndicTransToolkit`)

Handling Roman-script Hinglish: none of these models was pretrained on Roman Hindi
as a language of its own, so the input needs a choice of "source language":
    * mbart: source tag hi_IN (Hindi). The tokenizer treats Roman text as normal subwords;
      the English-looking parts are understood, the Roman-Hindi parts are the hard part.
    * mt5: no language tags; we add a text prefix "translate Hinglish to English: ".
    * indictrans2: expects Devanagari Hindi. We pass source tag hin_Deva. The optional
      --translit flag first converts Roman -> Devanagari (rule-based ITRANS, so it is only
      approximate); compare scores with and without it and describe the result in the report.

Memory tips for Colab/Kaggle T4 (16 GB): --batch-size 8 --grad-accum 4, mixed precision is
switched on automatically on a GPU (bf16 if available, otherwise fp16, never for mT5 which
becomes unstable in fp16 and is then run in fp32).

Outputs (TAG = finetune_<model>, or finetune_<model>_zeroshot when --epochs 0):
    results/TAG.json   results/TAG_errors.csv
Checkpoints are saved to checkpoints/ (gitignored, never commit them).
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make src/ importable

from evaluate import evaluate, save_errors, save_result  # noqa: E402
from preprocess import read_tsv  # noqa: E402

DATA_DIR = Path("data/processed")
HF_NAMES = {
    "mbart": "facebook/mbart-large-50-many-to-many-mmt",
    "mt5": "google/mt5-small",
    "indictrans2": "ai4bharat/indictrans2-indic-en-dist-200M",
}
MT5_PREFIX = "translate Hinglish to English: "


class Translator:
    """Wraps one pretrained model so that the training loop is the same for all of them."""

    def __init__(self, kind, hf_name, device, translit=False):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.kind, self.device, self.translit = kind, device, translit
        if kind == "mbart":
            from transformers import MBart50TokenizerFast
            self.tok = MBart50TokenizerFast.from_pretrained(hf_name, src_lang="hi_IN", tgt_lang="en_XX")
            self.gen_kwargs = {"forced_bos_token_id": self.tok.convert_tokens_to_ids("en_XX")}
        elif kind == "indictrans2":
            from IndicTransToolkit.processor import IndicProcessor  # pip install IndicTransToolkit
            self.tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            self.ip = IndicProcessor(inference=True)
            self.gen_kwargs = {}
        else:  # mt5
            self.tok = AutoTokenizer.from_pretrained(hf_name)
            self.gen_kwargs = {}
        kwargs = {"trust_remote_code": True} if kind == "indictrans2" else {}
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_name, **kwargs).to(device)

    # ---- input / output text handling -------------------------------------------------
    def prepare_source(self, sentences):
        """Apply the model-specific input format described in the module docstring."""
        if self.kind == "mt5":
            return [MT5_PREFIX + s for s in sentences]
        if self.kind == "indictrans2":
            if self.translit:
                from indic_transliteration import sanscript  # pip install indic-transliteration
                sentences = [sanscript.transliterate(s, sanscript.ITRANS, sanscript.DEVANAGARI)
                             for s in sentences]
            return self.ip.preprocess_batch(sentences, src_lang="hin_Deva", tgt_lang="eng_Latn")
        return list(sentences)  # mbart: plain text, the tokenizer adds the hi_IN tag

    def encode(self, sources, targets=None, max_len=64):
        """Tokenise a batch; with targets it also returns the `labels` used for the loss."""
        sources = self.prepare_source(sources)
        enc = self.tok(sources, text_target=targets, max_length=max_len, truncation=True,
                       padding=True, return_tensors="pt")
        if targets is not None:
            # -100 tells the loss function to ignore padding positions.
            enc["labels"][enc["labels"] == self.tok.pad_token_id] = -100
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def translate(self, sentences, beams, batch_size=32, max_new_tokens=64):
        self.model.eval()
        out = []
        for i in range(0, len(sentences), batch_size):
            enc = self.encode(sentences[i:i + batch_size])
            gen = self.model.generate(**enc, num_beams=beams, max_new_tokens=max_new_tokens,
                                      **self.gen_kwargs)
            decoded = self.tok.batch_decode(gen, skip_special_tokens=True)
            if self.kind == "indictrans2":
                decoded = self.ip.postprocess_batch(decoded, lang="eng_Latn")
            out.extend(d.strip() for d in decoded)
        return out


def autocast_dtype(kind, device):
    """Mixed precision choice: faster/lighter on GPU, but never fp16 for mT5."""
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None if kind == "mt5" else torch.float16


def train(tr, train_pairs, val_pairs, args, device):
    """Standard fine-tuning loop: AdamW, linear warmup/decay, gradient accumulation."""
    from transformers import get_linear_schedule_with_warmup

    model = tr.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(train_pairs) + args.batch_size - 1) // args.batch_size
    total_updates = max(1, args.epochs * steps_per_epoch // args.grad_accum)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_updates), total_updates)
    dtype = autocast_dtype(args.model, device)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    best, ckpt = float("inf"), Path(f"checkpoints/finetune_{args.model}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_pairs)
        t0, running, n = time.time(), 0.0, 0
        for step, i in enumerate(range(0, len(train_pairs), args.batch_size), 1):
            chunk = train_pairs[i:i + args.batch_size]
            batch = tr.encode([s for s, _ in chunk], [t for _, t in chunk])
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                loss = model(**batch).loss / args.grad_accum  # HF computes the loss from `labels`
            scaler.scale(loss).backward()
            running, n = running + loss.item() * args.grad_accum, n + 1
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        val_loss = evaluate_loss(tr, val_pairs, args.batch_size, dtype, device)
        print(f"epoch {epoch}  train {running / n:.3f}  val {val_loss:.3f}  ({time.time() - t0:.0f}s)")
        if val_loss < best:  # keep the best epoch only
            best = val_loss
            model.save_pretrained(ckpt)
            tr.tok.save_pretrained(ckpt)
    if ckpt.exists():  # reload the best epoch for the final evaluation
        from transformers import AutoModelForSeq2SeqLM
        kwargs = {"trust_remote_code": True} if args.model == "indictrans2" else {}
        tr.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, **kwargs).to(device)
    return best


@torch.no_grad()
def evaluate_loss(tr, pairs, batch_size, dtype, device):
    tr.model.eval()
    total, n = 0.0, 0
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        batch = tr.encode([s for s, _ in chunk], [t for _, t in chunk])
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            total += tr.model(**batch).loss.item()
        n += 1
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser(description="Fine-tune pretrained MT models on PHINC")
    p.add_argument("--model", choices=HF_NAMES, required=True)
    p.add_argument("--hf-name", help="override the hub model name, e.g. google/mt5-base")
    p.add_argument("--epochs", type=int, default=3, help="0 = zero-shot (no training)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch-size x this")
    p.add_argument("--lr", type=float, default=3e-5, help="mT5 usually likes a higher rate, e.g. 3e-4")
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--translit", action="store_true", help="indictrans2 only: Roman -> Devanagari first")
    p.add_argument("--max-train", type=int, default=0, help="use only N training pairs (smoke test)")
    p.add_argument("--max-test", type=int, default=0, help="use only N test pairs (smoke test)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_pairs = read_tsv(DATA_DIR / "train.tsv")
    val_pairs = read_tsv(DATA_DIR / "val.tsv")
    test_pairs = read_tsv(DATA_DIR / "test.tsv")
    if args.max_train:
        train_pairs, val_pairs = train_pairs[:args.max_train], val_pairs[:max(8, args.max_train // 4)]
    if args.max_test:
        test_pairs = test_pairs[:args.max_test]

    tr = Translator(args.model, args.hf_name or HF_NAMES[args.model], device, args.translit)
    best = None
    if args.epochs > 0:
        best = train(tr, train_pairs, val_pairs, args, device)

    hyps = tr.translate([s for s, _ in test_pairs], args.beam)
    refs = [t for _, t in test_pairs]
    scores = evaluate(hyps, refs)
    print("TEST:", scores)

    tag = f"finetune_{args.model}" + ("_zeroshot" if args.epochs == 0 else "")
    save_result({**scores, "mode": "zero-shot" if args.epochs == 0 else "fine-tuned",
                 "best_val_loss": None if best is None else round(best, 4),
                 "hyperparameters": vars(args)}, f"results/{tag}.json", model=tag)
    save_errors([s for s, _ in test_pairs], hyps, refs, f"results/{tag}_errors.csv")
    print(f"saved results/{tag}*")


if __name__ == "__main__":
    main()

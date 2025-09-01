## Spam SMS Classification — Fine-Tuning GPT-2 

A compact, end-to-end project that fine-tunes a pretrained GPT-2 model to classify SMS messages as spam or not spam (ham). The workflow runs from data acquisition → balanced sampling → tokenization & padding → model surgery (adding a classifier head) → training & evaluation → inference & model persistence. It’s designed to be easy to read for recruiters and engineers skimming your repo.
(Technically: decoder-only Transformer, causal self-attention; complexity per layer ≈ O(B · L² · d) where B=batch, L=sequence length, d=hidden size. With L≈120 the compute is modest and Colab-friendly.)

### Highlights

**Dataset:** UCI SMS Spam Collection (originally ~5.5k ham / ~0.7k spam).
(Plain-text TSV with two columns Label/Text; typical SMS length ≤ ~120 GPT-2 tokens; mild noise/abbreviations improving robustness.)

**Balance handling:** To keep training simple and fast, ham is undersampled to match spam, yielding a balanced set for clearer learning curves and shorter training time.
(Post-balancing size ≈ 2 × #spam examples; shuffling with a fixed seed prevents distribution drift; alternative for prod: class weights or oversampling to retain all ham.)

**Tokenizer:** GPT-2 BPE via tiktoken, with the special <|endoftext|> pad token (ID 50256) for right-padding.
(GPT-2 has no native PAD; using <|eot|> is a common workaround. We also construct an attention mask implicitly via fixed-length tensors; since we always read the final position’s logits, the pad token acts as a consistent summary slot.)

**Batching:** All sequences are padded to the longest training message (≈ 120 tokens) for uniform batch tensors.
(Right-padding preserves full information; no truncation on short messages. Memory scales linearly with B and quadratically with L due to attention.)

**Model:** Start from GPT-2 small (124M), replace the LM head with a 2-class linear head, and fine-tune only the new head + last transformer block + final LayerNorm.
(Trainable params ≈ last block (~10–11M) + head (~1.5k) + LN (~1.5k); the rest (~113M) frozen → fast convergence with limited compute.)

**Loss/Target:** Cross-entropy computed on the last token’s logits (efficient sequence-level target for decoder-only models).
(Formally CE = -log softmax(z_last)[y]; argmax(z_last) equals argmax(softmax(z_last)), so softmax is optional for prediction.)

**Training:** AdamW(lr=5e-5, weight_decay=0.1), 5 epochs, batch_size=8, lightweight periodic eval during training.
(Defaults betas=(0.9,0.999), eps=1e-8). With ~1.0–1.1k train examples → ~130 steps/epoch → ~650 optimizer steps total. Optional: gradient clipping, mixed precision for GPUs.)

**Outputs:** Printed train/val/test accuracy & loss, PDF plots for loss/accuracy, a saved .pth checkpoint, and a simple classify_review() for inference.
(Checkpoint contains the full model state; size comparable to GPT-2 small weights. Consider safetensors or 16-bit EMA for leaner storage.)

**What this project demonstrates (at a glance)**

**Responsible dataset handling:** Download, unzip, normalize labels (ham→0, spam→1), balance classes, and split into train/val/test (70/10/20) with fixed seeds.
(Shuffle → split after balancing to avoid cross-split leakage; deterministic via random_state=123.)

**Production-style dataloaders:** A custom SpamDataset that encodes, truncates, and pads once up-front for speed and reproducibility.
(Pre-tokenization amortizes cost; ensures consistent shapes ([B, L]) and dtypes (torch.long).)

**LLM → classifier:** Freeze most of GPT-2, swap the 50,257-vocab head for a 2-logit head, and fine-tune a thin slice on CPU/GPU for quick adaptation.
(Selective unfreezing stabilizes training and reduces overfitting; retains general linguistic features from pretraining.)

**Clear evaluation:** Loss & accuracy curves, plus “quick estimates during training” vs “full-dataset metrics” at the end.
(Intermittent eval uses a capped number of batches for speed; final eval uses all batches for unbiased metrics.)

**Simple inference:** One helper classify_review(text, ...) returns "spam" or "not spam".
(Mirrors training-time preprocessing to avoid train/serving skew; single forward pass with torch.no_grad().)

## Data & Preprocessing
Download & structure

## Source: UCI ML Repo — SMS Spam Collection.

The notebook downloads a ZIP, extracts it, and renames the raw file to sms_spam_collection/SMSSpamCollection.tsv.

**Load with:**

pd.read_csv("sms_spam_collection/SMSSpamCollection.tsv",
            sep="\t", header=None, names=["Label", "Text"])


(Explicit schema avoids header misreads; TSV prevents comma collision within messages.)

**Balance & splits**

**Balancing:** Randomly undersample ham to match the number of spam rows (seeded for repeatability).
(Ensures 50/50 prior → accuracy becomes informative; reduces epoch time.)

**Encode labels:** {"ham": 0, "spam": 1}.
(Stored as int64 in Pandas; cast to torch.long for loss.)

**Split:** 70% train / 10% validation / 20% test, with shuffling and fixed seed.
(Stratified by construction because of exact class balance; proportions preserved.)

**Persist:** Save train.csv, validation.csv, test.csv for reuse and faster iteration.
(Keeps preprocessing deterministic across sessions; decouples training from raw download.)

**Tokenization, Padding, and Dataloaders**
**Why padding?**

Transformers expect uniform tensor shapes per batch. Padding to the longest training message preserves information (vs truncating everything to the shortest) while keeping batching simple and efficient.
(Right-padding chooses a consistent “summary position” at the last index; the final hidden state attends to the full prefix.)

**Implementation**

Tokenizer: GPT-2 BPE via tiktoken; pad token ID 50256 (<|endoftext|>).
(Byte-level BPE is robust to emojis/rare symbols common in SMS.)

**SpamDataset:**

Pre-tokenizes all texts once at initialization.

If max_length=None (training), it discovers the true max (≈120 tokens).

Validation/test are padded to the training max; any longer texts are truncated.
(All sequences → shape [L]; batches → [B, L]; labels → [B].)

DataLoader: batch_size=8, shuffle=True for training; seeds fixed for reproducibility.
(num_workers=0 suits Colab; set pin_memory=True on CUDA for faster host→device copies.)

**Model:** From GPT-2 LM to 2-Class Classifier

**Config:** GPT-2 small — context 1024, vocab 50257, embedding 768, 12 layers, 12 heads.
(Causal mask ensures autoregressive flow; rotary/ALiBi not used in vanilla GPT-2.)

**Weights:** Load pretrained GPT-2 weights via a helper (e.g., download_and_load_gpt2(...)).
(Sets model.eval() for sanity checks; switch to train() during fine-tuning.)

**Freeze:** All parameters frozen by default, then unfreeze only:

the last transformer block,

the final LayerNorm,

and a new Linear(emb_dim, 2) classification head that replaces out_head.
(Head params = 768×2 + 2 ≈ 1,538; negligible vs total. Selective unfreezing helps adapt high-level features while preserving earlier layers.)

**Why last token only?**

For decoder-only LLMs, the final token’s hidden state summarizes the sequence. Mapping that vector to 2 logits (spam vs ham) is an efficient, well-established approach for sequence-level classification without adding special “CLS” tokens.
(Equivalent to using a learned summary position; avoids modifying tokenizer/vocab.)

## Training & Evaluation
**Objectives & loops**

**Loss:** Cross-entropy on model(inputs)[:, -1, :] (the last token’s logits).
(Optionally add label smoothing (e.g., ε=0.1) to regularize; omitted here for clarity.)

**Optimizer:** AdamW(lr=5e-5, weight_decay=0.1).
(In prod, exclude bias and LayerNorm from weight decay via param groups to preserve scale; here we keep defaults for simplicity.)

**Scheduler:** Not used (kept simple); easy to add if desired.
(Cosine decay with warmup often improves stability.)

**Epochs:** 5 (fast on CPU; very fast on GPU).
(With ~130 steps/epoch, wall-clock fits in a typical Colab GPU session.)

**Eval during training:** Lightweight (e.g., eval_iter=5 batches) to keep feedback snappy.
(Reduces training stall; final metrics run on full splits.)

**Full metrics after training:** Recompute train/val/test accuracy and loss over entire splits.
(Add precision/recall/F1 and ROC-AUC for imbalanced scenarios or threshold tuning.)

## Plots & artifacts

**loss-plot** — train vs val loss over epochs (with a second x-axis for “examples seen”).

**accuracy-plot** — train vs val accuracy over epochs.

**review_classifier** — saved model weights for reuse.
(Save also the tokenizer spec and max_length to ensure consistent serving.)

Expect initial accuracy near 50% (balanced random) before fine-tuning. Accuracy improves within a few epochs; exact numbers depend on seed, hardware, and minor environment differences.
(Monitor train–val gap for overfitting; adjust weight_decay/dropout or freeze fewer layers if underfitting.)

Inference (Use the model as a spam classifier)
text_spam = "You are a winner... receive $1000 cash or a $2000 award."
print(classify_review(text_spam, model, tokenizer, device,
                      max_length=train_dataset.max_length))
# → "spam"

text_ham = "Hey, just checking if we're still on for dinner tonight?"
print(classify_review(text_ham, model, tokenizer, device,
                      max_length=train_dataset.max_length))
# → "not spam"


classify_review() mirrors dataset preprocessing: encode → (truncate to supported context) → pad → forward pass → argmax over 2 logits.
(Run under torch.no_grad(); for throughput, batch multiple texts and enable autocast on GPU.)

Reproducibility & Design Notes

Seeds: Use torch.manual_seed(123) and seeded DataFrame shuffles for stable splits.

Context length: Assert the longest training sequence ≤ 1024 (GPT-2 context).

Padding ID: GPT-2 special token 50256 (<|endoftext|>).

Shape fix: Use model.pos_emb.weight.shape[0] for supported context length (prevents accidental truncation to the embedding dimension).

Why undersampling? Keeps runs fast and class balance clean for teaching. For production, consider full data with class weights, focal loss, or oversampling.
(Also consider threshold calibration (Platt/temperature scaling) for operational use.)

How to run (Colab-friendly)

Run the notebook top-to-bottom.

It downloads data and creates train.csv / validation.csv / test.csv.

Initializes tokenizer/datasets/dataloaders.

Loads GPT-2 weights, swaps the head, sets trainable layers.

Trains for 5 epochs with periodic eval.

Plots loss/accuracy and prints final metrics.

Saves review_classifier.pth.

Test classify_review(...) with your own SMS text.
(On CUDA: set model.to('cuda'), pin_memory=True, and consider torch.cuda.amp.autocast().)

Libraries Used

Core: torch, torch.utils.data, pandas, numpy

Tokenizer: tiktoken (GPT-2 BPE)

Utils: urllib.request, ssl, zipfile, pathlib, os

Viz: matplotlib
(Python ≥3.10; PyTorch 2.x. Match CUDA build to your runtime if using GPU.)

Extensions & Next Steps

Metrics: Add precision/recall/F1, ROC-AUC, and a confusion matrix.

Regularization: Tune transformer drop_rate and optimizer weight_decay; add early stopping.

Data: Train on the full dataset (no undersampling) with class weighting.

Head variants: Try mean-pooling all token states or a small MLP head.

Tokenization: Compare GPT-2 BPE vs WordPiece; experiment with smaller context lengths for speed.

Deployment: Export to TorchScript or ONNX; wrap inference in a minimal FastAPI service.
(For privacy, consider client-side tokenization and PII scrubbing before inference.)

Files Produced (during one run)

sms_spam_collection/SMSSpamCollection.tsv (raw TSV)

train.csv, validation.csv, test.csv

loss-plot.pdf, accuracy-plot.pdf

review_classifier.pth (saved weights)
(Optionally store max_length.json and tokenizer.json for portable serving.)

Why this project is portfolio-worthy

Shows clean ML reasoning: class balance, seeded splits, appropriate loss/metrics.

Demonstrates Transformer adaptation for classification (head swap + selective fine-tuning).

Clear separation of concerns: data, dataloaders, model, training, evaluation, inference.

Reproducible, light enough for CPU/Colab, yet illustrates LLM fine-tuning principles end-to-end.
(Also communicates hardware efficiency and practical trade-offs that matter in real teams.)

### **SPAM SMS CLASSIFICATION AND DATA ANALYSIS — GPT-2 FINE-TUNING**

**Introduction**:
SMS spam undermines user trust, drives customer support costs, and can expose users to scams and phishing. Traditional keyword rules and shallow models often fail against obfuscation (e.g., “fr££”, zero-width spaces, URL shorteners). This project fine-tunes a pretrained **GPT-2 (small, 124M)** language model for **binary classification** (spam vs. ham) on the **UCI SMS Spam Collection**, combining the linguistic breadth of a large LM with a lightweight, production-friendly head for robust detection.

The analysis is crucial for several reasons:

**Reducing fraudulent exposure:** Early, accurate detection blocks malicious outreach, limiting financial and reputational damage.
**Customer trust & safety:** Fewer false negatives (missed spam) and calibrated thresholds improve user safety; fewer false positives preserve legitimate conversations.
**Operational efficiency:** A high-precision filter lowers manual review workload and downstream customer support volume.
**Adaptability to drift:** LM features generalize across obfuscation (unicode tricks, emoji, variants of “win/claim/cash”), improving resilience to evolving campaigns.
**Regulatory & audit readiness:** Clear dataset handling, deterministic splits, and transparent training/evaluation make the system easier to audit and govern.

---

**Objective**:
Develop, fine-tune, and validate a GPT-2–based classifier that accurately predicts whether an SMS is spam. Beyond raw accuracy, the goal is to build a **repeatable, auditable** ML workflow: deterministic preprocessing, balanced sampling for pedagogical clarity, rigorous evaluation, artifact logging (plots & checkpoints), and guidance for thresholding, calibration, and production hardening. Key metrics include **accuracy**, **loss** trends, and (optional) **precision/recall/F1** and **ROC-AUC** when operating on the original imbalanced distribution.

---

**Tools and Technologies Used:**

* **Python** for end-to-end development.
* **PyTorch**: model definition, selective fine-tuning, training loops.
* **tiktoken**: GPT-2 byte-pair encoding (BPE) tokenizer; robust to emojis/rare chars.
* **Pandas / NumPy**: data ingestion, balancing, split management.
* **Matplotlib**: training curves (loss/accuracy) saved as PDFs.
* **Colab/CPU/GPU**: runs efficiently on CPU; very fast on common GPUs.

*(Artifacts produced during a run: `loss-plot.pdf`, `accuracy-plot.pdf`, `review_classifier.pth`, and cached CSV splits.)*

---

**Modeling Approaches Used:**

* **GPT-2 (small, 124M) with classification head**:
  Freeze the full backbone, **unfreeze last transformer block + final LayerNorm + a new Linear head (768→2)**. Train on **cross-entropy** computed at the **last token’s** logits.
  *Why last token?* In a causal decoder, the final position attends to the entire prefix—serving as a learned summary without introducing a special CLS token.

* *(Optional baseline, if added for comparison)* **Logistic Regression / Linear SVM** on simple text features (n-grams, TF-IDF) for interpretability; useful for sanity checks and ablations.

**Results (summary)**:

* The LM-head approach converges quickly on the **balanced** subset (ham=spam), achieving accuracy well above the random 50% baseline within a few epochs.
* **Loss curves** trend down smoothly; **accuracy curves** rise and stabilize by epochs 4–5 (exact values vary by seed/hardware).
* On the original imbalanced dataset (optional experiment), pair accuracy with precision/recall to avoid misleading conclusions.

**Interpretability**:

* While the GPT-2 classifier is not inherently linear/transparent, you can inspect influential tokens via saliency (grad×input) or perturbation tests. For stakeholder explainability, keep a **linear baseline** and **keyword analytics** (URLs, currency symbols, imperative verbs).

**Business Impact**:

* A calibrated threshold reduces harmful spam while minimizing false positives that frustrate users. The lightweight head and selective unfreezing enable **cost-efficient retraining** as spam patterns drift.

---

**IMPLEMENTATION:**
Open the training notebook in Colab: **[Open Colab Notebook](https://github.com/Aishwarya-chen11/Fine-tuned-LLM-Classification-Model/blob/main/Fine_tuned_LLM_classification_model.ipynb)**

Source dataset (UCI SMS Spam Collection): https://archive.ics.uci.edu/dataset/228/sms+spam+collection.

---

**Data Preprocessing:**

* **Loading**: Read the TSV into a DataFrame with columns `["Label","Text"]`.
* **Cleaning (lightweight)**: Trim whitespace; drop empty rows. (Optional) Normalize unicode (NFKC) to mitigate homoglyph obfuscation.
* **Label Encoding**: Map `ham→0`, `spam→1`.
* **Balancing (educational mode)**: Undersample ham to match the spam count (≈747 each). Creates a **50/50** prior so **accuracy** is interpretable and training is fast.
* **Deterministic Splits**: Shuffle with a fixed seed and split **70/10/20** into train/val/test; persist as `train.csv`, `validation.csv`, `test.csv`.

**Tokenization & Padding**:

* **Tokenizer**: GPT-2 BPE (byte-level) via `tiktoken` → robust to emoji, URLs, rare symbols.
* **Max length**: Discover **true max** from the training set (≈120 tokens for this corpus).
* **Padding**: Right-pad with GPT-2’s `<|endoftext|>` (ID **50256**) to a fixed length `L` (train max). Val/Test are padded/truncated to match `L`.
* **Dataset / DataLoader**: Pre-tokenize once for speed; return tensors of shape **`[B, L]`** (input ids) and **`[B]`** (labels). Train with `batch_size=8`, `drop_last=True`; eval without dropping.

---

**Training & Evaluation:**

* **Backbone**: GPT-2 small (context **1024**, vocab **50257**, emb **768**, 12 layers, 12 heads).
* **Head**: `Linear(768,2)` replaces the LM head.
* **Selective unfreezing**: Train only the **last block**, **final LayerNorm**, and **the new head**—reduces compute and overfitting.
* **Objective**: Cross-entropy on **last token** logits: `model(x)[:, -1, :]` vs. labels.
* **Optimizer**: `AdamW(lr=5e-5, weight_decay=0.1)`; epochs **5**; seed **123**.
* **Eval cadence**: Lightweight interim eval (`eval_iter=5` batches) for fast feedback; final metrics on full splits.
* **Artifacts**:

  * **ROC/PR (optional)** if running on the original imbalanced distribution.
  * **Loss curve** → `loss-plot.pdf`
  * **Accuracy curve** → `accuracy-plot.pdf`
  * **Checkpoint** → `review_classifier.pth`

---

**Model Performance Comparison:**

**Training curves**

* *Loss*: Monotonically decreasing for train/val, with minimal gap → low overfitting on balanced subset.
* *Accuracy*: Increases steadily; stabilizes after epochs 4–5.

**Confusion trends** (typical):

* False negatives often contain subtle spam with polite tone and no obvious keywords.
* False positives may include legitimate messages containing amounts (“\$20”) or links (meeting links, OTP portals).

<img src="https://github.com/Aishwarya-chen11/Fine-tuned-LLM-Classification-Model/blob/main/download_2.png" alt="Description" width="600"/>

<img src="https://github.com/Aishwarya-chen11/Fine-tuned-LLM-Classification-Model/blob/main/download_3.png.jpg" alt="Description" width="600"/>

**Suggested additional metrics**

* **Precision/Recall/F1** at default threshold.
* **ROC-AUC** and **PR-AUC** (more informative under class imbalance).
* **Calibration** (temperature scaling) for probability-based policies.

*(Attach images from your run; e.g.,)*

* Loss curve: `artifacts/loss-plot.pdf`
* Accuracy curve: `artifacts/accuracy-plot.pdf`

---

**Analysis and Conclusion:**
Fine-tuning only the final block and head yields strong accuracy on the balanced subset with minimal compute—ideal for education and rapid prototyping. The approach is resilient to obfuscation (thanks to byte-level BPE and LM pretraining) and easily extended: unfreeze additional blocks for capacity, add dropout for regularization, incorporate class weights for the original, imbalanced distribution, and layer on explainability where required.

---

**Actionable Insights and Recommendations:**

**1. Model choice & scope**
Use the **GPT-2 classifier head** for production-like performance under tight compute. Keep a **linear baseline** for explainability and regression tests.

**2. Thresholding & policy**
Tune the decision threshold to optimize cost-weighted objectives (e.g., penalize false negatives more heavily). Pair with **calibration** to support policy rules (“block if p(spam) ≥ 0.9; review if 0.6–0.9”).

**3. Imbalance handling (original corpus)**
If training on the full imbalanced set, prefer **class-weighted CE** or **focal loss** over simple undersampling to preserve data and improve minority recall.

**4. Drift monitoring**
Spam tactics evolve (new domains, Unicode, shorteners). Schedule **periodic re-fine-tuning** (selective unfreezing) and monitor live **precision/recall**. Maintain a **shadow evaluation** stream for new templates.

**5. Feature logging & guardrails**
Log indicative features (URL count, currency symbols, phone numbers, imperative verbs) for auditability and quick triage. Use **PII scrubbing** in logs.

**6. Explainability**
For stakeholders/regulators, attach **token-level saliency** or **perturbation** explanations; maintain a simple interpretable model as a reference.

**7. Deployment & latency**
Export a lightweight **PyTorch** checkpoint; batch inputs for throughput; use **AMP** on GPU. At the edge, consider a distilled or quantized variant.

**8. Security hygiene**
Normalize text (NFKC), strip zero-width characters, and expand URL shorteners before inference where possible.

**9. Data pipeline quality**
Deduplicate messages, drop empty/garbled rows, and ensure deterministic splits (seeded). Cache tokenized sequences for faster cold starts.

**10. Continual improvement**
Collect hard negatives (false negatives/positives) into a feedback dataset for scheduled improvement cycles.

---

**EXPLORATORY DATA ANALYSIS**

*(Run EDA on raw data before balancing to understand real-world priors.)*

**1. Class Ratio (Pie / Bar)**

* **Insight:** Ham dominates (≈85–87%), spam ≈13–15%. This imbalance explains why **accuracy alone** is misleading; pair with recall/PR-AUC.
* **Recommendation:** Use **class-weighted loss** at scale; report **per-class** metrics.

**2. Message Length Distribution (Density / Box)**

* **Insight:** Spam tends to be slightly longer and more variable; ham clusters shorter with conversational tone.
* **Recommendation:** Feature engineering (for baselines): include **char/tok length**; for LLM pipeline, keep **L≈120** to retain content yet limit compute.

**3. URLs, Numbers, Currency Symbols (Box / Countplots)**

* **Insight:** Spam has higher counts of **URLs**, **digits**, and **currency symbols** (`$, £`), plus words like “free, claim, prize, call”.
* **Recommendation:** Add **URL normalization/expansion** and domain reputation features to reduce false negatives; maintain a curated allowlist for internal links to reduce false positives.

**4. Capitalization & Punctuation (Box / Violin)**

* **Insight:** Excessive **ALL-CAPS**, **exclamation marks**, and **urgent cues** correlate with spam.
* **Recommendation:** Monitor these as drift signals; for baselines, include them as engineered features.

**5. Time-based Patterns (If timestamps available)**

* **Insight:** Spam campaigns often spike at certain hours/days.
* **Recommendation:** Add **temporal features** to downstream risk scoring or rate limiting.

---

**Conclusion**
The GPT-2 fine-tuning approach provides a practical, compute-efficient path to high-quality SMS spam detection. With deterministic preprocessing, selective unfreezing, and clear evaluation, this project is **deployment-ready** in spirit and **teachable** in form. Extending with calibration, class-weighted training on the full distribution, and drift-aware monitoring will further harden the system for production.

---

By adopting the recommendations above—especially **threshold calibration**, **class-weighted loss** on the original distribution, and **continuous monitoring**—teams can reduce spam exposure while maintaining excellent user experience and auditability.

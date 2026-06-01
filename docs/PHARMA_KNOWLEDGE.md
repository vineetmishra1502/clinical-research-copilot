# Pharma Knowledge Reference

**Purpose:** Complete reference for every medical term, drug, biomarker,
study design, and clinical outcome used in this project. Written for
someone with a software/data background, not a clinical background.

**Scope:** Covers the 5 drugs in our knowledge base, the cancer types
they treat, clinical trial terminology, biomarkers, evidence hierarchy,
and outcomes vocabulary used across all retrieval queries.

---

## Table of Contents

1. [How Cancer Works — The Basics](#1-how-cancer-works--the-basics)
2. [The 5 Drugs in Our Knowledge Base](#2-the-5-drugs-in-our-knowledge-base)
3. [The Immune System and Checkpoint Inhibitors](#3-the-immune-system-and-checkpoint-inhibitors)
4. [Biomarkers — What They Are and Why They Matter](#4-biomarkers--what-they-are-and-why-they-matter)
5. [PD-L1, TPS, and CPS — Explained in Detail](#5-pd-l1-tps-and-cps--explained-in-detail)
6. [Cancer Types in Our Database](#6-cancer-types-in-our-database)
7. [Clinical Outcomes — What Trials Measure](#7-clinical-outcomes--what-trials-measure)
8. [Study Types and Evidence Hierarchy](#8-study-types-and-evidence-hierarchy)
9. [Clinical Trial Terminology](#9-clinical-trial-terminology)
10. [Diabetes — Separate Disease, Same System](#10-diabetes--separate-disease-same-system)
11. [Why These 5 Drugs Together](#11-why-these-5-drugs-together)
12. [Quick Reference Glossary](#12-quick-reference-glossary)

---

## 1. How Cancer Works — The Basics

Normal cells in your body divide, do their job, and die in a controlled
cycle. Cancer happens when a cell's DNA gets damaged in a way that
removes the brakes on this cycle. The cell starts dividing uncontrollably,
ignores signals to stop, and eventually forms a tumor.

Two key things cancer cells do that make them dangerous:

**They grow without permission.** Normal cells only divide when they
receive growth signals from neighboring cells. Cancer cells produce their
own growth signals or become hypersensitive to tiny amounts of signal.
This is why drugs like osimertinib (which blocks a specific growth signal
receptor called EGFR) are effective — they cut off the fuel supply.

**They hide from the immune system.** Your immune system has killer
T-cells that scan for abnormal cells and destroy them. Cancer cells have
evolved several tricks to evade this surveillance. The most important one
for our drugs: expressing PD-L1 on their surface, which sends a "don't
kill me" signal to T-cells. Pembrolizumab and nivolumab block this trick.

---

## 2. The 5 Drugs in Our Knowledge Base

### 2.1 Pembrolizumab (Keytruda) — Merck

**Drug class:** PD-1 inhibitor (checkpoint inhibitor / immunotherapy)

**How it works:** Cancer cells wear a protein called PD-L1 on their
surface. This protein binds to PD-1 receptors on T-cells and sends the
signal "I am a normal cell, do not attack me." Pembrolizumab is a
monoclonal antibody that blocks the PD-1 receptor on T-cells. With PD-1
blocked, the T-cells can no longer receive the "don't attack" signal.
They recognize the cancer cells as foreign and destroy them.

Think of it as: cancer cells are wearing a fake ID. Pembrolizumab takes
away the ability to read the fake ID. The bouncer (T-cell) now sees
the cancer cell has no valid ID and removes it.

**Approved cancer types (as of 2025):**
- Non-small cell lung cancer (NSCLC) — first line alone (TPS ≥50%) or with chemotherapy
- Small cell lung cancer (SCLC)
- Head and neck squamous cell carcinoma (HNSCC)
- Melanoma — one of the earliest approvals, still a major indication
- Triple-negative breast cancer (TNBC)
- Urothelial carcinoma (bladder cancer)
- Colorectal cancer (MSI-H/dMMR subtype)
- Gastric and gastroesophageal junction cancer
- Cervical cancer
- Endometrial carcinoma
- Biliary tract cancer
- Malignant pleural mesothelioma (approved September 2024)
- HER2-positive gastric/GEJ cancer in combination with trastuzumab
  (approved March 2025)
- Resectable locally advanced HNSCC as neoadjuvant (approved June 2025)

**Key clinical trials in our database:**
- KEYNOTE-024: First-line NSCLC TPS ≥50%, pembrolizumab vs chemotherapy
- KEYNOTE-189: First-line non-squamous NSCLC with chemotherapy
- KEYNOTE-407: First-line squamous NSCLC with chemotherapy
- KEYNOTE-355: Triple-negative breast cancer
- KEYNOTE-716: Melanoma adjuvant
- KEYNOTE-483: Mesothelioma (2024)

**Trade name:** Keytruda
**Route:** Intravenous infusion every 3 or 6 weeks

**Why it matters for our system:** Pembrolizumab has the largest and most
complex literature of any cancer drug. Over 24,000 papers on PubMed.
It is the largest revenue-generating drug in the world. Our NSCLC and
melanoma retrieval queries are dominated by pembrolizumab evidence.

---

### 2.2 Nivolumab (Opdivo) — Bristol Myers Squibb

**Drug class:** PD-1 inhibitor (checkpoint inhibitor / immunotherapy)

**How it works:** Same mechanism as pembrolizumab — blocks PD-1 on
T-cells, preventing cancer cells from sending the "don't attack me"
signal. Pembrolizumab and nivolumab target the same protein (PD-1) but
are different antibody molecules with slightly different binding
characteristics.

**Key difference from pembrolizumab:** Nivolumab is often combined with
ipilimumab (a CTLA-4 inhibitor — a different checkpoint) for a
"dual checkpoint blockade" approach. This combination is approved for
melanoma, renal cell carcinoma, NSCLC, and hepatocellular carcinoma.
Pembrolizumab is not typically combined with ipilimumab.

**Approved cancer types (as of 2025):**
- NSCLC (with and without chemotherapy)
- Melanoma — first approved checkpoint inhibitor indication
- Renal cell carcinoma (alone and with ipilimumab — CheckMate-214)
- Hepatocellular carcinoma (with ipilimumab added to NCCN 2025 guidelines)
- Gastric/esophageal cancer
- Colorectal cancer (MSI-H)
- Urothelial carcinoma
- Mesothelioma
- Subcutaneous formulation approved December 2024 (Opdivo Qvantig)

**Key clinical trials in our database:**
- CheckMate-017 and CheckMate-057: NSCLC second-line
- CheckMate-066: First-line melanoma
- CheckMate-214: Renal cell carcinoma with ipilimumab
- CheckMate-649: Gastric cancer with chemotherapy

**Trade name:** Opdivo
**Route:** Intravenous infusion every 2, 4, or 6 weeks

**Why it matters for our system:** Nivolumab and pembrolizumab are direct
competitors — many queries explicitly compare the two. When our rewriter
detects "nivolumab vs pembrolizumab," it sets drug=null and generates
two sub-questions, one for each drug, so the system retrieves and
compares evidence from both.

---

### 2.3 Osimertinib (Tagrisso) — AstraZeneca

**Drug class:** EGFR inhibitor (targeted therapy / tyrosine kinase inhibitor)

**How it works:** About 10-15% of NSCLC patients (higher in Asian
populations, ~30-40%) have a mutation in the EGFR gene. EGFR is a
protein on cell surfaces that receives growth signals. When EGFR is
mutated, it gets stuck in the "on" position — constantly telling cells
to divide even without a growth signal. Osimertinib is a small molecule
that fits into the active site of the mutated EGFR protein and blocks it,
cutting off the constant growth signal.

Think of it as: a car where the accelerator pedal is stuck. Osimertinib
is a wedge that physically blocks the pedal from being pressed further.

**Critical point — why this is different from pembrolizumab:** Osimertinib
only works in EGFR-mutant NSCLC. In EGFR wild-type (normal gene) NSCLC,
it does nothing useful. And critically: pembrolizumab generally does NOT
work well in EGFR-mutant NSCLC. When patients have EGFR mutations, the
standard of care is osimertinib, not pembrolizumab. This is a major
source of confusion in the literature — always check for EGFR status.

**EGFR mutation types:**
- Exon 19 deletion (del19) — most common, ~45% of EGFR mutations
- L858R (exon 21 substitution) — second most common, ~40%
- T790M — a resistance mutation that develops after first-generation EGFR
  inhibitors; osimertinib was specifically designed to overcome T790M

**Approved indications:**
- First-line EGFR-mutant NSCLC (exon 19 del or L858R)
- T790M mutation-positive NSCLC after first-generation EGFR inhibitor
- Stage III EGFR-mutant NSCLC after chemoradiation (approved September 2024)
- Adjuvant treatment after tumor resection (ADAURA trial)

**Key clinical trials in our database:**
- FLAURA: First-line vs earlier EGFR inhibitors, OS benefit
- FLAURA2: Osimertinib + chemotherapy combination
- ADAURA: Adjuvant treatment, dramatically improved disease-free survival
- LAURA: Stage III after chemoradiation (September 2024)

**Trade name:** Tagrisso
**Route:** Oral tablet once daily

**Why it matters for our system:** Osimertinib is AstraZeneca's flagship
drug. The EGFR vs checkpoint inhibitor distinction is medically important — our retriever
must not confuse evidence from EGFR-mutant and EGFR wild-type populations.
The cancer_type=nsclc filter without EGFR status can mix fundamentally
different patient populations.

---

### 2.4 Trastuzumab (Herceptin) — Roche/Genentech

**Drug class:** HER2-targeted therapy (monoclonal antibody)

**How it works:** HER2 (Human Epidermal Growth Factor Receptor 2) is a
protein that helps cells grow. In normal cells, HER2 is present in small
amounts. In HER2-positive cancer, the HER2 gene is amplified — cells have
many more copies than normal, producing excessive HER2 protein. This
drives uncontrolled cell growth. Trastuzumab is a monoclonal antibody
that binds directly to HER2 protein on cancer cell surfaces and blocks
it from sending growth signals. It also marks cancer cells for destruction
by the immune system.

Think of it as: a cell phone that's permanently stuck on a "grow grow
grow" call. Trastuzumab hangs up the call and also puts a flag on the
phone so immune cells can find and destroy it.

**HER2 testing:** Before giving trastuzumab, HER2 status must be tested:
- IHC (immunohistochemistry) score 3+ = HER2 positive (definitely treat)
- IHC 2+ with ISH amplification = HER2 positive (treat)
- IHC 0-1+ = HER2 negative (do not treat with trastuzumab)
- HER2-low (IHC 1+ or 2+ without amplification) = newer category, some
  benefit from trastuzumab deruxtecan (T-DXd), a newer antibody-drug
  conjugate (not trastuzumab itself)

**Approved indications:**
- HER2-positive breast cancer — early stage (adjuvant) and metastatic
- HER2-positive gastric and gastroesophageal junction cancer
- HER2-positive breast cancer in combination with pembrolizumab
  (approved as part of KEYNOTE-119 studies)

**Key clinical trials in our database:**
- HERA: Adjuvant HER2-positive breast cancer
- CLEOPATRA: First-line metastatic HER2-positive breast cancer
  (trastuzumab + pertuzumab + docetaxel)
- EMILIA: Second-line with T-DM1 (trastuzumab emtansine)
- ToGA: HER2-positive gastric cancer

**Trade name:** Herceptin (original); biosimilars widely available
**Route:** Intravenous infusion or subcutaneous injection

**Historical significance:** Trastuzumab was approved in 1998 and
transformed HER2-positive breast cancer from one of the worst prognoses
to one of the most treatable subtypes. It demonstrated that targeting a
specific protein driver (rather than just killing all fast-dividing cells)
could dramatically improve outcomes. It opened the era of targeted therapy.

**Why it matters for our system:** 25+ years of literature means massive
evidence base. Trastuzumab papers frequently discuss HER2 testing,
combination regimens, resistance mechanisms, and newer HER2-targeted drugs
(T-DM1, T-DXd, pertuzumab). Our retriever needs to distinguish original
trastuzumab evidence from the newer HER2-targeted agents.

---

### 2.5 Metformin (Glucophage) — Generic/Multiple manufacturers

**Drug class:** Biguanide (antidiabetic / insulin sensitizer)

**How it works:** Metformin is fundamentally different from the other 4
drugs — it treats type 2 diabetes, not cancer. Its mechanism operates
through several interconnected pathways:

**Primary mechanism — mitochondrial complex I inhibition:**
Metformin accumulates inside mitochondria (the cell's energy factories)
and mildly inhibits Complex I of the respiratory chain. This reduces ATP
production and raises AMP levels. The cell detects the energy deficit.

**Secondary mechanism — AMPK activation:**
The raised AMP-to-ATP ratio activates AMPK (AMP-activated protein kinase),
the cell's master energy sensor. Think of AMPK as the low-battery warning
on a phone — when energy is low, AMPK tells the cell to stop expensive
activities and switch to energy conservation mode.

**Downstream effects of AMPK activation:**
- Liver: Stops gluconeogenesis (making new glucose from scratch) →
  blood glucose drops
- Liver: Reduces lipogenesis (fat production) → lower triglycerides
- Muscle: Improves insulin signaling → cells absorb glucose more
  efficiently
- Gut: Alters microbiome composition, changes glucose absorption
- Weight: Weight-neutral to modest weight loss (unlike many diabetes drugs)

**Clinical outcomes:**
- Reduces HbA1c (3-month average blood sugar) by 1-1.5%
- Lowers cardiovascular risk
- Does not cause hypoglycemia (dangerously low blood sugar) unlike insulin
- Available as a generic — very inexpensive
- First-line therapy for type 2 diabetes per all major guidelines

**Emerging research areas in our database:**
- Metformin in cancer prevention and treatment (AMPK activates tumor
  suppressor pathways)
- Metformin in polycystic ovary syndrome (PCOS)
- Metformin in aging/longevity research
- COVID-19 outcomes in diabetic patients on metformin

**Trade name:** Glucophage (original); generic metformin universally used
**Route:** Oral tablet, twice or three times daily with meals

**Why it matters for our system:** Metformin adds a non-oncology dimension
to the knowledge base, covering type 2 diabetes management, insulin
resistance, and metabolic disease. It also appears in cancer literature
as an investigational agent. Queries about metformin require understanding
that HbA1c, fasting glucose, and insulin resistance are the relevant
endpoints — not OS, PFS, or ORR which are cancer-specific.

---

## 3. The Immune System and Checkpoint Inhibitors

### 3.1 How the immune system normally fights cancer

Your immune system has several layers. For cancer, the key players are:

**T-cells (T lymphocytes):** The killer soldiers. They patrol the body
looking for abnormal cells. When a T-cell finds a cancer cell, it can
release chemicals that punch holes in the cancer cell membrane, killing it.

**Dendritic cells:** The scouts. They find cancer cells, take samples of
the cancer cell's proteins (antigens), and present them to T-cells. This
"briefing" activates the T-cells to recognize and kill that specific
cancer type.

**Natural killer (NK) cells:** A faster but less specific defense. They
kill cells that display stress signals without needing prior briefing.

### 3.2 How cancer evades the immune system

The most clinically important evasion mechanism for our drugs:

**PD-1/PD-L1 pathway:**
- PD-1 is a receptor on T-cells. When activated, it puts the T-cell to
  sleep — it stops attacking.
- PD-L1 is a protein displayed on the surface of normal cells that sends
  the "I am normal, don't kill me" signal to PD-1.
- Cancer cells hijack this system by expressing PD-L1 on their surfaces.
  They trick T-cells into standing down.

**CTLA-4 pathway (relevant for ipilimumab, often combined with nivolumab):**
- CTLA-4 is another brake on T-cells, active at the activation stage
  (before T-cells even get to the tumor).
- Ipilimumab blocks CTLA-4, releasing T-cells from their brakes earlier
  in the activation process.

### 3.3 What "checkpoint" means

The immune system has built-in brakes (checkpoints) to prevent it from
attacking normal tissues (autoimmunity). PD-1 and CTLA-4 are two of these
checkpoints. Checkpoint inhibitors remove these brakes in cancer patients,
allowing the immune system to attack cancer cells that were previously
being ignored.

**The trade-off:** Removing the brakes can cause the immune system to
attack normal organs. These are called immune-related adverse events (irAEs):
- Pneumonitis (lung inflammation) — particularly concerning in NSCLC
- Colitis (bowel inflammation)
- Hepatitis (liver inflammation)
- Thyroiditis (thyroid dysfunction)
- Skin reactions (rash, vitiligo)

Severe irAEs require stopping the checkpoint inhibitor and treating with
corticosteroids to suppress the immune system.

---

## 4. Biomarkers — What They Are and Why They Matter

A biomarker is a measurable biological indicator that tells you something
about a patient or their disease. In oncology, biomarkers answer questions:
- Will this drug work for this patient?
- Has the cancer responded to treatment?
- Is the cancer coming back?

**Predictive biomarker:** Predicts whether a patient will respond to a
specific treatment. PD-L1 TPS is predictive for pembrolizumab benefit.

**Prognostic biomarker:** Predicts how the disease will progress regardless
of treatment. EGFR mutation status is both predictive (for EGFR inhibitors)
and prognostic.

**The biomarkers in our database:**

| Biomarker | What it measures | Relevant drug |
|---|---|---|
| PD-L1 TPS | % of tumor cells expressing PD-L1 | Pembrolizumab, nivolumab |
| PD-L1 CPS | PD-L1 in tumor + immune cells | Pembrolizumab in gastric/bladder |
| EGFR mutation | Growth receptor gene mutation | Osimertinib |
| HER2 status | HER2 protein expression level | Trastuzumab |
| HbA1c | 3-month blood sugar average | Metformin |
| MSI-H/dMMR | DNA repair gene status | Pembrolizumab (any cancer) |
| TMB | Tumor mutation burden | Pembrolizumab (high TMB) |

---

## 5. PD-L1, TPS, and CPS — Explained in Detail

### 5.1 PD-L1

PD-L1 (Programmed Death-Ligand 1) is a protein. What matters is:
- Where it is expressed (tumor cells? immune cells? both?)
- How much of it is expressed (what percentage of cells have it?)

### 5.2 TPS — Tumor Proportion Score

**Definition:** The percentage of viable tumor cells showing PD-L1
membrane staining, out of all viable tumor cells in the sample.

```
TPS = (PD-L1 positive tumor cells) / (all viable tumor cells) × 100%
```

A pathologist looks at a tumor biopsy under a microscope with a special
PD-L1 stain. They count which cells are stained (PD-L1 positive) vs
unstained (PD-L1 negative).

**TPS categories in our database:**

| Value | What it means | Clinical implication |
|---|---|---|
| TPS < 1% | Negative — nearly no PD-L1 | Pembrolizumab monotherapy unlikely to help; chemotherapy preferred |
| TPS 1-49% | Low/intermediate expression | Some benefit possible; combination with chemotherapy typically used |
| TPS ≥ 50% | High expression | Pembrolizumab monotherapy approved first-line; strong survival benefit |

**The landmark numbers (KEYNOTE-024 trial, NSCLC):**
- Patients with TPS ≥50% treated with pembrolizumab:
  Median OS = 26.3 months
- Same patients treated with chemotherapy:
  Median OS = 13.4 months
- That is a 96% improvement in median survival

This is why TPS ≥50% is the most important filter in checkpoint inhibitor
queries. Mixing TPS <1% and TPS ≥50% patients would produce a meaningless
averaged result.

### 5.3 CPS — Combined Positive Score

Used for certain cancer types where immune cells inside the tumor also
express PD-L1 (gastric, bladder, cervical cancers).

```
CPS = (PD-L1 positive tumor cells + macrophages + lymphocytes)
      / (all viable tumor cells) × 100
```

Note: CPS can exceed 100. A CPS ≥1 is the common threshold for some
pembrolizumab indications (gastric cancer, HER2-positive gastric cancer
approved March 2025).

**Why TPS and CPS are both in our database:** Some papers report TPS
(mainly lung cancer), others report CPS (mainly gastric, bladder, cervical).
The same patient could have different scores on both scales. When the
metadata extraction catches "PD-L1 positive" without specifying which score,
we store it as "PD-L1-positive" — the general term.

### 5.4 PD-L1-positive vs TPS ≥50%

These are NOT the same thing:

```
PD-L1-negative  = TPS < 1%  (very little to no PD-L1)
PD-L1-positive  = TPS ≥ 1%  (any detectable PD-L1 — vague)
TPS ≥ 50%       = high PD-L1 (specific threshold — most clinically useful)
```

A patient is "PD-L1 positive" if TPS ≥1%. But that could mean TPS = 2%
(barely positive, poor response to monotherapy) or TPS = 95% (strongly
positive, excellent response). This is why specific TPS thresholds matter
far more than the general "positive/negative" label.

---

## 6. Cancer Types in Our Database

### 6.1 Lung Cancers

**NSCLC — Non-Small Cell Lung Cancer**
About 85% of all lung cancers. Divided into subtypes:
- Adenocarcinoma: Most common subtype, arises from mucus-secreting cells.
  Most likely to have EGFR mutations, ALK rearrangements. Peripheral lung.
- Squamous cell carcinoma: Arises from squamous cells lining airways.
  Less likely to have targetable mutations. Central lung. Often in smokers.
- Large cell carcinoma: Less common, heterogeneous group.

**Why NSCLC is our primary cancer type:**
Pembrolizumab, nivolumab, and osimertinib are all approved in NSCLC but
for completely different patient populations. Pembrolizumab works in
PD-L1-high NSCLC. Osimertinib works in EGFR-mutant NSCLC. These
populations often have INVERSE overlap — EGFR-mutant patients often have
low PD-L1 and respond poorly to checkpoint inhibitors.

**SCLC — Small Cell Lung Cancer**
About 15% of lung cancers. Much more aggressive. Associated almost
exclusively with smoking. Spreads very early. Limited treatment options.
Nivolumab has approval here. Osimertinib does not work in SCLC (no EGFR
mutation driver in SCLC).

### 6.2 Breast Cancers

Breast cancer is not one disease — it is several defined by what proteins
drive it:

**HER2-positive breast cancer**
The HER2 gene is amplified — cells have too many copies. About 15-20%
of breast cancers. Previously very aggressive; now one of the most
treatable subtypes because of drugs like trastuzumab. Trastuzumab
dramatically improves outcomes. The CLEOPATRA trial showed adding
pertuzumab to trastuzumab plus chemotherapy further improved OS.

**Triple-negative breast cancer (TNBC)**
Negative for estrogen receptor (ER-), progesterone receptor (PR-), and
HER2-. About 15-20% of breast cancers. No hormone or HER2 target to
use. Historically difficult to treat. Pembrolizumab is now approved
(KEYNOTE-355) for PD-L1-positive TNBC.

**HR-positive / HER2-negative**
The most common breast cancer subtype (~70%). Driven by estrogen and/or
progesterone. Treated with hormone blockers (tamoxifen, aromatase
inhibitors). Less overlap with our database drugs.

### 6.3 Melanoma

Skin cancer arising from melanocytes (pigment-producing cells). The first
cancer where checkpoint inhibitors demonstrated dramatic, durable responses.
About 40-50% of patients with advanced melanoma have BRAF V600E mutations
(treated with targeted BRAF/MEK inhibitors). Pembrolizumab and nivolumab
are both first-line options, sometimes as monotherapy, sometimes combined
(nivolumab + ipilimumab).

### 6.4 Renal Cell Carcinoma (RCC)

Kidney cancer. Clear cell RCC is the most common subtype. Highly
immunogenic (visible to the immune system). Excellent response rates to
checkpoint inhibitors. Nivolumab + ipilimumab (CheckMate-214) established
dual checkpoint blockade as a standard first-line option.

### 6.5 Colorectal Cancer (CRC)

**MSI-H / dMMR CRC:** A subgroup (about 5% of metastatic CRC) where DNA
mismatch repair genes are defective (dMMR = deficient mismatch repair,
MSI-H = microsatellite instability high). These tumors accumulate many
mutations, making them very visible to the immune system. Pembrolizumab
is approved as first-line treatment for MSI-H/dMMR CRC regardless of
PD-L1 status.

**MSS CRC (most colorectal cancers):** Mismatch repair-proficient (pMMR),
microsatellite stable. Do NOT respond to checkpoint inhibitors. This is
an important negative predictive biomarker — PDL1 testing is not useful
here; MSI/MMR testing is what matters.

### 6.6 Gastric / GEJ Cancer

Stomach and gastroesophageal junction cancer. Both pembrolizumab and
nivolumab are approved here in combination with chemotherapy. HER2
testing is done — HER2-positive gastric cancer can be treated with
trastuzumab (ToGA trial). In March 2025, pembrolizumab + trastuzumab +
chemotherapy was approved for HER2-positive gastric cancer.

CPS (not TPS) is used as the PD-L1 biomarker for gastric cancer.

---

## 7. Clinical Outcomes — What Trials Measure

These are the outcome measures stored in our `endpoints` metadata field.

### 7.1 OS — Overall Survival

**What it is:** How long patients live from the start of treatment (or
from randomization) until death from any cause.

**Why it matters:** The gold standard. It answers the most fundamental
question: "Does this drug help patients live longer?" It cannot be
manipulated or misinterpreted — the patient is either alive or not.

**How it is reported:**
- Median OS: The time point at which half the patients in a group have
  died. "Median OS of 26.3 months" means half the patients were still
  alive after 26.3 months.
- OS rate at X months/years: "1-year OS rate of 70%" means 70% of
  patients were still alive at 12 months.
- Hazard ratio (HR): Compares the rate of death between two groups.
  HR = 0.63 means the treated group had a 37% lower risk of dying at
  any given time point than the control group.

**Limitation:** Takes the longest to measure — patients must die to
contribute OS data. Can take 5+ years in some cancers.

### 7.2 PFS — Progression-Free Survival

**What it is:** How long from treatment start until the cancer grows
(progresses) or the patient dies, whichever comes first.

**Why it matters:** Faster to measure than OS. A tumor growing on a scan
is detectable months before the patient dies. Used as a surrogate for OS
in many trials, especially Phase II.

**Limitation:** Tumor growth does not always equal death, and drugs can
improve PFS without improving OS (if patients switch to effective second-
line therapies after progression). This is why OS remains the gold
standard despite PFS being easier to measure.

**How it is reported:** Same format as OS — median PFS, PFS rate at
X months, hazard ratio.

### 7.3 ORR — Objective Response Rate

**What it is:** The percentage of patients whose tumor shrinks by at
least 30% (partial response, PR) or disappears completely (complete
response, CR).

```
ORR = (patients with CR + patients with PR) / total patients × 100%
```

**Why it matters:** The fastest early signal that a drug is biologically
active against a cancer. Useful in Phase I/II trials where you need a
quick indication of whether a drug is worth developing further.

**Limitation:** Tumor shrinkage does not always translate to longer
survival. A drug can produce impressive ORR but fail to improve OS.

**Complete Response (CR):** Tumor completely disappears on all scans and
in all measurable sites. The gold standard response.

**Partial Response (PR):** Tumor shrinks ≥30% but does not disappear.

### 7.4 DFS — Disease-Free Survival

**What it is:** In patients who have had their tumor surgically removed
(resected), DFS measures the time until the cancer comes back or the
patient dies. Used in adjuvant (post-surgery) settings.

**Relevance:** Osimertinib's ADAURA trial showed dramatically improved
DFS in early-stage EGFR-mutant NSCLC patients who took osimertinib after
surgery. 96% vs 44% DFS at 24 months.

### 7.5 EFS — Event-Free Survival

Similar to DFS/PFS but used in specific contexts. "Event" is defined
based on the trial — could include disease progression, cancer recurrence,
new treatment needed, or death. Often used in neoadjuvant (pre-surgery)
settings.

### 7.6 DoR — Duration of Response

**What it is:** Among patients who responded to treatment (CR or PR),
how long did the response last before the cancer grew again?

**Why it matters:** A drug might produce responses in 40% of patients,
but if those responses only last 2 months, the benefit is limited. A
drug that produces responses in 20% of patients but those responses last
3 years may be more valuable. Checkpoint inhibitors are notable for
producing durable (long-lasting) responses in some patients, even complete
remissions that persist years after stopping treatment.

### 7.7 TTP — Time to Progression

**What it is:** Time from treatment start until tumor growth. Unlike PFS,
deaths without progression are not counted (they are "censored").

**Less commonly used** than PFS because it can overestimate benefit by
excluding patients who die before their tumor grows.

### 7.8 AE — Adverse Events

**What it is:** Side effects. Clinical trials systematically record and
grade all adverse events using the Common Terminology Criteria for Adverse
Events (CTCAE) scale:
- Grade 1: Mild, no intervention needed
- Grade 2: Moderate, some intervention needed
- Grade 3: Severe, hospitalization may be required
- Grade 4: Life-threatening, urgent intervention required
- Grade 5: Death from the adverse event

**For checkpoint inhibitors:** The most important AEs are immune-related:
pneumonitis, colitis, hepatitis, endocrinopathies (thyroid, adrenal),
nephritis. These require stopping the drug and immunosuppression.

**For osimertinib:** Diarrhea, rash (common), interstitial lung disease
(rare but serious), QTc prolongation (heart rhythm).

**For trastuzumab:** Cardiotoxicity (heart function impairment) — requires
regular cardiac monitoring. Infusion reactions.

**For metformin:** Gastrointestinal (nausea, diarrhea) — most common,
usually resolves after dose adjustment. Lactic acidosis (very rare, mainly
in kidney failure patients). Vitamin B12 deficiency with long-term use.

---

## 8. Study Types and Evidence Hierarchy

This maps directly to our `study_type` and `evidence_level` fields.

### Evidence pyramid — from strongest to weakest:

```
Level 1 (strongest)
  ├── Systematic review and meta-analysis
  │     Combines all RCTs on a topic into one analysis.
  │     Highest level of evidence when done rigorously.
  │
  ├── RCT (Randomized Controlled Trial)
  │     Patients randomly assigned to treatment or control.
  │     Randomization removes selection bias.
  │     The standard for proving a drug works.
  │
  ├── Phase 3 Trial
  │     Large RCT (hundreds to thousands of patients).
  │     Designed to prove efficacy and safety for FDA approval.
  │     Has a control arm (standard of care or placebo).
  │
  └── Phase 2 Trial
        Smaller RCT or single-arm study.
        Designed to estimate efficacy and identify the right dose.
        Often used for accelerated approval.

Level 2 (moderate)
  ├── Real-world study
  │     Analyzes data from actual clinical practice, not a trial.
  │     Includes patients who would be excluded from trials
  │     (elderly, multiple conditions). More generalizable but
  │     less controlled.
  │
  ├── Retrospective cohort
  │     Looks backward at patient records. Identifies a group
  │     that received a treatment and compares to those who did not.
  │     Subject to selection bias.
  │
  ├── Pooled analysis
  │     Combines individual patient data from multiple trials.
  │     More powerful than single trial but not as clean as meta-analysis.
  │
  └── Observational study
        Researchers observe without intervening. Good for safety signals
        and rare events. Cannot prove causation.

Level 3 (weakest)
  ├── Case series
  │     A group of patients (usually <10-20) treated similarly.
  │     No control group. Can suggest a signal but not confirm it.
  │
  └── Case report
        A single patient. Useful for rare adverse events or unexpected
        responses. Not generalizable.
```

### Why evidence level matters for our system

The grader node evaluates retrieved chunks. A chunk from an RCT with
n=1,966 patients (evidence_level=1) carries different clinical weight than
a chunk from a case report with n=1 patient (evidence_level=3).

When the Efficacy Agent retrieves evidence about pembrolizumab OS benefit,
it should prioritize Level 1 evidence (RCTs, meta-analyses). When the
Safety Agent looks for rare adverse events, Level 3 evidence (case reports)
is actually more appropriate because rare events won't appear in large trials
with predefined endpoints.

---

## 9. Clinical Trial Terminology

### 9.1 Trial phases

**Phase I:** First-in-human testing. Small number of patients (20-80).
Goal: find the right dose and identify major side effects. Not about
proving efficacy.

**Phase II:** Medium-sized study (50-300 patients). Goal: first efficacy
signal — does the drug work? Identifies the patient population likely
to benefit. Often single-arm (no control group).

**Phase III:** Large trial (300-3000+ patients). Head-to-head comparison
against standard of care or placebo. Goal: prove the drug is better.
This is what regulators require for approval.

**Phase IV:** Post-approval studies. Monitors safety in the real world.
Can expand the approved indication.

### 9.2 Trial naming conventions

**KEYNOTE trials:** Pembrolizumab trials by Merck
- KEYNOTE-024: First pembrolizumab monotherapy approval in NSCLC
- KEYNOTE-189: Pembrolizumab + chemo in non-squamous NSCLC
- Numbers are sequential but do not indicate importance

**CheckMate trials:** Nivolumab trials by Bristol Myers Squibb
- CheckMate-017: NSCLC squamous
- CheckMate-214: Renal cell carcinoma (nivolumab + ipilimumab)

**FLAURA, ADAURA, LAURA:** Osimertinib trials by AstraZeneca
- FLAURA: First-line EGFR-mutant NSCLC
- ADAURA: Adjuvant osimertinib after resection
- LAURA: Stage III after chemoradiation (2024)

**NCT numbers:** Every clinical trial registered on ClinicalTrials.gov
gets a unique identifier: NCT followed by 8 digits. Example: NCT02578680.
These are stored in our `clinical_trial_id` field for deduplication —
the same trial may be reported in multiple publications.

### 9.3 Randomization and control arms

**Randomized:** Patients are assigned to treatment or control by chance
(like a coin flip). This prevents the doctor choosing which patients get
the new drug (which would create bias — they might choose healthier patients
for the new drug, making it look better than it is).

**Control arm options:**
- Placebo: Sugar pill. Ethical only when no effective treatment exists.
- Standard of care (SoC): Current best treatment. Most common in oncology.
- Best supportive care: Just managing symptoms, no active cancer treatment.
- Active comparator: Another approved drug (head-to-head comparison).

### 9.4 Hazard ratio (HR)

The most important statistical measure in survival analysis.

```
HR = 0.63 means: at any given time point, patients in the treatment
     group have a 37% lower risk of experiencing the event (death,
     progression) compared to the control group.

HR < 1.0 = treatment is better (lower hazard/risk)
HR = 1.0 = no difference
HR > 1.0 = control is better (treatment has higher risk)
```

**95% Confidence Interval (CI):** If the CI includes 1.0 (e.g., 0.85 to
1.12), the result is not statistically significant — the difference could
be due to chance. If the CI does not include 1.0 (e.g., 0.52 to 0.76),
the result is statistically significant.

**p-value:** The probability that the observed result occurred by chance.
p < 0.05 is the conventional threshold for statistical significance. A
p-value tells you IF a difference exists; the hazard ratio tells you HOW
BIG the difference is.

### 9.5 Neoadjuvant vs Adjuvant

**Neoadjuvant:** Treatment given BEFORE surgery. Goal: shrink the tumor
to make surgery easier or possible, kill any cancer cells that may have
spread, test whether the tumor responds to the drug (response predicts
future outcomes).

**Adjuvant:** Treatment given AFTER surgery. Goal: kill any remaining
cancer cells that the surgeon couldn't see and remove. Aims to prevent
recurrence. Osimertinib (ADAURA trial) is a major adjuvant success story.

**Perioperative:** Treatment given both before AND after surgery.

---

## 10. Diabetes — Separate Disease, Same System

Metformin exists in our knowledge base because it is the world's most
prescribed diabetes drug. Understanding what it treats is important for
retrieval.

### 10.1 Type 2 Diabetes (T2DM)

**What it is:** A metabolic disease where the body either does not make
enough insulin or does not respond normally to insulin (insulin resistance).
Result: blood glucose stays elevated after meals.

**Insulin's job:** When you eat carbohydrates, blood glucose rises. The
pancreas releases insulin. Insulin acts like a key that opens doors on
muscle and fat cells, allowing glucose to enter and be used for energy.
Blood glucose returns to normal.

**In T2DM:** The "locks" (insulin receptors) on cells become less
responsive. The pancreas initially compensates by producing more insulin.
Eventually, the pancreas cannot keep up, and blood glucose remains
chronically elevated. This damages blood vessels, nerves, kidneys, eyes,
and the heart over decades.

**Difference from Type 1 Diabetes:**
Type 1 is an autoimmune disease where the immune system destroys the
insulin-producing cells in the pancreas completely. Type 1 patients
produce no insulin and must inject it to survive. Metformin is not
used in Type 1 as the primary treatment.

### 10.2 Key outcome measures in diabetes (different from cancer)

| Measure | What it is | Target |
|---|---|---|
| HbA1c | 3-month blood sugar average (glycosylated hemoglobin) | <7% typically |
| Fasting glucose | Blood sugar after 8 hours without eating | 70-100 mg/dL normal |
| Postprandial glucose | Blood sugar 2 hours after eating | <140 mg/dL normal |
| HOMA-IR | Insulin resistance score | Lower = more sensitive |
| eGFR | Kidney function score | Metformin contraindicated if very low |

**HbA1c explained:** Red blood cells live about 3 months. Glucose
gradually sticks to hemoglobin (the protein in red blood cells) in
proportion to average blood glucose over that period. HbA1c is measured
as a percentage. Normal is ~5%. Diabetes diagnosis at ≥6.5%. Treatment
target usually <7%. Metformin reduces HbA1c by 1-1.5 percentage points.

---

## 11. Why These 5 Drugs Together

The 5 drugs were chosen deliberately to cover multiple dimensions:

| Dimension | Coverage |
|---|---|
| Mechanism | Immunotherapy (pembro, nivo), targeted therapy (osimertinib, trastuzumab), metabolic (metformin) |
| Cancer type | NSCLC, breast, melanoma, gastric, RCC, colorectal, many more |
| Company | Merck (pembro), BMS (nivo), AstraZeneca (osimertinib), Roche (trastuzumab), generic (metformin) |
| Literature size | 24K+ (pembro), 15K+ (nivo), 8K+ (osimertinib), 20K+ (trastuzumab), 10K+ (metformin) |
| Therapeutic area | Oncology + metabolic disease |
| Evidence age | Trastuzumab (25 years), metformin (60 years), osimertinib (10 years) |

**For interviews:** Each drug maps to a major pharma company and a major
therapeutic area. Osimertinib → AstraZeneca (interview target).
Pembrolizumab → Merck. Nivolumab → Bristol Myers Squibb. Trastuzumab →
Roche. This cross-company, cross-mechanism coverage demonstrates the
system works for any pharma context, not just one company's portfolio.

---

## 12. Quick Reference Glossary

| Term | Full form | Simple meaning |
|---|---|---|
| NSCLC | Non-Small Cell Lung Cancer | Main type of lung cancer (85%) |
| SCLC | Small Cell Lung Cancer | Aggressive lung cancer (15%) |
| TNBC | Triple-Negative Breast Cancer | Breast cancer with no hormone/HER2 targets |
| PD-1 | Programmed Death-1 | Receptor on T-cells — cancer exploits it |
| PD-L1 | Programmed Death-Ligand 1 | Protein on cancer cells — sends "don't kill me" |
| TPS | Tumor Proportion Score | % of tumor cells with PD-L1 |
| CPS | Combined Positive Score | PD-L1 in tumor + immune cells (gastric, bladder) |
| EGFR | Epidermal Growth Factor Receptor | Growth signal receptor — mutated in some NSCLC |
| HER2 | Human Epidermal Growth Factor Receptor 2 | Overexpressed in some breast/gastric cancers |
| T2DM | Type 2 Diabetes Mellitus | Insulin resistance → high blood glucose |
| HbA1c | Glycosylated Hemoglobin | 3-month blood sugar average |
| AMPK | AMP-Activated Protein Kinase | Cell's energy sensor — metformin activates it |
| OS | Overall Survival | How long patients live |
| PFS | Progression-Free Survival | How long until cancer grows |
| ORR | Objective Response Rate | % of patients whose tumor shrinks ≥30% |
| DFS | Disease-Free Survival | Post-surgery time until cancer returns |
| CR | Complete Response | Tumor disappears completely |
| PR | Partial Response | Tumor shrinks ≥30% |
| AE | Adverse Event | Side effect |
| irAE | Immune-related Adverse Event | Side effects from checkpoint inhibitors |
| HR | Hazard Ratio | Risk comparison between treatment and control |
| CI | Confidence Interval | Range of uncertainty around an estimate |
| RCT | Randomized Controlled Trial | Most rigorous study design |
| MSI-H | Microsatellite Instability-High | Defective DNA repair — responds to pembro |
| dMMR | Deficient Mismatch Repair | Same as MSI-H — different way to measure |
| TMB | Tumor Mutation Burden | Number of mutations — high TMB may predict response |
| SoC | Standard of Care | Current best approved treatment |
| FDA | Food and Drug Administration | US drug regulator |
| NCT | National Clinical Trial number | Unique ID for each registered trial |
| GEJ | Gastroesophageal Junction | Where stomach meets esophagus |
| HNSCC | Head and Neck Squamous Cell Carcinoma | Cancer of mouth/throat/voice box |
| RCC | Renal Cell Carcinoma | Kidney cancer |
| HCC | Hepatocellular Carcinoma | Primary liver cancer |
| EGFR del19 | EGFR exon 19 deletion | Most common EGFR mutation — osimertinib works |
| L858R | EGFR L858R mutation | Second most common EGFR mutation |
| T790M | EGFR T790M resistance mutation | Emerges after first-gen EGFR inhibitors |
| KEYNOTE | Merck's pembrolizumab trial series | Named trials: KEYNOTE-024, 189, 407... |
| CheckMate | BMS's nivolumab trial series | Named trials: CheckMate-017, 214... |
| FLAURA | AZ's osimertinib first-line trial | First-line EGFR-mutant NSCLC |
| ADAURA | AZ's osimertinib adjuvant trial | Post-surgery EGFR-mutant NSCLC |
| CLEOPATRA | Roche's trastuzumab + pertuzumab trial | HER2+ metastatic breast cancer |
| Neoadjuvant | Treatment before surgery | Shrink tumor, test drug response |
| Adjuvant | Treatment after surgery | Kill remaining cells, prevent recurrence |

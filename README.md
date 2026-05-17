

# 🧠 LLM-HTI: Large language model-enabled herb-target interaction prediction for intelligent traditional medicine discovery

## 📌 Overview

This project proposes a **large language model (LLM)-driven computational framework** for herb–target interaction (HTI) prediction, aiming to bridge **Traditional Chinese Medicine (TCM)** and **modern biomedical knowledge**.

Unlike traditional methods that rely solely on graph structure or handcrafted features, our framework integrates:

* 🧠 **LLM-based semantic representation**
* 🌐 **Knowledge graph topology**
* ⚙️ **Adaptive optimization mechanism**

to model the **multi-component and multi-target mechanisms** of TCM.

---

## 🧩 Framework Architecture

The proposed framework consists of three major modules:

### 1. Knowledge Graph Construction & Semantic Extraction

* Build a **Traditional Chinese and Western Medicine Knowledge Graph (TWKG)**
* Use **large language models** to extract **latent semantic embeddings** from:

  * Herb–target interactions
  * Biomedical databases
  * TCM literature

---

### 2. Graph Representation Learning

* Integrate:

  * 📌 Semantic embeddings (from LLM)
  * 🔗 Graph topology (from TWKG)
* Learn joint representations via a **graph neural network (GNN)**
* Inspired by residual-style collaborative filtering architectures (e.g., LR-GCCF)

---

### 3. Target Prediction with Adaptive Optimization

* Introduce a **parameter-adaptive updating mechanism**
* Dynamically adjusts parameters based on:

  * gradient importance
* Benefits:

  * ✅ Improved training stability
  * ✅ Faster convergence
  * ✅ No change to model architecture

---

## 📄 Abstract

Large language models, owing to their strong knowledge integration capabilities, are reshaping the research paradigm of virtual screening and molecular design and are emerging as a key technological driver of modern drug discovery.

Inspired by these advances, this study proposes a large-model-driven computational framework for integrative Chinese and Western medicine knowledge graphs, aiming to accurately identify potential herb–target interactions, thereby elucidating the multi-component and multi-target synergistic mechanisms of traditional Chinese medicine and providing methodological support for its development.

This study leverages large language models to extract latent semantic information from herb–target interactions and integrates it with the topological structure of the Traditional Chinese and Western Medicine Knowledge Graph (TWKG), enabling a joint representation of traditional medicine characteristics and their links to modern biomedical knowledge.

On this basis, **we introduce a parameter-adaptive updating mechanism** that dynamically selects parameters according to gradient importance, thereby improving training stability and optimization efficiency without altering the model architecture.

On a large-scale dataset containing **38,002 high-quality HTI pairs**, our method achieves:

* **HR@10 = 0.4702**
* **NDCG@10 = 0.4026**

outperforming existing baselines.

Collectively, this work provides a new computational framework for herb–target mechanism analysis and intelligent drug discovery driven by the integration of traditional and modern medicine.

---

## 🗂️ Project Structure

```
.
├── code/                     # Training and evaluation scripts
│   ├── train.py
│   ├── test.py
│
├── Knowledge_embedding/     # Precomputed semantic embeddings
│   ├── herb_embedding.txt
│   ├── gene_embedding.txt
│
├── data/
│   ├── training_set.npy
│   ├── testing_set.npy
│   ├── val_set.npy
│   ├── set_all.npy
│   ├── name2id
│
├── fig1.jpg                 # Model architecture
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

We recommend using **Anaconda**:

```bash
conda create -n LLM-HTI python=3.9
conda activate LLM-HTI
pip install -r requirements.txt
```

---

## 📊 Data Description

Each dataset (`.npy`) contains:

* Herb–target interaction pairs
* Bidirectional mappings:

  * herbs → targets
  * targets → herbs
* Interaction statistics

Additional file:

* `name2id`: maps entity IDs to real herb/gene names

---

## 🚀 Usage

### 1. Prepare Embeddings

* Generate semantic embeddings using LLMs
* Store in:

```
/Knowledge_embedding/
```

---

### 2. Train Model

```bash
cd code
python train.py
```

---

### 3. Evaluate Model

```bash
cd code
python test.py
```

---

## 📌 Notes

* Ensure **embedding dimensions match model input dimensions**
* The framework is **not strictly end-to-end**:

  * LLM embeddings are precomputed
* Supports flexible replacement of:

  * LLM models
  * GNN architectures

---

## 📈 Results

| Metric  | Score  |
| ------- | ------ |
| HR@10   | 0.4702 |
| NDCG@10 | 0.4026 |

✅ Demonstrates strong ranking performance on HTI prediction
✅ Outperforms existing baseline methods

---

## 🔬 Key Contributions

* 🚀 First integration of **LLM semantic knowledge + TWKG topology**
* 🧠 Captures **cross-domain knowledge (TCM + modern biomedicine)**
* ⚙️ Introduces **adaptive parameter updating mechanism**
* 📊 Achieves **state-of-the-art HTI prediction performance**

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{XXX2026,
  title={LLM-Driven Herb–Target Interaction Prediction via Integrative Knowledge Graph Learning},
  author={XXX},
  journal={XXX},
  year={2026}
}
```

---

This project is inspired by:

* Knowledge graph embedding methods
* Graph neural networks for recommendation
* Recent advances in large language models for biomedical applications




---

## ⭐ Star This Repo

If you find this project helpful, please consider giving it a ⭐!

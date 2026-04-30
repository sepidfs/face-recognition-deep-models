
<img width="624" height="418" alt="image" src="https://github.com/user-attachments/assets/d0c4dcdd-edb1-4b41-b679-8dc4e8a14a68" />



# Face Recognition Deep Models

🚀 A multi-model deep learning framework for **face verification, identification, and embedding learning** using real-world unconstrained facial data.

---

## 📊 Dataset

This project uses the **Labeled Faces in the Wild (LFW)** dataset, a standard benchmark for face recognition under real-world conditions.

🔗 Dataset Source:
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

### Dataset Characteristics

* **13,233 face images**

* **5,749 unique individuals**

* Images collected from the web under **unconstrained conditions**
  (pose, lighting, expression, background variation)

* Faces are:

  * Automatically detected using **Viola–Jones**
  * Centered and aligned (deep-funneled version)

👉 This makes LFW a **challenging and realistic dataset** for evaluating face recognition systems ([Kaggle][1])

---

## 🧩 Data Structure

The dataset includes both **image data and metadata files**:

* `people.csv`
  → Contains each person's name and number of images

* `pairs.csv`
  → Contains image pairs for verification tasks (same vs different)

### Two Learning Setups

#### 🔹 Verification (Pair-based)

* Input: (image₁, image₂)
* Output: same / different
* Used for:

  * Siamese Network
  * Triplet Network

#### 🔹 Identification (Classification)

* Input: single image
* Output: identity label
* Used for:

  * MobileNetV2 classifier

---

## ⚙️ Data Processing

* Image resizing: **128×128 / 160×160**
* Normalization: pixel scaling to [0,1]
* Data augmentation:

  * Horizontal flip
  * Rotation
  * Zoom
  * Brightness variation

---

## 🧠 Models Implemented

This project explores **four core deep learning paradigms**:

---

### 🔹 1. Autoencoder (SSIM + MSE)

* Unsupervised representation learning
* Learns **128-dimensional latent embeddings**
* Loss:

$$
\mathcal{L} = \text{MSE} + (1 - \text{SSIM})
$$

✅ Strength:

* Best overall performance (~0.88)
* High perceptual reconstruction quality

💡 Insight:
Structural similarity (SSIM) captures identity features better than pixel-wise loss.

---

### 🔹 2. Siamese Network

* Pair-based similarity learning
* Shared encoder architecture
* Distance-based decision (L1 / L2)

✅ Result: ~70% validation accuracy

💡 Insight:
Fine-tuning pretrained backbones (VGG16) significantly improves performance.

---

### 🔹 3. Triplet Network

* Uses (Anchor, Positive, Negative)
* Learns discriminative embedding space

Loss:

$$
\mathcal{L} = \max(0, d(A,P) - d(A,N) + \alpha)
$$

✅ Result: ~78% accuracy

💡 Insight:
Triplet loss improves **global separation of identities**.

---

### 🔹 4. MobileNetV2 (Transfer Learning)

* Pretrained on ImageNet
* Lightweight (~2.2M parameters)
* Efficient embedding extraction

✅ Result: ~80% validation accuracy

💡 Insight:
Best trade-off between **speed and accuracy**.

---

## 🏆 Key Findings

* **Best Overall Model:** Autoencoder (SSIM-based)
* **Best Lightweight Model:** MobileNetV2
* **Best Embedding Separation:** Triplet Network
* **Best Verification Setup:** Fine-tuned Siamese

---

## 💡 Core Takeaways

* Embedding learning outperforms classification in low-data settings
* Transfer learning is essential for real-world datasets
* Metric learning improves identity discrimination
* Combining multiple models leads to deeper understanding

---

## 🛠️ Tech Stack

Python • TensorFlow • PyTorch • OpenCV • NumPy • Matplotlib

---

## 🚀 Demo

```bash
python Demo.py
```

---

## 👤 Author

Sepideh Forouzi


---

[1]: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?utm_source=chatgpt.com "Labelled Faces in the Wild (LFW) Dataset"

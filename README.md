# Null_class_internship



This project integrates a complete deep learning pipeline combining visual data processing, text embedding using transformers, and image generation using a GAN enhanced with self-attention. It is ideal for projects that involve AI-powered multimedia workflows, especially in text-to-image synthesis.

---

## 📦 Installation

Install all required dependencies in your Python environment:

```bash
pip install torch torchvision scikit-learn matplotlib opencv-python transformers datasets streamlit
pip install -r requirements.txt
```

---

# 📁 Project Structure: Task-Based Breakdown

This project is divided into three main tasks:

---

## 🧩 Task 1: Image Loading and Display

**Location:** `task_1/`

### Description:

This task focuses on loading and displaying images using OpenCV and Matplotlib. It helps you understand image preprocessing and visualization.

### Script:

* `display_iamge.ipynb`: Loads an image, converts from BGR to RGB, and displays it.

### How to Run:

```bash
run on colab
```

Ensure `image.jpg` is present in the `task_1/` directory.

---

## 🧩 Task 2: Text Embedding with Transformers

**Location:** `task_2/`

### Description:

This task involves generating embeddings from textual input using transformer-based models via a Streamlit application.

### Script:

* `app.py`: Launches a Streamlit interface for inputting text and viewing token embeddings.

### How to Run:

```bash
streamlit run app.py
```

### Features:

* Choose models like BERT, DistilBERT, or CLIP.
* View token IDs and actual tokens.
* Save token-level embeddings to `saved_embeddings/embedding.pt`.

---

## 🧩 Task 3: Self-Attention GAN for Image Generation

**Location:** `task_3/`

### Description:

Implements a GAN with self-attention to generate flower images using the Oxford Flowers102 dataset. Includes evaluation metrics.

### Script:

* `gan_self_attention.ipynb `: Trains the GAN and saves generated images + models.

### How to Run:

```bash
run on colab
```

### Output:

* Generated images saved to `generated_images/`
* Models saved as `.pth` files
* Evaluation metrics include confusion matrix, precision, recall, and accuracy.

---

## ✅ Summary

Each task is modular and targets a different phase of the AI pipeline:

* Task 1: Image Handling
* Task 2: Text Understanding
* Task 3: Image Synthesis

Together, they build a strong foundation for text-to-image generation workflows.


## ✅ Summary

This project demonstrates a full-stack ML pipeline for text and image processing:

* Image handling via OpenCV + Matplotlib
* Text embedding using transformer models
* Image generation using a GAN with self-attention
* Evaluation using real classification metrics

---

## 📬 Feedback and Contributions

Feel free to fork this repo, open issues, or make pull requests to enhance the functionality. Contributions and improvements are welcome!

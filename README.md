# Transfer & Zero-Shot Learning - Multi-Modal Deep Learning Assignment

This repository explores deep transfer learning and zero-shot generalization across multiple data modalities: Images, Audio, Video, and Text. The goal is to evaluate and compare pretrained models as both feature extractors and fine-tuned classifiers using modern architectures like EfficientNet, BERT, BiT, and CLIP.

---

## Key Concepts

### Transfer Learning
Transfer learning leverages knowledge from a model pretrained on one task and adapts it to a different but related task. It is especially useful when labeled data is limited.

### Zero-Shot Learning
Zero-shot learning enables a model to correctly classify or interpret data from classes it has never seen before, usually by leveraging textual descriptions or semantic embeddings.

---

## Part 1: Supervised Contrastive Learning vs. Softmax Classifier

This part investigates the difference in representational power between:
- A softmax classifier using categorical crossentropy loss.
- A supervised contrastive learning framework that learns to cluster semantically similar data in feature space.

Tasks covered:
- CIFAR-10 classification
- Visualization of embeddings using t-SNE
- Confusion matrix comparisons

Colab Notebook: [Open in Colab](https://colab.research.google.com/drive/1fkOuFchsXCFdBeHDIAP3aptpjy7bJd8x?usp=sharing)

Reference:
- https://keras.io/examples/vision/supervised-contrastive-learning/

---

## Part 2: Transfer Learning on Various Modalities

This part demonstrates the power of pretrained models across four different data types:

### Image:
- Dataset: Cats vs Dogs
- Model: EfficientNetB0
- Approach: Feature extraction and fine-tuning for binary classification

### Audio:
- Model: YAMNet (TFHub)
- Use Case: Extract audio embeddings from .wav files and classify them using logistic regression

### Video:
- Model: I3D (Inflated 3D ConvNet)
- Task: Action recognition via video embedding extraction and classification

### Text:
- Dataset: IMDb movie reviews
- Model: BERT (from TFHub)
- Use Case: Text classification using sentence-level pooled embeddings

Colab Notebook: [Open in Colab](https://colab.research.google.com/drive/14MZ1YPcrmYFvVL3PfjH_vEeoKFL3TMu0?usp=sharing)

References:
- https://www.tensorflow.org/tutorials/images/transfer_learning
- https://blog.tensorflow.org/2021/03/transfer-learning-for-audio-data-with-yamnet.html
- https://www.tensorflow.org/hub/tutorials/tf2_text_classification

---

## Part 3: Zero-Shot Transfer Learning

This part demonstrates zero-shot learning by leveraging text-image alignment and large pretrained visual models.

### CLIP (OpenAI)
- Model: ViT-B/32
- Task: Image classification based on natural language prompts with no additional training

### BigTransfer (BiT)
- Dataset: tf_flowers
- Model: BiT m-r50x1 from TensorFlow Hub
- Task: Fine-tune BiT as a feature extractor with a classification head

Colab Notebook: [Open in Colab](https://colab.research.google.com/drive/1QM_RC5VE_rsHwtLCfKqnddrE9-l3YDx2?usp=sharing)

References:
- https://towardsdatascience.com/how-to-try-clip-openais-zero-shot-image-classifier-439d75a34d6b
- https://keras.io/examples/vision/bit/
- https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub

---

## Part 4: Vision Classifiers with EfficientNet and BiT

This part demonstrates transfer learning on three core vision datasets — MNIST, Fashion MNIST, and CIFAR-10 — using pretrained models: **EfficientNetB0** and **BiT** (simulated with ResNet50V2 from Keras Applications). It showcases both **feature extraction** and **fine-tuning**, while enabling fast experimentation by subsampling the dataset and resizing inputs.

### Datasets:
- **MNIST** (grayscale handwritten digits)
- **Fashion MNIST** (grayscale clothing images)
- **CIFAR-10** (RGB natural images)

### Key Features:
- Uses smaller dataset subsets (e.g., 3000 training samples) for speed
- Converts grayscale images to RGB for compatibility with ImageNet-pretrained models
- Resizes images to 96×96
- Applies data augmentation and normalization
- Implements:
  - Feature extraction with **EfficientNetB0**
  - Fine-tuning top layers of EfficientNet
  - Transfer learning using **BiT-like ResNet50V2**
- Plots training/validation curves and visualizes predictions

### Models and Techniques:
- **EfficientNetB0**: Pretrained feature extractor with optional fine-tuning
- **BiT (simulated)**: ResNet50V2 used in place of original BiT
- Training and evaluation performed using Keras Functional API
- Results include accuracy/loss plots and prediction visualizations

**Colab Notebook**: [Open in Colab](https://colab.research.google.com/drive/1SBO5IkvZfpQWP6uZsugeIUTTes1Xs3zJ?usp=sharing)

### References:
- https://keras.io/api/applications/efficientnet/
- https://keras.io/api/applications/resnet/
- https://www.tensorflow.org/tutorials/images/transfer_learning

---

## Folder Structure
```
transfer-learning-assignment/
├── part1_supervised_contrastive.ipynb
├── part2_multimodal_transfer.ipynb
├── part3_zero_shot_clip_bit.ipynb
├── part4_vision_models.ipynb
└── README.md
```

---

## Requirements
- Python 3.10+
- TensorFlow 2.11+
- PyTorch 2.0+
- tensorflow-hub, tensorflow-datasets, scikit-learn, matplotlib, ftfy, regex, tqdm

Install dependencies:
```bash
pip install tensorflow tensorflow-hub tensorflow-datasets torch torchvision sklearn matplotlib ftfy regex tqdm
```

---

## References
- OpenAI CLIP: https://github.com/openai/CLIP
- TensorFlow Hub: https://tfhub.dev
- TensorFlow Datasets: https://www.tensorflow.org/datasets
- Keras Examples: https://keras.io/examples/
- LangGraph Agent Patterns: https://www.youtube.com/watch?v=aHCDrAbH_go


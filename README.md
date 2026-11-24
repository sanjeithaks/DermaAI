<div align="center">

# 🩺 Skin Lesion Classifier

**A deep learning web app that classifies dermatoscopic skin images into 7 lesion types, built on the HAM10000 dataset.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![Status](https://img.shields.io/badge/status-active-brightgreen)

[Overview](#overview) •
[Demo](#demo) •
[Tech Stack](#tech-stack) •
[Getting Started](#getting-started) •
[How It Works](#how-it-works) •
[Results](#results) •
[Disclaimer](#️-disclaimer)

</div>

---

## Overview

This project trains a **MobileNetV2** transfer-learning model on the **HAM10000** dermatoscopic image dataset (10,000+ labeled images across 7 diagnostic categories) and serves live predictions through a **Streamlit** web app — upload a lesion image, get an instant classification with confidence scores.

It was built end-to-end — data preprocessing, class-imbalance handling, model training, and a live interface — as a first deep learning project applying transfer learning to a real, publicly available medical imaging dataset.


## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| ML Framework | TensorFlow / Keras |
| Base Model | MobileNetV2 (ImageNet-pretrained, transfer learning) |
| Dataset | [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (Kaggle) |
| Web Interface | Streamlit |
| Deployment | Docker (optional) |

## Project Structure

```
skin_cancer_prediction_model/
├── data_preprocessing.py    # Loads metadata, splits data, handles class imbalance, augmentation
├── train_model.py           # Two-phase MobileNetV2 transfer learning + evaluation
├── app.py                   # Streamlit web app for live inference
├── requirements.txt         # Dependencies to run the app
├── requirements-train.txt   # Extra dependencies to train the model
└── Dockerfile                # Container build for hosting the app
```

> `skin_cancer_model.h5` (the trained model) isn't committed to this repo — it's a large binary file you generate yourself by running `train_model.py` (see below).

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/imdipul/skin_cancer_prediction_model.git
cd skin_cancer_prediction_model
```

### 2. Get the dataset
Download the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) from Kaggle and unzip it into a `data/` folder in the project root, so you end up with:
```
data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

### 3. Set up your environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-train.txt
```
> **Apple Silicon (M1/M2/M3/M4) users:** `requirements.txt` lists `tensorflow-cpu`, which has no macOS build. Swap that line for plain `tensorflow` before installing.

### 4. Train the model
```bash
python data_preprocessing.py   # optional sanity check
python train_model.py
```
This produces `skin_cancer_model.h5`, plus `training_history.png`, `confusion_matrix.png`, and `classification_report.txt`.

### 5. Run the app
```bash
streamlit run app.py
```
Open the URL Streamlit prints (usually `http://localhost:8501`), upload a lesion image, and see the prediction live.

### Raspberry Pi image receiver

Start the FastAPI image receiver on the Windows computer:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn camera_receiver:app --host 0.0.0.0 --port 8000
```

The receiver accepts JPEG and PNG multipart uploads at `POST /upload-image`,
using the form field name `file`. It saves each validated upload in
`captured_images/`, assigns an image ID and UTC timestamp, and limits uploads
to 10 MiB by default. The response includes `success`, `image_id`, `timestamp`,
`filename`, and `upload_status`. `GET /health` reports whether the API is
running. Set `DERMAAI_MAX_UPLOAD_BYTES` to change the limit or
`DERMAAI_CAPTURE_DIR` to change the storage directory. The receiver can be
reached from a Raspberry Pi on the same network at
`http://<windows-computer-ip>:8000/upload-image`; allow the chosen port through
Windows Firewall on the private network. Keep the receiver on a trusted LAN;
this endpoint does not provide authentication or HTTPS.

Run the receiver tests with:

```powershell
python -m unittest discover -s tests -v
```

## How It Works

**1. Preprocessing & class imbalance** — HAM10000 is heavily skewed (melanocytic nevi make up ~67% of images). `data_preprocessing.py` computes inverse-frequency class weights, applies augmentation (flips, rotations, brightness/contrast jitter), and splits the data by `lesion_id` (grouped + stratified) so the same lesion never leaks across train/val/test.

**2. Model** — MobileNetV2, pretrained on ImageNet, fine-tuned in two phases: first with the base frozen (training only the classification head), then with the top ~40 layers unfrozen at a much lower learning rate.

**3. Inference** — `app.py` loads the saved model, preprocesses the uploaded image identically to training, and returns a softmax probability over all 7 classes.

### The 7 classes

| Code | Full Name | Typical Concern |
|---|---|---|
| `akiec` | Actinic keratoses / intraepithelial carcinoma | High |
| `bcc` | Basal cell carcinoma | High |
| `bkl` | Benign keratosis-like lesions | Low |
| `df` | Dermatofibroma | Low |
| `mel` | Melanoma | High |
| `nv` | Melanocytic nevi (moles) | Low |
| `vasc` | Vascular lesions | Low |

## Results

*Fill in with your own numbers from `classification_report.txt` after training:*

| Metric | Score |
|---|---|
| Overall accuracy | — |
| Macro-averaged recall | — |
| Macro-averaged F1 | — |

> Because of the class imbalance, macro-averaged recall matters more than raw accuracy here — a model that always predicts `nv` would score ~67% accuracy while being medically useless.

## Roadmap

- [ ] Add Grad-CAM visualizations to show which region of the image drove each prediction
- [ ] Try ResNet50 and compare accuracy vs. model size
- [ ] Host the app publicly (Hugging Face Spaces / Streamlit Community Cloud)
- [ ] Experiment with oversampling on top of class weighting for the rarest classes

## ⚠️ Disclaimer

This project is an educational deep learning exercise, evaluated on a public research dataset. **It is not a certified medical device and must never be used for self-diagnosis or real medical decisions.** Always consult a qualified dermatologist about any skin concern.

## Acknowledgments

- Dataset: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci Data 5, 180161 (2018), hosted on [Kaggle by kmader](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
- Base architecture: [MobileNetV2](https://arxiv.org/abs/1801.04381) (Sandler et al., 2018)

## License

 [MIT](https://choosealicense.com/licenses/mit/)

## Author

**Dipul Poudel** ([@imdipul](https://github.com/imdipul))
Website: [dipulpoudel.com.np](https://dipulpoudel.com.np) • LinkedIn: [linkedin.com/in/imdipul](https://linkedin.com/in/imdipul)

#  Plant Disease Detection

An AI-powered plant leaf disease classification project using **TensorFlow + MobileNetV2 (Transfer Learning)**.  
This repository includes:
- a **training pipeline** (`train.py`),
- a **single-image prediction script** (`predict.py`), and
- a **Streamlit web app** (`app.py`) for interactive inference, confidence visualization, prediction history, and PDF report download.

---

##  Dataset

This project is designed to work with the **PlantVillage** dataset hosted on Hugging Face:

- Dataset: **mohanty/PlantVillage**
- Link: https://huggingface.co/datasets/mohanty/PlantVillage

> Place the dataset in a local folder named `dataset/` so that each disease class is a subfolder (directory-per-class structure).

Expected structure:

```text
dataset/
  Apple___Apple_scab/
    image1.jpg
    image2.jpg
    ...
  Apple___Black_rot/
    ...
  Corn___Common_rust/
    ...
  ...
```

---

##  Features

- Transfer learning with **MobileNetV2**
- Automatic class-name export to `class_names.json`
- Streamlit UI for image upload and prediction
- Confidence score with top-5 class probability chart
- In-session prediction history table
- PDF report generation from the web app

---

## 🛠️ Tech Stack

- Python 3.9+
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- Pandas
- ReportLab

---

##  Installation

1. Clone the repository:

```bash
# HTTPS
git clone https://github.com/jwalasai7077/Plant-Disease-Detection.git

# or SSH
git clone git@github.com:jwalasai7077/Plant-Disease-Detection.git

cd Plant-Disease-Detection
```
 Repository URL: https://github.com/jwalasai7077/Plant-Disease-Detection

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
```

3. Install dependencies:

```bash
pip install tensorflow streamlit numpy pillow pandas reportlab
```

---

##  Model Training

Run training with:

```bash
python train.py
```

What this does:
- Loads images from `dataset/`
- Uses an 80/20 train-validation split
- Builds a transfer learning classifier on top of MobileNetV2
- Trains for 40 epochs
- Saves:
  - `plant_disease_model.h5`
  - `class_names.json`

---

##  Local Prediction (CLI)

Use `predict.py` for a quick single-image prediction.

```bash
python predict.py
```

By default, it predicts on:
- `test.jpg`

If needed, edit `img_path` in `predict.py` to test another image.

---

##  Run the Web App

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (typically `http://localhost:8501`).

### App workflow
1. Upload a leaf image (`.jpg`, `.jpeg`, `.png`)
2. View predicted disease and confidence score
3. Inspect top-5 probability chart
4. Track prediction history in the session
5. Download a PDF report

---

##  Project Structure

```text
Plant-Disease-Detection/
├── app.py                  # Streamlit web app
├── train.py                # Model training script
├── predict.py              # CLI prediction script
├── README.md
├── test.jpg                # Sample test image
├── plant_disease_model.h5  # Generated after training
└── class_names.json        # Generated after training
```

---

##  Notes

- Ensure the dataset folder is correctly named `dataset` and is in the project root.
- `predict.py` currently reads class names from `dataset/` folder names; for strict consistency with training output, you may prefer to load `class_names.json` there as well.
- First-time training may download pretrained MobileNetV2 weights from the internet.

---

##  Acknowledgment

- PlantVillage dataset on Hugging Face: https://huggingface.co/datasets/mohanty/PlantVillage
- TensorFlow/Keras transfer learning ecosystem


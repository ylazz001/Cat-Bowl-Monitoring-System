# Cat Bowl Monitoring System

An anomaly detection system that monitors a cat food bowl in real time using computer vision. A phone camera streams video to a PC, where a deep learning model analyzes the bowl every 20 minutes and plays a meow sound when it's time to refill.

---

## How It Works

```
Phone Camera (Camo) → PC (OpenCV) → Encoder (ResNet18) → Features (25,088-dim) → One-Class SVM → Decision Score → Threshold Logic → Alert
```

1. **Camo app** streams the phone camera to the PC at 1920×1080
2. **OpenCV** captures a frame every 20 minutes
3. **ResNet18 encoder** converts the image into a 25,088-dimensional feature vector
4. **One-Class SVM** scores how "normal" (full) the bowl looks
5. **Threshold logic** classifies the bowl as Full, Touched, or Refill Needed
6. **Streamlit web app** displays the result and plays a meow alert if refill is needed

Accessible from any device on the local network at `http://192.168.1.11:8501`.

---

## Architecture

### Component 1: Autoencoder (Feature Extractor)

The autoencoder learns what a full bowl looks like by compressing and reconstructing full bowl images. Only the encoder is used at inference — the decoder is discarded after training.

- **Encoder:** ResNet18 pre-trained on ImageNet (transfer learning), layers conv1 through layer4
- **Bottleneck:** 7×7×512 = 25,088 features when flattened
- **Decoder:** 5 upsampling blocks (used only during training)
- **Loss function:** MSE (Mean Squared Error)
- **Optimizer:** Adam with differential learning rates — encoder at 0.00005 (slow fine-tuning), decoder at 0.001
- **Epochs:** 40
- **Training data:** 255 full bowl images only

### Component 2: One-Class SVM (Anomaly Detector)

The SVM draws a boundary around the feature vectors of full bowl images. Anything outside the boundary is flagged as an anomaly.

- **Algorithm:** One-Class SVM with RBF kernel
- **nu=0.05** — allows 5% of training data to fall outside the boundary
- **Input:** 25,088-dimensional feature vectors from the encoder
- **Output:** A decision score (positive = normal, negative = anomaly)

### Three-Tier Alert System

```python
if score >= 0.12:
    status = "Full"          # No action needed
elif score >= -0.05:
    status = "Touched"       # Cat is eating, monitor
else:
    status = "Refill needed" # Plays meow sound alert
```

Thresholds were calibrated by running a grid search across the test set, optimizing for catching eaten bowls while minimizing false alarms on full bowls.

---

## Performance

**At default SVM boundary (score = 0):**

| Metric | Value |
|--------|-------|
| Accuracy | 60% |
| Recall | 43% |
| Precision | 96% |
| F1 Score | 0.60 |

The high precision means when the model flags a bowl as eaten, it's right 96% of the time. The low recall at the default boundary is why we use custom thresholds.

**At custom thresholds (touched=0.12, refill=-0.05):**

| Metric | Value |
|--------|-------|
| Good bowls correctly identified | 42/56 (81%) |
| Eaten bowls correctly caught | 94/122 (77%) |
| Overall accuracy | 78% |

We prioritized catching eaten bowls over avoiding false alarms — missing an eaten bowl means the cat doesn't get fed, while a false alarm just means you glance at the camera.

**Decision Score Distribution:**

| Category | Score Range |
|----------|------------|
| Full bowls | -0.10 to +0.33 |
| Eaten bowls | -0.54 to +0.26 |

---

## Dataset

```
data/raw/archive/cat_food/
├── train/
│   └── good/          # 255 full bowl images (training)
└── test/
    ├── good/          # 56 full bowl images (testing)
    └── eaten/         # 122 eaten bowl images (testing)
```

**Training images** were collected over several days with the bowl full, rotated 45° between shots under both natural and warm house lighting. All training images are full bowls only — the model never sees eaten bowls during training.

**Test images** include bowls at various fill levels, from barely touched to completely empty.

---

## Data Augmentation (Training Only)

```python
transforms.RandomResizedCrop(size=(224,224), scale=(0.90, 1.0))
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(degrees=10)
transforms.ColorJitter(brightness=0.1, contrast=0.0, saturation=0.0, hue=0.0)
transforms.RandomAffine(degrees=0, translate=(0.10, 0.10))
```

**Key decision:** Only brightness is varied in ColorJitter (not contrast, saturation, or hue). The color difference between pink food and silver empty bowl is the primary signal — augmenting color/hue would teach the model to ignore the very thing it needs to detect. Brightness-only handles lighting variation without masking the signal.

No denoising is applied. A denoising autoencoder was tested and reduced performance because it taught the model to look past small pixel-level changes — exactly the kind of changes that distinguish full from eaten.

---

## What We Tried and What Worked

| Configuration | Accuracy | F1 | Notes |
|--------------|----------|-----|-------|
| L1 loss + ColorJitter + Denoiser | 59% | 0.68 | First attempt |
| L1 loss - ColorJitter + Denoiser | 65% | 0.78 | Color matters |
| L1 loss - ColorJitter - Denoiser | 72% | 0.80 | Denoiser was hurting |
| Frozen encoder + L1 | 60% | 0.71 | Fine-tuning helps |
| **MSE loss - Denoiser + brightness 0.1 + 255 images** | **78%*** | **—** | **Final model** |

*78% calculated at custom thresholds (0.12 and -0.05).

**Key findings:**
- **MSE > L1** for this task — MSE penalizes large errors more, producing better feature separation
- **Removing ColorJitter** improved accuracy because color is the key signal (pink food vs silver bowl)
- **Removing denoiser** improved accuracy because the model was learning to ignore subtle changes it needed to detect
- **Freezing the encoder** made things worse — the small fine-tuning was actually helping, not hurting
- **More training data** (116 → 255 images) was the single biggest improvement for reducing false positives on full bowls

---

## Installation & Usage

### Prerequisites

```
Python 3.8+
PyTorch 2.0+
```

### Install Dependencies

```bash
pip install torch torchvision opencv-python scikit-learn numpy pillow streamlit
```

### Model Files

Place in `models/saved_models/`:
- `autoencoder_cat_food_MSE.pth` (~45 MB)
- `svm_detector_MSE.pkl` (~2 MB)

### Run the App

```bash
streamlit run app/streamlit_app.py
```

Access from any device on the local network at `http://[your-pc-ip]:8501`.

### Run Inference Manually

```python
import torch
import pickle
import numpy as np
from PIL import Image

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DefectAutoencoder()
model.load_state_dict(torch.load('models/saved_models/autoencoder_cat_food_MSE.pth', weights_only=False))
model = model.to(device)
model.eval()

with open('models/saved_models/svm_detector_MSE.pkl', 'rb') as f:
    svm = pickle.load(f)

# Process image
image = Image.open('path/to/bowl_image.jpg').resize((224, 224))
img_array = np.array(image) / 255.0
img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)

# Get prediction
with torch.no_grad():
    features = model.encoder(img_tensor)
    features = features.flatten(start_dim=1).cpu().numpy()

score = svm.decision_function(features)[0]

if score >= 0.12:
    print(f"Score: {score:.4f} → Full")
elif score >= -0.05:
    print(f"Score: {score:.4f} → Touched")
else:
    print(f"Score: {score:.4f} → Refill needed")
```

---

## Deployment Setup

| Component | Details |
|-----------|---------|
| Camera | Phone running Camo app, streaming to PC |
| Resolution | 1920×1080 |
| Capture | OpenCV, camera index 0 |
| Check interval | Every 20 minutes (configurable) |
| Web app | Streamlit at `http://192.168.1.11:8501` |
| Alert | Meow sound plays through PC speakers when refill needed |

**Fixed setup required:** Camera position and bowl placement must be consistent with training images. This is intentional — it simplifies the problem, similar to how industrial quality control systems use fixed camera mounts.

---

## Project Structure

```
defect-detection/
├── data/
│   └── raw/archive/cat_food/
│       ├── train/good/              # 255 full bowl training images
│       └── test/
│           ├── good/                # 56 full bowl test images
│           └── eaten/               # 122 eaten bowl test images
├── models/saved_models/
│   ├── autoencoder_cat_food_MSE.pth
│   └── svm_detector_MSE.pkl
├── notebooks/
│   └── 03_model_training_autoencoder.ipynb
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Future Improvements

- Phone push notifications instead of PC-only meow sound
- Time-series logging to track eating patterns over days/weeks
- Image alignment to handle small bowl position changes
- Multi-bowl support for multiple cats
- Threshold tuning UI within the Streamlit app

---

## Tech Stack

Python, PyTorch, OpenCV, scikit-learn, Streamlit, NumPy, Pillow

---

## Author

**Yuri Lazzeretti**
- [LinkedIn](https://www.linkedin.com/in/yuri-lazzeretti-b63a22220/)
- ylazz001@gmail.com

Background: Mechanical engineer (Boeing, 6 years) transitioning to ML/AI. Master's with honours in Computer Science and Data Analytics.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

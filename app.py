import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pickle
#import requests
from io import BytesIO
import time
import cv2
import streamlit.components.v1 as components
import base64
import pygame

# ============================================================
# MODEL DEFINITION
# We must copy this class here because PyTorch needs to know
# the exact architecture before it can load the saved weights.
# ============================================================
class DefectAutoencoder(nn.Module):
    def __init__(self):
        super(DefectAutoencoder, self).__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')

        # ENCODER: uses pretrained ResNet18 layers to compress
        # the image down to a small but information-rich representation
        # Input:  [1, 3, 224, 224] (1 image, 3 channels, 224x224 pixels)
        # Output: [1, 512, 7, 7]   (512 feature maps, 7x7 pixels each)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # DECODER: takes the compressed representation and tries
        # to reconstruct the original image back to 224x224
        # We trained it only on normal bowls, so it reconstructs
        # abnormal bowls poorly - that's how we detect anomalies
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ============================================================
# LOAD MODELS
# @st.cache_resource means this function only runs ONCE when
# the app first starts, then the models stay in memory.
# Without this, the 60MB model would reload on every check,
# making the app very slow.
# ============================================================
@st.cache_resource
def load_models():
    # Load the autoencoder weights into the architecture we defined above
    autoencoder = DefectAutoencoder()
    autoencoder.load_state_dict(torch.load(
        r'C:\Users\ylazz\Desktop\defect-detection\models\saved_models\autoencoder_cat_food_MSE_0.pth',
        weights_only=False
    ))
    # eval() switches off training mode - important for inference
    # In training mode, some layers behave differently (e.g. dropout, batchnorm)
    autoencoder.eval()

    # Load the SVM - this was saved with pickle, a standard
    # Python tool for saving and loading any object
    with open(r'C:\Users\ylazz\Desktop\defect-detection\models\saved_models\svm_detector_MSE_0.pkl', 'rb') as f:
        svm = pickle.load(f)

    return autoencoder, svm


# ============================================================
# GRAB IMAGE FROM PHONE CAMERA
# IP Webcam app on phone creates a mini web server.
# Visiting /shot.jpg returns the current camera frame as an image.
# requests.get() fetches it just like a browser would.
# BytesIO converts the raw downloaded bytes into a file-like
# object that PIL can open as an image.
# timeout=5 means if the phone doesn't respond in 5 seconds,
# give up and raise an error (caught by try/except in main loop)
# ============================================================
def get_image_from_phone():
    # Open camera 0 - this is Camo (your phone camera)
    camera = cv2.VideoCapture(0)
    
    # Grab a frame
    ret, frame = camera.read()
    
    # Always release the camera after grabbing
    # If we don't do this, the camera stays locked
    # and the next check won't be able to open it
    camera.release()
    
    if not ret:
        raise Exception("Could not capture frame from camera")
    
    # OpenCV reads images in BGR color order (Blue Green Red)
    # but PIL and our model expect RGB (Red Green Blue)
    # This line swaps the color channels to fix that
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert from numpy array to PIL Image
    # which is what our analyze_image function expects
    image = Image.fromarray(frame_rgb)
    
    return image


# ============================================================
# ANALYZE IMAGE
# This replicates exactly what your notebook did:
# preprocess → encode → flatten → SVM score → apply thresholds
# ============================================================
def analyze_image(image, autoencoder, svm):
    # Step 1: Preprocess - same as your notebook's DefectDataset class
    img = image.convert('RGB')        # ensure 3 channels
    img = img.resize((224, 224))      # resize to what model expects
    img_array = np.array(img) / 255.0 # normalize pixels to 0-1 range
    img_array = img_array.transpose(2, 0, 1)  # (H,W,C) → (C,H,W) for PyTorch
    img_array = img_array.astype(np.float32)  # PyTorch needs float32

    # Step 2: Add batch dimension
    # Your notebook processed 32 images at a time: [32, 3, 224, 224]
    # Here we have 1 image, but PyTorch still expects a batch dimension
    # unsqueeze(0) turns [3, 224, 224] into [1, 3, 224, 224]
    img_tensor = torch.tensor(img_array).unsqueeze(0)

    # Step 3: Extract features using only the encoder
    # We don't need the decoder here - we just want the compressed
    # representation to feed into the SVM
    # Output shape: [1, 512, 7, 7] → flattened to [1, 25088]
    with torch.no_grad():  # no_grad saves memory - we're not training
        features = autoencoder.encoder(img_tensor)
        features = features.flatten(start_dim=1)

    # Step 4: Convert to numpy - SVM expects numpy arrays not tensors
    features_numpy = features.cpu().numpy()

    # Step 5: Get SVM decision score
    # decision_function returns a score, not just -1/+1
    # More negative = more abnormal = more different from a full bowl
    score = svm.decision_function(features_numpy)[0]

    # Step 6: Apply your two thresholds
    # These were determined from your notebook experiments
    if score >= 0.12:
        status = "Full"
        alert = "success"
        message = "✅ Bowl looks full - no action needed"
    elif score >= -0.05:
        status = "Touched"
        alert = "info"
        message = "🔔 Bowl has been touched - cat is eating"
    else:
        status = "Refill needed"
        alert = "error"
        message = "⚠️ Refill needed - bowl is low!"

    return status, score, message, alert


# ============================================================
# APP LAYOUT
# Everything below this point is what the user sees.
# ============================================================
st.title("Cat Bowl Monitor 🐱")

# Load models at startup - thanks to @st.cache_resource this
# only actually runs once no matter how many times the page refreshes
st.write("Loading models...")
autoencoder, svm = load_models()
st.write("✅ Models loaded and ready!")

# How often to check the bowl (in seconds)
CHECK_INTERVAL = 1800

st.write(f"Checking bowl every {CHECK_INTERVAL / 60} minutes automatically")

# ---- Placeholders ----
# st.empty() reserves a spot on the page without putting anything there yet.
# Each time we check the bowl we REPLACE the content in these spots
# instead of adding new content below the old content.
# Without placeholders, every 2 minutes a new image and status would
# appear below the old ones and the page would grow forever.
# Create two columns - left is wider for the image
# right is narrower for the status info
# The numbers (2,1) control the ratio - image gets 2/3, status gets 1/3
col1, col2 = st.columns([2, 1])

# Create placeholders inside each column
with col1:
    image_placeholder = st.empty()

with col2:
    status_placeholder = st.empty()
    score_placeholder = st.empty()
    time_placeholder = st.empty()

    st.divider()  # draws a horizontal line to separate alerts from buttons
    st.subheader("📸 Collect Training Data")

    # Define the three folder paths
    train_good_path = r'C:\Users\ylazz\Desktop\defect-detection\data\raw\archive\cat_food\train\good'
    test_good_path = r'C:\Users\ylazz\Desktop\defect-detection\data\raw\archive\cat_food\test\good'
    test_eaten_path = r'C:\Users\ylazz\Desktop\defect-detection\data\raw\archive\cat_food\test\eaten'

    # Button 1: Save full bowl as training image
    if st.button("💾 Save as Train Good"):
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{train_good_path}\\camo_{timestamp}.jpg"
            st.session_state.current_image.save(filename)
            st.success(f"✅ Saved to train/good!")
        except Exception as e:
            st.error(f"Could not save: {e}")

    # Button 2: Save full bowl as test good image
    if st.button("💾 Save as Test Good"):
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{test_good_path}\\camo_{timestamp}.jpg"
            st.session_state.current_image.save(filename)
            st.success(f"✅ Saved to test/good!")
        except Exception as e:
            st.error(f"Could not save: {e}")

    # Button 3: Save eaten/empty bowl as test defective image
    if st.button("💾 Save as Test Eaten"):
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{test_eaten_path}\\camo_{timestamp}.jpg"
            st.session_state.current_image.save(filename)
            st.success(f"✅ Saved to test/eaten!")
        except Exception as e:
            st.error(f"Could not save: {e}")

# ---- Session State ----
# st.session_state is a dictionary that persists between reruns.
# Normal variables get wiped every time Streamlit reruns the script.
# session_state survives - so we can remember what the status was
# last time we checked.
# The 'if not in' check prevents resetting it on every rerun.
if "last_status" not in st.session_state:
    st.session_state.last_status = None

# ============================================================
# MAIN LOOP
# This runs forever, checking the bowl every 2 minutes.
# try/except means if anything goes wrong (phone unreachable,
# WiFi drops, etc.) we show a warning instead of crashing.
# ============================================================
if "current_image" not in st.session_state:
    st.session_state.current_image = None

pygame.mixer.init()
def play_sound():
    pygame.mixer.music.load("meow.mp3")
    pygame.mixer.music.play()

if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    if st.button("▶️ Start Monitoring"):
        st.session_state.started = True
        st.rerun()
    st.stop()  # don't run the loop until the button is clicked

while True:
    try:
        # Grab current frame from phone camera
        image = get_image_from_phone()

        # Run it through the model
        status, score, message, alert = analyze_image(image, autoencoder, svm)

        # Always update the image, score and timestamp
        # regardless of whether status changed
        image_placeholder.image(image, caption="Current bowl", width=300)
        st.session_state.current_image = image
        score_placeholder.write(f"SVM score: {score:.3f}")
        time_placeholder.write(f"Last checked: {time.strftime('%H:%M:%S')}")

        # Only show a prominent alert if the status has CHANGED
        # This prevents from getting the same alert every X minutes
        # when nothing has changed
        if status != st.session_state.last_status:
            if alert == "success":
                status_placeholder.success(f"🆕 Status changed! {message}")
            elif alert == "info":
                status_placeholder.info(f"🆕 Status changed! {message}")
            else:
                status_placeholder.error(f"🆕 Status changed! {message}")
            # Store the new status so we can compare next time
            st.session_state.last_status = status
        else:
            # Status has not changed - show quietly without alarming
            status_placeholder.write(f"Status unchanged: {status}")

        # Play meow sound on EVERY check when refill is needed
        # (not just when status first changes)
        if alert == "error":
            play_sound()

    except Exception as e:
        # If the phone camera is unreachable, show a warning
        # but keep the loop running - it will try again in 2 minutes
        status_placeholder.warning(f"⚠️ Could not reach phone camera: {e}")

    # Wait before checking again
    time.sleep(CHECK_INTERVAL)
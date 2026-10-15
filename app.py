from pathlib import Path
import hashlib
import io

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2


# 1. CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_mobilenetv2.pth"

CLASS_NAMES = ["Cattle", "Buffalo", "Other"]
IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# 2. STREAMLIT PAGE

st.set_page_config(
    page_title="Cattle and Buffalo Classifier",
    page_icon="🐄",
    layout="centered"
)

st.title("🐄 Cattle & Buffalo Classifier")

st.write(
    "Upload an animal image or use your camera to classify it "
    "as Cattle, Buffalo, or Other."
)


# 3. LOAD TRAINED MODEL

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    # Recreate the trained MobileNetV2 architecture
    model = mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    # Load trained weights
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


# 4. IMAGE PREPROCESSING
# Must match the training/evaluation transforms

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 5. PREDICTION FUNCTION

def predict_image(image, model):

    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = probabilities.argmax().item()

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index].item()

    probability_dict = {
        CLASS_NAMES[i]: probabilities[i].item()
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, probability_dict


# 6. CHOOSE IMAGE SOURCE
# Device upload is selected by default

st.subheader("Choose Image Source")

image_source = st.radio(
    "Select how you want to provide an image:",
    options=[
        "Upload from Device",
        "Use Camera"
    ],
    index=0,
    horizontal=True
)

uploaded_file = None

if image_source == "Upload from Device":

    uploaded_file = st.file_uploader(
        "Choose an animal image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Select an image stored on your device."
    )

else:

    uploaded_file = st.camera_input(
        "Take a picture of the animal"
    )


# 7. DISPLAY IMAGE AND WAIT FOR PREDICTION BUTTON

if uploaded_file is not None:

    try:
        image_bytes = uploaded_file.getvalue()

        # Identify the current image so old predictions
        # are not shown for a newly selected image.
        image_key = hashlib.sha256(
            image_source.encode("utf-8") + image_bytes
        ).hexdigest()

        if st.session_state.get("current_image_key") != image_key:

            st.session_state["current_image_key"] = image_key

            st.session_state.pop(
                "prediction_result",
                None
            )

        with Image.open(io.BytesIO(image_bytes)) as img:
            image = img.convert("RGB")

        if image_source == "Upload from Device":
            st.subheader("Selected Image")
            st.image(
                image,
                use_container_width=True
            )


        # Prediction happens ONLY when the button is clicked
        if st.button(
            "Predict",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Analyzing image..."):

                model = load_model()

                predicted_class, confidence, probabilities = (
                    predict_image(image, model)
                )

                st.session_state["prediction_result"] = {
                    "image_key": image_key,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities
                }

        # DISPLAY SAVED RESULT
        # Result remains visible across normal reruns

        result = st.session_state.get("prediction_result")

        if result and result["image_key"] == image_key:

            st.divider()

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Predicted Class",
                    value=result["predicted_class"]
                )

            with col2:
                st.metric(
                    label="Confidence",
                    value=f"{result['confidence'] * 100:.2f}%"
                )

            if result["confidence"] < 0.65:
                st.warning(
                    "The prediction has low confidence. "
                    "Try a clearer image."
                )

            st.subheader("Class Probabilities")

            for class_name in CLASS_NAMES:

                probability = result["probabilities"][class_name]

                st.write(
                    f"**{class_name}: "
                    f"{probability * 100:.2f}%**"
                )

                st.progress(probability)

            st.caption(
                "Confidence is a model estimate, not a guarantee. "
                "Unfamiliar images may still receive high confidence."
            )

    except Exception as error:
        st.error(f"Could not process the image: {error}")

else:

    # Clear old results when no image is selected
    st.session_state.pop("current_image_key", None)
    st.session_state.pop("prediction_result", None)

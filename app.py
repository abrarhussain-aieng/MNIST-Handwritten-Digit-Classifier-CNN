import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn.keras")


model = load_model()


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔢 MNIST Handwritten Digit Classifier")

st.write(
    "Upload an image of a digit and let the CNN predict it."
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Convert to NumPy
    img_array = np.array(image)

    # --------------------------------------------------------
    # Convert to white digit on black background
    # --------------------------------------------------------

    if np.mean(img_array) > 127:
        image = ImageOps.invert(image)

    # Improve contrast
    image = ImageOps.autocontrast(image)

    # --------------------------------------------------------
    # Threshold image
    # --------------------------------------------------------

    binary = image.point(
        lambda p: 255 if p > 40 else 0
    )

    # Find the digit
    bbox = binary.getbbox()

    if bbox is not None:
        image = image.crop(bbox)

    # --------------------------------------------------------
    # ADD PADDING
    # --------------------------------------------------------

    width, height = image.size

    padding = int(
        max(width, height) * 0.20
    )

    new_width = width + (padding * 2)
    new_height = height + (padding * 2)

    padded = Image.new(
        "L",
        (new_width, new_height),
        0
    )

    padded.paste(
        image,
        (padding, padding)
    )

    image = padded

    # --------------------------------------------------------
    # MAKE IMAGE SQUARE
    # --------------------------------------------------------

    width, height = image.size

    size = max(width, height)

    canvas = Image.new(
        "L",
        (size, size),
        0
    )

    x = (size - width) // 2
    y = (size - height) // 2

    canvas.paste(
        image,
        (x, y)
    )

    # --------------------------------------------------------
    # RESIZE TO 28 × 28
    # --------------------------------------------------------

    image = canvas.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------------------------

    image_array = np.array(image)

    # Same normalization used during training
    image_array = image_array.astype(
        "float32"
    ) / 255.0

    # CNN input shape:
    # (1, 28, 28, 1)

    image_array = image_array.reshape(
        1,
        28,
        28,
        1
    )

    return image_array, image


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Show original image
    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    # Predict button
    if st.button("Predict Digit"):

        # Preprocess
        processed_image, display_image = (
            preprocess_image(image)
        )

        # ----------------------------------------------------
        # SHOW PROCESSED IMAGE
        # ----------------------------------------------------

        st.subheader("Processed Image")

        st.image(
            display_image,
            caption="Image sent to the CNN",
            width=200
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            processed_image,
            verbose=0
        )[0]

        # Predicted digit
        predicted_digit = int(
            np.argmax(prediction)
        )

        # Confidence
        confidence = float(
            np.max(prediction)
        ) * 100

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.success(
            f"Predicted Digit: {predicted_digit}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.subheader(
            "Prediction Probabilities"
        )

        for digit, probability in enumerate(
            prediction
        ):

            probability_percent = (
                float(probability) * 100
            )

            st.write(
                f"Digit {digit}: "
                f"{probability_percent:.2f}%"
            )

            st.progress(
                float(probability)
            )
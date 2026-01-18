import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

# --- CONFIGURATION ---
MODEL_PATH = 'mnist_model.h5'
IMG_SIZE = 28


# --- LOAD MODEL ---
# --- LOAD MODEL (UPDATED) ---
@st.cache_resource
def load_trained_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)

        # --- FIX: FORCE BUILD THE GRAPH ---
        # We pass a dummy zero array through the model.
        # This fixes the "layer has never been called" error.
        dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
        _ = model(dummy_input)

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Run train_model.py first.")
        return None


model = load_trained_model()


# --- GRAD-CAM FUNCTION (ROBUST FIX) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # 1. Find the index of the target layer
    layers = model.layers
    target_layer_index = None
    for i, layer in enumerate(layers):
        if layer.name == last_conv_layer_name:
            target_layer_index = i
            break

    if target_layer_index is None:
        return np.zeros((28, 28))  # Fallback if layer not found

    # 2. Compute Gradients using explicit forward pass
    with tf.GradientTape() as tape:
        # Part A: Run from Input -> Target Layer
        x = img_array
        for layer in layers[:target_layer_index + 1]:
            x = layer(x)

        # This is the "activations" we want to visualize
        conv_output = x

        # CRITICAL: Watch this tensor!
        # This tells TF: "Track how changes here affect the output"
        tape.watch(conv_output)

        # Part B: Run from Target Layer -> Output
        for layer in layers[target_layer_index + 1:]:
            x = layer(x)

        preds = x

        # Part C: Get the score for the top predicted class
        top_pred_index = tf.argmax(preds[0])
        class_channel = preds[:, top_pred_index]

    # 3. Calculate Gradient
    # This will no longer be None because we manually connected the path
    grads = tape.gradient(class_channel, conv_output)

    # 4. Process Gradients (Pool & Weight)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 5. Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Safe handling for NaN (div by zero)
    if np.isnan(heatmap).any():
        return np.zeros((28, 28))

    return heatmap.numpy()
# --- UI LAYOUT ---
st.title("CNN Digit Recognition & Grad-CAM")
st.markdown("Draw a digit (0-9) to see the prediction and model activation.")

col1, col2 = st.columns(2)

with col1:
    st.write("### Input")
    canvas = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=15,
        stroke_color="#FFFFFF",  # White drawing on Black background (MNIST style)
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.write("### Analysis")
    if canvas.image_data is not None and model is not None:
        # Preprocessing
        img = cv2.cvtColor(canvas.image_data, cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        if np.sum(img) > 0.1:  # If canvas is not empty
            prediction = model.predict(img, verbose=0)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction)

            st.metric("Prediction", f"{predicted_label}", f"{confidence:.2%}")

            # Find last conv layer for Grad-CAM
            layer_names = [layer.name for layer in model.layers]
            conv_layers = [name for name in layer_names if 'conv' in name]

            if conv_layers:
                heatmap = make_gradcam_heatmap(img, model, conv_layers[-1])

                # Resize heatmap to display
                heatmap = cv2.resize(heatmap, (280, 280))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                st.image(heatmap, caption="Where the model looked (Grad-CAM)")
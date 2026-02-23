from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import cv2

# ✅ PDF imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter

app = Flask(__name__)

model = load_model("pneumonia_model.h5", compile=False)

last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break


def get_gradcam_heatmap(img_array):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# ✅ PDF Generation Function
def generate_pdf(result, confidence, advice, img_path, heatmap_path):
    pdf_path = os.path.join("static", "report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("Pneumonia Detection Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Prediction: {result}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"Confidence: {confidence}%", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"Advice: {advice}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("Uploaded X-ray:", styles["Heading3"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Image(img_path, width=3 * inch, height=3 * inch))
    elements.append(Spacer(1, 0.3 * inch))

    if heatmap_path:
        elements.append(Paragraph("Abnormality Heatmap:", styles["Heading3"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Image(heatmap_path, width=3 * inch, height=3 * inch))

    doc.build(elements)

    return pdf_path


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    if confidence > 0.5:
        result = "PNEUMONIA"
        confidence_percent = round(confidence * 100, 2)
        advice = "⚠ Abnormality detected. Please consult a doctor immediately."
    else:
        result = "NORMAL"
        confidence_percent = round((1 - confidence) * 100, 2)
        advice = "✔ No signs of pneumonia detected. No immediate concern."

    heatmap_path = None

    if result == "PNEUMONIA":
        heatmap = get_gradcam_heatmap(img_array)
        heatmap = cv2.resize(heatmap, (150, 150))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original_img = cv2.imread(filepath)
        original_img = cv2.resize(original_img, (150, 150))

        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        heatmap_path = os.path.join("static", "heatmap_" + file.filename)
        cv2.imwrite(heatmap_path, superimposed_img)

    # ✅ Generate PDF
    pdf_path = generate_pdf(result, confidence_percent, advice, filepath, heatmap_path)

    return render_template("index.html",
                           prediction=result,
                           confidence=confidence_percent,
                           advice=advice,
                           img_path=filepath,
                           heatmap_path=heatmap_path,
                           pdf_path=pdf_path)


if __name__ == "__main__":
    app.run(debug=True)
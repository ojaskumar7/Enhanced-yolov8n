import io
import base64
import random
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = "best.pt"          # change if your file lives elsewhere
model = YOLO(MODEL_PATH)
print(f"Model loaded from '{MODEL_PATH}'")

# ── Distinct colours per class (auto-generated) ───────────────────────────────
def _class_colour(class_id: int) -> tuple:
    """Return a consistent RGB colour for a given class index."""
    rng = random.Random(class_id * 2654435761)   # deterministic per class
    return (rng.randint(50, 230), rng.randint(50, 230), rng.randint(50, 230))


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read image
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Run inference
    results = model(image, verbose=False)[0]

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.0%}"
        colour = _class_colour(cls)

        # Box (3 px border for visibility)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=7)

        # Label background + text
        text_w, text_h = draw.textlength(label, font=None), 12
        pad = 3
        draw.rectangle(
            [x1, y1 - text_h - pad * 2, x1 + int(text_w) + pad * 2, y1],
            fill=colour,
        )
        draw.text((x1 + pad, y1 - text_h - pad), label, fill="white")

        detections.append({"class": model.names[cls], "confidence": round(conf, 3),
                           "box": [x1, y1, x2, y2]})

    # Encode result image as base64
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "image":      f"data:image/jpeg;base64,{encoded}",
        "count":      len(detections),
        "detections": detections,
    })


if __name__ == "__main__":
    print("🚀  Starting server → http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
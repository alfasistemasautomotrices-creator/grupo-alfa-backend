"""
Grupo Alfa - Catalog Extractor Backend
Flask + OpenCV + pdf2image + PaddleOCR
"""

import os
import re
import json
import uuid
import zipfile
import threading
import queue
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, Response, abort
from flask_cors import CORS
from pdf2image import convert_from_bytes

# --- OCR ---------------------------------------------------------------
from paddleocr import PaddleOCR

# CORREGIDO ✅
OCR = PaddleOCR(use_angle_cls=True, lang="es")

# --- Configuración -----------------------------------------------------
BASE_DIR = Path(__file__).parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

JOBS = {}
PROGRESS_QUEUES = {}


def push_progress(job_id, payload):
    q = PROGRESS_QUEUES.get(job_id)
    if q is not None:
        q.put(payload)


# --- OpenCV ------------------------------------------------------------
def detect_product_boxes(img_bgr, min_area_ratio=0.01, max_area_ratio=0.6):
    h_img, w_img = img_bgr.shape[:2]
    page_area = h_img * w_img

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        10,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    closed = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        area = w * h
        ratio = area / page_area
        aspect = w / float(h) if h else 0

        if (
            min_area_ratio <= ratio <= max_area_ratio
            and 0.3 <= aspect <= 4.0
            and w > 80
            and h > 80
        ):
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: (round(b[1] / 50), b[0]))

    return boxes


# --- OCR helpers -------------------------------------------------------
PART_RE = re.compile(
    r"\b([A-Z0-9]{2,}[\-\/]?[A-Z0-9]+(?:[\-\/][A-Z0-9]+)*)\b"
)

KNOWN_BRANDS = {
    "BOSCH",
    "ACDELCO",
    "DENSO",
    "NGK",
    "MONROE",
    "BREMBO",
    "MAHLE",
    "GATES",
    "SKF",
    "VALEO",
    "TRW",
    "AISIN",
}


def ocr_block(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    result = OCR.ocr(rgb, cls=True)

    lines = []

    if result and result[0]:
        for line in result[0]:
            txt = line[1][0].strip()

            if txt:
                lines.append(txt)

    full_text = " | ".join(lines)

    upper = full_text.upper()

    brand = next((b for b in KNOWN_BRANDS if b in upper), "")

    part_number = ""

    for m in PART_RE.finditer(upper):
        candidate = m.group(1)

        if any(ch.isdigit() for ch in candidate):
            part_number = candidate
            break

    application = lines[-1] if lines else ""

    return {
        "part_number": part_number or "SIN_NUM",
        "brand": brand or "GENERICO",
        "application": application,
        "raw_text": full_text,
    }


def safe_filename(s):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_") or "PIEZA"


# --- Pipeline ----------------------------------------------------------
def process_pdf_job(job_id, pdf_bytes, dpi=200):
    job_dir = JOBS_DIR / job_id

    images_dir = job_dir / "images"

    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        push_progress(
            job_id,
            {
                "stage": "convert",
                "progress": 5,
                "status": "Convirtiendo páginas...",
            },
        )

        pages = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="png")

        all_parts = []

        used_names = set()

        for page_idx, pil_page in enumerate(pages, start=1):

            img = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)

            boxes = detect_product_boxes(img)

            for x, y, w, h in boxes:

                crop = img[y : y + h, x : x + w]

                ocr = ocr_block(crop)

                base = (
                    f"{safe_filename(ocr['brand'])}_"
                    f"{safe_filename(ocr['part_number'])}"
                )

                name = f"{base}.png"

                n = 2

                while name in used_names:
                    name = f"{base}_{n}.png"
                    n += 1

                used_names.add(name)

                cv2.imwrite(str(images_dir / name), crop)

                all_parts.append(
                    {
                        "filename": name,
                        "part_number": ocr["part_number"],
                        "brand": ocr["brand"],
                        "application": ocr["application"],
                        "page": page_idx,
                    }
                )

        zip_path = job_dir / "piezas.zip"

        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as zf:

            for p in all_parts:
                zf.write(
                    images_dir / p["filename"],
                    arcname=p["filename"],
                )

        csv_path = job_dir / "catalogo.csv"

        pd.DataFrame(all_parts).to_csv(csv_path, index=False)

        result = {
            "job_id": job_id,
            "total": len(all_parts),
            "parts": all_parts,
            "zip_url": f"/download/zip/{job_id}",
            "csv_url": f"/download/csv/{job_id}",
        }

        JOBS[job_id] = {
            "status": "done",
            "result": result,
        }

    except Exception as e:

        JOBS[job_id] = {
            "status": "error",
            "error": str(e),
        }


# --- Flask -------------------------------------------------------------
@app.route("/health")
def health():
    return {"ok": True}


@app.route("/process-pdf", methods=["POST"])
def process_pdf():

    if "file" not in request.files:
        return jsonify({"error": "Falta archivo"}), 400

    f = request.files["file"]

    pdf_bytes = f.read()

    job_id = uuid.uuid4().hex[:12]

    JOBS[job_id] = {"status": "processing"}

    PROGRESS_QUEUES[job_id] = queue.Queue()

    t = threading.Thread(
        target=process_pdf_job,
        args=(job_id, pdf_bytes),
        daemon=True,
    )

    t.start()

    return jsonify(
        {
            "job_id": job_id,
            "progress_url": f"/progress/{job_id}",
            "result_url": f"/result/{job_id}",
        }
    )


@app.route("/result/<job_id>")
def get_result(job_id):

    job = JOBS.get(job_id)

    if not job:
        abort(404)

    return jsonify(job)


@app.route("/download/zip/<job_id>")
def download_zip(job_id):

    p = JOBS_DIR / job_id / "piezas.zip"

    if not p.exists():
        abort(404)

    return send_file(
        p,
        as_attachment=True,
        download_name=f"grupo-alfa-{job_id}.zip",
    )


@app.route("/download/csv/<job_id>")
def download_csv(job_id):

    p = JOBS_DIR / job_id / "catalogo.csv"

    if not p.exists():
        abort(404)

    return send_file(
        p,
        as_attachment=True,
        download_name=f"grupo-alfa-{job_id}.csv",
    )


@app.route("/image/<job_id>/<path:filename>")
def get_image(job_id, filename):

    p = JOBS_DIR / job_id / "images" / filename

    if not p.exists():
        abort(404)

    return send_file(
        p,
        mimetype="image/png",
    )


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True,
        threaded=True,
  )

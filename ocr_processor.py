import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import io
import os

# Set tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR'

def check_pdf_type(pdf_bytes):
    """
    Check if PDF is text-based or scanned image.
    Returns: 'text', 'scanned', or 'mixed'
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_pages = 0
        scanned_pages = 0
        for page in reader.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                text_pages += 1
            else:
                scanned_pages += 1
        total = text_pages + scanned_pages
        if total == 0:
            return 'unknown'
        if scanned_pages == 0:
            return 'text'
        if text_pages == 0:
            return 'scanned'
        return 'mixed'
    except Exception as e:
        return 'unknown'

def preprocess_image_for_ocr(image):
    """
    Use OpenCV to preprocess image for better OCR accuracy.
    Steps: grayscale → denoise → threshold → deskew
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold for better text contrast
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Deskew — fix tilted scans
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 10:  # Only fix small tilts
            h, w = thresh.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            thresh = cv2.warpAffine(
                thresh, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

    return thresh

def extract_text_from_scanned_pdf(pdf_bytes):
    """
    Extract text from scanned PDF using OpenCV + Tesseract OCR.
    Returns list of (page_num, text) tuples.
    """
    try:
        # Convert PDF pages to images
        images = convert_from_bytes(pdf_bytes, dpi=300)
        results = []
        for page_num, image in enumerate(images):
            # Preprocess with OpenCV
            processed = preprocess_image_for_ocr(image)
            # Run Tesseract OCR
            processed_pil = Image.fromarray(processed)
            text = pytesseract.image_to_string(
                processed_pil,
                config='--psm 6 --oem 3'
            )
            if text.strip():
                results.append((page_num, text.strip()))
        return results
    except Exception as e:
        return []

def extract_images_from_pdf(pdf_bytes):
    """
    Extract images/charts from PDF pages using OpenCV.
    Returns list of (page_num, image) tuples.
    """
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150)
        extracted = []
        for page_num, page_image in enumerate(images):
            img_array = np.array(page_image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Detect regions that look like charts/images
            # using edge detection
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Find large rectangular regions (likely charts/tables)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10000:  # Large enough to be a chart
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.3 < aspect_ratio < 3.5:  # Reasonable shape
                        cropped = page_image.crop((x, y, x+w, y+h))
                        extracted.append((page_num, cropped))
                        break  # One per page to avoid too many

        return extracted[:5]  # Max 5 images
    except Exception as e:
        return []

def process_uploaded_pdf(pdf_bytes, filename):
    """
    Main function — intelligently processes any uploaded PDF.
    1. Detects if text or scanned
    2. Uses appropriate extraction method
    3. Returns extracted text and metadata
    """
    result = {
        "text": "",
        "pdf_type": "unknown",
        "pages_processed": 0,
        "ocr_used": False,
        "images_found": 0,
        "quality_score": 0,
        "message": ""
    }

    try:
        # Step 1: Check PDF type
        pdf_type = check_pdf_type(pdf_bytes)
        result["pdf_type"] = pdf_type

        if pdf_type == 'text':
            # Normal text extraction
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            full_text = ""
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += f"\n[Page {page_num}]\n{text}"
            result["text"] = full_text
            result["pages_processed"] = len(reader.pages)
            result["ocr_used"] = False
            result["quality_score"] = 95
            result["message"] = f"Text-based PDF — extracted directly ({len(reader.pages)} pages)"

        elif pdf_type in ['scanned', 'mixed']:
            # Use OpenCV + OCR
            ocr_results = extract_text_from_scanned_pdf(pdf_bytes)
            full_text = ""
            for page_num, text in ocr_results:
                full_text += f"\n[Page {page_num}]\n{text}"
            result["text"] = full_text
            result["pages_processed"] = len(ocr_results)
            result["ocr_used"] = True
            result["quality_score"] = 75
            result["message"] = f"Scanned PDF — OCR applied ({len(ocr_results)} pages processed)"

        else:
            # Try text first, fall back to OCR
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            full_text = ""
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 20:
                    full_text += f"\n[Page {page_num}]\n{text}"
            if full_text.strip():
                result["text"] = full_text
                result["ocr_used"] = False
                result["message"] = "Extracted using direct text method"
            else:
                ocr_results = extract_text_from_scanned_pdf(pdf_bytes)
                for page_num, text in ocr_results:
                    full_text += f"\n[Page {page_num}]\n{text}"
                result["text"] = full_text
                result["ocr_used"] = True
                result["message"] = "Fell back to OCR extraction"

        # Step 2: Check for images
        images = extract_images_from_pdf(pdf_bytes)
        result["images_found"] = len(images)

        # Step 3: Calculate quality score
        if result["text"]:
            word_count = len(result["text"].split())
            if word_count > 1000:
                result["quality_score"] = 95
            elif word_count > 500:
                result["quality_score"] = 80
            elif word_count > 100:
                result["quality_score"] = 60
            else:
                result["quality_score"] = 30

        return result, images

    except Exception as e:
        result["message"] = f"Error: {str(e)}"
        return result, []
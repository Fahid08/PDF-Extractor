
import os 
import json
import fitz
from PIL import Image
import pytesseract
# import torch
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification


def extract_pages(document_path, page_numbers, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    document = fitz.open(document_path)

    for page_number in page_numbers:
        page = document.load_page(page_number)
        pix = page.get_pixmap()
        output_image = os.path.join(save_path, f"page_{page_number + 1}.png")
        pix.save(output_image)

def ocr_from_image(image_path):
    image = Image.open(image_path)
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = [word for word in ocr_data['text'] if word.strip()]
    bounding_boxes = [[left, top, left + width, top + height] for left, top, width, height in zip(ocr_data['left'], ocr_data['top'], ocr_data['width'], ocr_data['height']) if width and height]
    return words, bounding_boxes

def detect_page_labels(save_path):
    extracted_pages = [os.path.join(save_path, f) for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]

    for page in extracted_pages:
        print(page)
    
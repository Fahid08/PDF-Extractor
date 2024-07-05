from helpers import extract_pages, detect_page_labels, ocr_from_image
import os

extraction_save_path = "extracted_pages"
document_path = "dummy_data.pdf"
page_numbers = [23, 26]

extract_pages(document_path, page_numbers, extraction_save_path)

extracted_pages = [os.path.join(extraction_save_path, f) for f in os.listdir(extraction_save_path) if os.path.isfile(os.path.join(extraction_save_path, f))]

words, bbox = ocr_from_image(extracted_pages[0])

for i in range(len(words)):
    print(words[i], bbox[i])

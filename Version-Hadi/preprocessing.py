from helpers import extract_text, extract_tables_from_pdf
import os

data_file_path = '../sample_data.pdf'
extracted_text_save_path = 'extracted_text.txt'

if not os.path.exists(extracted_text_save_path):
    extracted_text = extract_text(data_file_path)

    with open(extracted_text_save_path, 'w', encoding='utf-8') as file:
        file.write(extracted_text)

# flavor = 'stream'
# tables = extract_tables_from_pdf(data_file_path, flavor=flavor)

# # Print extracted tables
# for i, table in enumerate(tables):
#     print(f"Table {i+1}")
#     print(table)
#     print("\n")
import camelot
import pandas as pd
import fitz
import re

## Text
def is_table_row(line):
    line_text = "".join(span["text"] for span in line["spans"])
    
    pattern = r'(\d+)|([-\s;,.]+)|([A-Za-z]\s[A-Za-z])'
    # r'^[0-9!@#$%^&*()\-_=+\[\]{};:\'",.<>/?\\|]+$'
    
    # case to handle lines containing only numbers and special characters
    if re.search(pattern, line_text) and len(line_text.split()) <= 10 :
        return True
    # case to handle text lines that are not a part of unstructured text
    if len(line_text.split()) < 5 and not line_text.strip().endswith('.'):
        return True
    
    return False


def extract_text(data_file_path):
    doc = fitz.open(data_file_path)
    text_data = ""

    # adjust number of pages to extract
    for page_num in range(5):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 0:  
                for line in block.get("lines", []):
                    if not is_table_row(line):
                        line_text = "".join(span["text"] for span in line["spans"])
                        text_data += line_text + "\n"

    return text_data

## Tables
def extract_tables_from_pdf(data_file_path, flavor='lattice'):    
    tables = camelot.read_pdf(data_file_path, pages='all', flavor=flavor, strip_text='\n')
    
    dfs = []
    for table in tables:
        df = table.df
        if stream_is_table(df):
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            dfs.append(df)
    
    return dfs

def stream_is_table(df):
    if df.shape[0] < 2 or df.shape[1] < 2:
        return False

    delimiter_counts = df.apply(lambda x: x.str.count(r'\s+').mean(), axis=1)
    if delimiter_counts.var() > 1:
        return False

    contains_numbers = df.apply(lambda x: x.str.contains('\d').any(), axis=1)
    contains_text = df.apply(lambda x: x.str.contains('[A-Za-z]').any(), axis=1)
    if not (contains_numbers.any() and contains_text.any()):
        return False

    return True
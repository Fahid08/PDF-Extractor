# Import Libraries
import spacy
import pdfplumber
from spacy.matcher import Matcher
import re
# Load Data and pre-process it!
pdf_path = "output1.pdf"
with pdfplumber.open(pdf_path) as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
# Use SpaCy to perform text processing steps!
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
entities_of_interest = ["ORG", "PERCENT", "GPE", "CARDINAL", "QUANTITY"]
extracted_info = []
for ent in doc.ents:
    if ent.label_ in entities_of_interest:
        extracted_info.append((ent.text, ent.label_))

# Matcher and labelling Entities and Values
matcher = Matcher(nlp.vocab)
# Define patterns for gases and other specific terms
patterns = [
    {"label": "GAS", "pattern": [{"LOWER": "co2"}]},
    {"label": "GAS", "pattern": [{"LOWER": "ch4"}]},
    {"label": "GAS", "pattern": [{"LOWER": "nh3"}]},
]

for pattern in patterns:
    matcher.add(pattern["label"], [pattern["pattern"]])

matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    extracted_info.append((span.text, nlp.vocab.strings[match_id]))

# Print extracted information
for info in extracted_info:
    print(info)
    
def extract_values_and_units(text):
    pattern = r"(\d+\.?\d*)\s?(%|ppm|ppb|g|kg|t|tonnes)"
    matches = re.findall(pattern, text)
    return matches

# Extract values and units
values_and_units = extract_values_and_units(text)
print(values_and_units)

# Combine entities with values and units
final_extraction = []
for entity in extracted_info:
    for value, unit in values_and_units:
        if entity[1] in ["PERCENT", "QUANTITY"]:
            final_extraction.append(f"{entity[0]} {value}{unit}")

# Print final extraction
for item in final_extraction:
    print(item)

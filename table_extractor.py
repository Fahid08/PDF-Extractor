import camelot

# Path to the PDF file
pdf_path = "dummy_data.pdf"

# Extract tables from the PDF
tables = camelot.read_pdf(pdf_path, pages='24')

# Print the number of tables found
print(f"Total tables extracted: {len(tables)}")

# Print and analyze the extracted tables
for i, table in enumerate(tables):
    print(f"Table {i}:")
    print(table.df)  # DataFrame of the extracted table
    # Optionally export to CSV or other formats
    # table.to_csv(f"table_{i}.csv")
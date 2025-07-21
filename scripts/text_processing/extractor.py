import fitz  # PyMuPDF
import os

from clean_text import TextCleaner
from functions import (
    is_two_column_layout,
    extract_text_from_single_column,
    extract_text_from_two_columns)

from utils import (
    PDF_PATHS,
    OUTPUT_PATH)



class ScientificPaperExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.text_cleaner = TextCleaner()

    def _extract_text(self):
        """Main function to Handle different types of documents to extract the text out of them."""
        all_text = ""

        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            print(f"Processing page {page_num + 1}/{self.doc.page_count}...")

            try:
                if is_two_column_layout(page):
                    page_text = extract_text_from_two_columns(page)
                    cleaned_text = self.text_cleaner.clean_text(page_text)
                else:
                    page_text = extract_text_from_single_column(page)
                    cleaned_text = self.text_cleaner.clean_text(page_text)

                all_text += cleaned_text + "\n"
            except Exception as e:
                print(f"Error processing page {page_num + 1}: {e}")

        return all_text

    def save_text_to_file(self, output_path):
        extracted_text = self._extract_text()
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(extracted_text)
        print(f"Text extracted and saved to {output_path}")


# CALL THE EXTRACTOR
if __name__ == "__main__":
    pdf_folder = PDF_PATHS
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)

            print(f"Processing: {filename}")

            extractor = ScientificPaperExtractor(pdf_path)

            output_path = os.path.join(OUTPUT_PATH,
                                       filename.replace(".pdf", ".txt"))
            extractor.save_text_to_file(output_path)

    print("All PDFs processed!")
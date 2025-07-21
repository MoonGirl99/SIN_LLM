import re
import unicodedata


class TextCleaner:
    def __init__(self):
        # Common Unicode replacements for scientific texts
        self.unicode_replacements = {
            '\u2019': "'",  # Smart single quote
            '\u201c': '"',  # Smart left double quote
            '\u201d': '"',  # Smart right double quote
            '\ufb01': 'fi',  # fi ligature
            '\ufb02': 'fl',  # fl ligature
            '\u2018': "'",  # Left single quote
            '\u2013': "-",  # En dash
            '\u2014': "-",  # Em dash
            '\u03b2': 'β',  # Beta symbol
            '\u03b1': 'α',  # Alpha symbol
            '\u03b3': 'γ',  # Gamma symbol
            '\u03bc': 'μ',  # Mu symbol
            '\u03c3': 'σ',  # Sigma symbol
            '\u03c0': 'π',  # Pi symbol
            '\u2260': '≠',  # Not equal
            '\u2264': '≤',  # Less than or equal
            '\u2265': '≥',  # Greater than or equal
            '\u00b0': '°',  # Degree symbol
            '\u00b1': '±',  # Plus-minus symbol
        }

    def clean_unicode(self, text: str) -> str:
        """Clean and normalize Unicode characters."""
        for old, new in self.unicode_replacements.items():
            text = text.replace(old, new)
        text = unicodedata.normalize('NFKD', text)
        return text

    def clean_scientific_notation(self, text: str) -> str:
        """Clean scientific notation and mathematical expressions."""
        text = re.sub(r'(\d+)\s*[×xX]\s*10\s*[-−]?\s*(\d+)', r'\1e-\2', text)  # Scientific notation
        return text

    def clean_citations(self, text: str) -> str:
        """Remove citations while preserving in-text references."""
        text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)  # Parenthetical citations
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)  # Numbered citations
        return text

    def clean_hyphenated_words(self, text: str) -> str:
        """Fix hyphenated words split across lines."""
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)  # Merge hyphenated words
        return text

    def clean_whitespace(self, text: str) -> str:
        """Clean and standardize whitespace."""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()  # Remove leading/trailing whitespace
        return text

    def clean_metadata(self, text: str) -> str:
        """Remove journal names, years, volume numbers, etc."""
        text = re.sub(r'\b[A-Za-z]+\s+Journal\s+of\s+.*?\d{4},.*?\d+-\d+\.?', '', text)
        return text

    def clean_document_structure(self, text: str) -> str:
        """Clean document structural elements."""
        text = re.sub(r'Page \d+\n', '', text)
        text = re.sub(r'(Fig\.|Figure|Table|With\s+\d+\s+figures)\s*\d*\.*', '', text)  # Updated line
        text = re.sub(r'https?://\S+|doi:\S+', '', text)
        text = re.sub(r'\d+ \w+ \d{4}', '', text)
        # email addresse
        text = re.sub(r'\S+@\S+', '', text)
        # DOI or doi
        text = re.sub(r'doi:\S+', '', text)
        text = re.sub(r'DOI:\S+', '', text)
        # remove any other URLs
        text = re.sub(r'http\S+', '', text)
        return text

    def clean_text(self, text: str) -> str:
        """Main cleaning function."""
        text = self.clean_unicode(text)
        text = self.clean_scientific_notation(text)
        text = self.clean_document_structure(text)
        text = self.clean_metadata(text)
        text = self.clean_citations(text)
        text = self.clean_hyphenated_words(text)
        text = self.clean_whitespace(text)
        return text


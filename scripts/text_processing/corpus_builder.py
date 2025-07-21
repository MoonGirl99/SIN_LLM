import os
import json
from pathlib import Path
from typing import List, Dict, Union


class CorpusBuilder:
    """A class for creating JSON corpora from text files."""

    def __init__(self, input_directory: Union[str, Path], output_file: Union[str, Path]):
        self.input_directory = Path(input_directory)
        self.output_file = Path(output_file)
        self.corpus = []

    def process_files(self) -> None:
        """Process all text files in the input directory."""
        if not self.input_directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.input_directory}")

        for txt_file in self.input_directory.glob('*.txt'):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                document = {
                    "file_name": txt_file.name,
                    "full_text": content
                }
                self.corpus.append(document)

            except Exception as e:
                print(f"Error processing {txt_file}: {str(e)}")

    def save_json(self, indent: int = 2) -> None:
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.corpus, f, indent=indent, ensure_ascii=False)
            print(f"Successfully created corpus with {len(self.corpus)} documents")
            print(f"Output saved to: {self.output_file}")

        except Exception as e:
            print(f"Error saving JSON file: {str(e)}")

    def build(self) -> None:
        self.process_files()
        self.save_json()

    def get_corpus(self) -> List[Dict[str, str]]:
        return self.corpus


if __name__ == "__main__":
    builder = CorpusBuilder(
        input_directory="extracted_pdfs",
        output_file="corpus.json"
    )
    builder.build()
# TODO check the cleaning part of the code once again
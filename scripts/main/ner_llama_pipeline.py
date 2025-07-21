import os
import csv
import re
from typing import List, Dict, Any

from llmner import FewShotNer
from llmner.data import AnnotatedDocument, Annotation
from llmner.utils import annotated_document_to_conll

# --- Few-shot configuration ---
FEW_SHOT_ENTITIES = {
    "common_name": "The non-scientific name used to identify a species in everyday language. These names often describe physical characteristics, geographic origin, or behavior of the organism.",
    "scientific_name": "The formal taxonomic name for a species, including binomial nomenclature, abbreviated forms, genus names alone, species names in taxonomic context, and Latin taxonomic descriptors. Scientific names are the standardized nomenclature used in biology to refer to specific species."
}

FEW_SHOT_EXAMPLES = [
    AnnotatedDocument(
        text="The complete sequence of the 16S rRNA gene of Mycoplasma felis, isolated from cats, was determined.",
        annotations=[
            Annotation(start=53, end=68, label="scientific_name", text="Mycoplasma felis"),
            Annotation(start=80, end=84, label="common_name", text="cats"),
        ],
    ),
    AnnotatedDocument(
        text="Mouse interleukin-2 (IL-2) stimulated the proliferation of mouse and rat cells but human IL-2 stimulated rat cells more effectively than mouse cells.",
        annotations=[
            Annotation(start=0, end=5, label="common_name", text="Mouse"),
            Annotation(start=59, end=64, label="common_name", text="mouse"),
            Annotation(start=69, end=72, label="common_name", text="rat"),
            Annotation(start=99, end=102, label="common_name", text="rat"),
            Annotation(start=129, end=134, label="common_name", text="mouse"),
        ],
    ),
    AnnotatedDocument(
        text="Arabidopsis thaliana and Oryza sativa were compared to study evolutionary differences between dicots and monocots.",
        annotations=[
            Annotation(start=0, end=19, label="scientific_name", text="Arabidopsis thaliana"),
            Annotation(start=24, end=36, label="scientific_name", text="Oryza sativa"),
        ],
    ),
    AnnotatedDocument(
        text="E. coli strains were resistant to ampicillin, while S. aureus showed susceptibility to methicillin.",
        annotations=[
            Annotation(start=0, end=7, label="scientific_name", text="E. coli"),
            Annotation(start=45, end=54, label="scientific_name", text="S. aureus"),
        ],
    ),
    AnnotatedDocument(
        text="The red fox, or Vulpes vulpes, is widespread across the Northern Hemisphere.",
        annotations=[
            Annotation(start=4, end=12, label="common_name", text="red fox"),
            Annotation(start=17, end=31, label="scientific_name", text="Vulpes vulpes"),
        ],
    )
]


def chunk_text(text: str, max_tokens: int = 512, chunk_overlap: int = 0) -> List[str]:
    """
    Langchain-style recursive text splitter without dependencies.
    Tries to split first on \n\n, then \n, then ., then space, then hard cut.
    """
    def split_recursive(text, max_tokens):
        separators = ["\n\n", "\n", r"(?<=[.?!])\s", " "]
        for sep in separators:
            parts = re.split(sep, text) if not sep.startswith("(?") else re.split(sep, text)
            chunks, current = [], []
            total_words = 0
            for part in parts:
                word_count = len(part.split())
                if total_words + word_count <= max_tokens:
                    current.append(part)
                    total_words += word_count
                else:
                    if current:
                        chunks.append(" ".join(current).strip())
                    current = [part]
                    total_words = word_count
            if current:
                chunks.append(" ".join(current).strip())
            if all(len(chunk.split()) <= max_tokens for chunk in chunks):
                return chunks
        # fallback: force cut
        words = text.split()
        return [
            " ".join(words[i:i+max_tokens])
            for i in range(0, len(words), max_tokens)
        ]
    base_chunks = split_recursive(text, max_tokens)
    if chunk_overlap > 0:
        overlapped_chunks = []
        for i in range(0, len(base_chunks)):
            current_chunk = base_chunks[i]
            if i > 0:
                previous_chunk = base_chunks[i - 1]
                overlap_words = " ".join(previous_chunk.split()[-chunk_overlap:])
                current_chunk = f"{overlap_words} {current_chunk}"
            overlapped_chunks.append(current_chunk.strip())
        return overlapped_chunks
    return base_chunks


def initialize_fewshot_ner(model: str = None, temperature: float = 0.1, prompting_method: str = "multi_turn") -> FewShotNer:
    """
    Initialize and contextualize the FewShotNer model.
    """
    if model is None:
        model = os.environ.get("LLMNER_MODEL", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
    few_model = FewShotNer(
        model=model,
        temperature=temperature,
        prompting_method=prompting_method,
        final_message_with_all_entities=True
    )
    few_model.contextualize(entities=FEW_SHOT_ENTITIES, examples=FEW_SHOT_EXAMPLES)
    return few_model


def process_s800_abstracts(
    abstracts_dir: str,
    few_model: FewShotNer,
    chunk_size: int = 1024,
    chunk_overlap: int = 0
) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """
    Process abstracts from S800 corpus with the given chunk size.
    Returns (results, conll_results)
    """
    results = []
    conll_results = []
    abstract_files = [f for f in os.listdir(abstracts_dir) if f.endswith('.txt')]
    for filename in abstract_files:
        doc_id = filename[:-4]
        if not allowed_docs or doc_id in allowed_docs:
            with open(os.path.join(abstracts_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
            chunks = chunk_text(content, max_tokens=chunk_size, chunk_overlap=chunk_overlap)
            for i, chunk in enumerate(chunks):
                model_output = few_model.predict([chunk])
                annotations = model_output[0]
                conll_format = annotated_document_to_conll(annotations)
                results.append({
                    'file': filename,
                    'doc_id': doc_id,
                    'content': content,
                    'annotations': annotations.annotations,
                })
                conll_results.append({
                    'doc_id': doc_id,
                    'conll': conll_format
                })
    return results, conll_results


def save_results_to_csv(results: List[Dict[str, Any]], csv_path: str):
    """
    Save the results to a CSV file in the specified format.
    """
    fieldnames = ['file', 'doc_id', 'content', 'Annotation Text', 'start', 'end', 'label']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            base_info = {
                'file': result['file'],
                'doc_id': result['doc_id'],
                'content': result['content'],
            }
            if 'annotations' in result and result['annotations']:
                for annotation in result['annotations']:
                    row = base_info.copy()
                    try:
                        row['Annotation Text'] = annotation['text'] if 'text' in annotation else ''
                        row['start'] = annotation['start'] if 'start' in annotation else ''
                        row['end'] = annotation['end'] if 'end' in annotation else ''
                        row['label'] = annotation['label'] if 'label' in annotation else ''
                    except (TypeError, KeyError):
                        try:
                            row['Annotation Text'] = getattr(annotation, 'text', '')
                            row['start'] = getattr(annotation, 'start', '')
                            row['end'] = getattr(annotation, 'end', '')
                            row['label'] = getattr(annotation, 'label', '')
                        except (AttributeError, TypeError):
                            row['Annotation Text'] = str(annotation)
                            row['start'] = ''
                            row['end'] = ''
                            row['label'] = ''
                    writer.writerow(row)
            else:
                row = base_info.copy()
                row['Annotation Text'] = ''
                row['start'] = ''
                row['end'] = ''
                row['label'] = ''
                writer.writerow(row)


def main():
    """
    Main function to run the pipeline as a script.
    """
    os.environ.setdefault("OPENAI_API_BASE", "https://api.deepinfra.com/v1/openai")
    os.environ.setdefault("OPENAI_API_KEY", "YOUR_API_KEY_HERE") 

    # Paths (can be parameterized)
    S800_DIR = "filepath"
    S800_ABSTRACTS_DIR = os.path.join(S800_DIR, "zoo")
    OUTPUT_CSV = "output.csv"
    
    # Initialize model
    few_model = initialize_fewshot_ner()
    # Process abstracts
    results, conll_results = process_s800_abstracts(
        abstracts_dir=S800_ABSTRACTS_DIR,
        few_model=few_model,
        chunk_size=1024,
        chunk_overlap=0
    )
    # Save results
    save_results_to_csv(results, OUTPUT_CSV)
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main() 
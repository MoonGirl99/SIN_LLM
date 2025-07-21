import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from rapidfuzz import fuzz
import warnings
warnings.filterwarnings('ignore')


def load_taxonomic_terms(file_path):
    """Load the list of generic taxonomic terms to filter out."""
    with open(file_path, 'r', encoding='utf-8') as f:
        terms = [line.strip().lower() for line in f if line.strip()]
    return set(terms)

def simple_post_process(input_csv_path, output_csv_path, taxonomic_terms_path):
    """
    Simple post-processing: remove generic terms, clean text.
    Parameters:
    - input_csv_path: Path to the raw NER results CSV
    - output_csv_path: Path to save the cleaned results
    - taxonomic_terms_path: Path to taxonomic terms list to filter out
    Returns:
    - cleaned_df: Processed DataFrame
    """
    df = pd.read_csv(input_csv_path)
    taxonomic_terms = load_taxonomic_terms(taxonomic_terms_path)
    df = df[df['Annotation Text'].notna() & (df['Annotation Text'] != '')]
    def clean_text(text):
        if pd.isna(text):
            return ''
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    df['Annotation Text'] = df['Annotation Text'].apply(clean_text)
    def is_not_generic_term(text):
        if pd.isna(text) or text == '':
            return False
        text_lower = text.lower().strip()
        return text_lower not in taxonomic_terms
    df = df[df['Annotation Text'].apply(is_not_generic_term)]
    df.to_csv(output_csv_path, index=False)
    return df


def load_ground_truth_from_csv(file_path):
    """
    Load ground truth from the CSV file with no headers.
    The data has 5 columns:
    - Column 1: Numeric ID (e.g., 75485)
    - Column 2: Document ID (e.g., "species012")
    - Column 3: Start position (e.g., 361)
    - Column 4: End position (e.g., 382)
    - Column 5: Entity text (e.g., "black-legged kittiwake")
    """
    df = pd.read_csv(file_path)
    annotations = defaultdict(list)
    for _, row in df.iterrows():
        doc_id = row['doc_id']
        start_pos = int(row['start'])
        end_pos = int(row['end'])
        entity_text = row['Annotation Text']
        label = row['label']
        annotations[doc_id].append({
            'text': entity_text,
            'start': start_pos,
            'end': end_pos,
            'label': label
        })
    return annotations

def prepare_predictions(results_df):
    """
    Convert prediction DataFrame to the format needed for evaluation.
    Use the doc_id column directly, just strip the .txt extension if present.
    """
    fixed_df = results_df.copy()
    fixed_df['clean_doc_id'] = fixed_df['doc_id'].apply(lambda x: x.replace('.txt', '') if isinstance(x, str) else x)
    fixed_df['original_end'] = fixed_df['end']
    fixed_df['end'] = fixed_df['end'].apply(lambda x: int(x)-1 if pd.notnull(x) else x)
    pred_data = defaultdict(list)
    for _, row in fixed_df.iterrows():
        doc_id = row['clean_doc_id']
        if pd.isna(row['start']) or pd.isna(row['end']):
            continue
        pred_data[doc_id].append({
            'text': row['Annotation Text'],
            'start': int(row['start']),
            'end': int(row['end']),
            'label': 'Species'
        })
    return pred_data, fixed_df

def calculate_exact_match_metrics(gt_data, pred_data, position_tolerance=5):
    total_gt = sum(len(entities) for entities in gt_data.values())
    total_pred = sum(len(entities) for entities in pred_data.values())
    true_positives = 0
    match_details = []
    for doc_id, gt_entities in gt_data.items():
        if doc_id not in pred_data:
            continue
        pred_entities = pred_data[doc_id]
        matched_preds = set()
        for gt_entity in gt_entities:
            gt_text = gt_entity['text'].lower()
            for i, pred_entity in enumerate(pred_entities):
                if i in matched_preds:
                    continue
                pred_text = pred_entity['text'].lower()
                text_match = gt_text == pred_text
                start_match = abs(gt_entity['start'] - pred_entity['start']) <= position_tolerance
                end_match = abs(gt_entity['end'] - pred_entity['end']) <= position_tolerance
                if text_match and start_match and end_match:
                    true_positives += 1
                    matched_preds.add(i)
                    match_details.append({
                        'doc_id': doc_id,
                        'gt_text': gt_entity['text'],
                        'pred_text': pred_entity['text'],
                        'gt_start': gt_entity['start'],
                        'pred_start': pred_entity['start'],
                        'gt_end': gt_entity['end'],
                        'pred_end': pred_entity['end']
                    })
                    break
    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': total_pred - true_positives,
        'false_negatives': total_gt - true_positives,
        'match_details': match_details
    }

def calculate_partial_match_metrics(gt_data, pred_data, verbose=False):
    total_gt = sum(len(entities) for entities in gt_data.values())
    total_pred = sum(len(entities) for entities in pred_data.values())
    true_positives = 0
    match_details = []
    for doc_id, gt_entities in gt_data.items():
        if doc_id not in pred_data:
            continue
        pred_entities = pred_data[doc_id]
        matched_preds = set()
        for gt_entity in gt_entities:
            gt_text = gt_entity['text'].lower()
            for i, pred_entity in enumerate(pred_entities):
                if i in matched_preds:
                    continue
                pred_text = pred_entity['text'].lower()
                if (gt_text in pred_text or pred_text in gt_text):
                    true_positives += 1
                    matched_preds.add(i)
                    match_details.append({
                        'doc_id': doc_id,
                        'gt_text': gt_entity['text'],
                        'pred_text': pred_entity['text'],
                        'match_type': 'substring'
                    })
                    if verbose:
                        print(f"Substring match: '{gt_text}' <-> '{pred_text}'")
                    break
    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': total_pred - true_positives,
        'false_negatives': total_gt - true_positives,
        'match_details': match_details
    }

def calculate_enhanced_partial_match_metrics(gt_data, pred_data, use_fuzzy=True, fuzzy_threshold=0.8, use_word_overlap=True, verbose=False):
    total_gt = sum(len(entities) for entities in gt_data.values())
    total_pred = sum(len(entities) for entities in pred_data.values())
    true_positives = 0
    match_details = []
    for doc_id, gt_entities in gt_data.items():
        if doc_id not in pred_data:
            continue
        pred_entities = pred_data[doc_id]
        matched_preds = set()
        for gt_entity in gt_entities:
            gt_text = gt_entity['text'].lower().strip()
            for i, pred_entity in enumerate(pred_entities):
                if i in matched_preds:
                    continue
                pred_text = pred_entity['text'].lower().strip()
                match_type = None
                similarity_score = 0
                if gt_text == pred_text:
                    match_type = "exact"
                    similarity_score = 1.0
                elif gt_text in pred_text or pred_text in gt_text:
                    match_type = "substring"
                    similarity_score = min(len(gt_text), len(pred_text)) / max(len(gt_text), len(pred_text))
                elif use_fuzzy:
                    fuzzy_score = fuzz.ratio(gt_text, pred_text) / 100.0
                    if fuzzy_score >= fuzzy_threshold:
                        match_type = "fuzzy"
                        similarity_score = fuzzy_score
                elif use_word_overlap and len(gt_text.split()) > 1:
                    gt_words = set(gt_text.split())
                    pred_words = set(pred_text.split())
                    overlap = gt_words.intersection(pred_words)
                    if len(overlap) > 0:
                        match_type = "word_overlap"
                        similarity_score = len(overlap) / len(gt_words.union(pred_words))
                if match_type:
                    true_positives += 1
                    matched_preds.add(i)
                    match_details.append({
                        'doc_id': doc_id,
                        'gt_text': gt_entity['text'],
                        'pred_text': pred_entity['text'],
                        'match_type': match_type,
                        'similarity': similarity_score
                    })
                    if verbose:
                        print(f"{match_type.title()} match (sim={similarity_score:.3f}): '{gt_text}' <-> '{pred_text}'")
                    break
    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': total_pred - true_positives,
        'false_negatives': total_gt - true_positives,
        'match_details': match_details
    }

def optimal_entity_matching(gt_entities, pred_entities, similarity_threshold=0.8):
    if not gt_entities or not pred_entities:
        return []
    similarity_matrix = np.zeros((len(gt_entities), len(pred_entities)))
    for i, gt_entity in enumerate(gt_entities):
        gt_text = gt_entity['text'].lower().strip()
        for j, pred_entity in enumerate(pred_entities):
            pred_text = pred_entity['text'].lower().strip()
            if gt_text == pred_text:
                similarity_matrix[i][j] = 1.0
            elif gt_text in pred_text or pred_text in gt_text:
                similarity_matrix[i][j] = 0.9
            else:
                fuzzy_score = fuzz.ratio(gt_text, pred_text) / 100.0
                if fuzzy_score >= similarity_threshold:
                    similarity_matrix[i][j] = fuzzy_score
    matches = []
    used_preds = set()
    potential_matches = []
    for i in range(len(gt_entities)):
        for j in range(len(pred_entities)):
            if similarity_matrix[i][j] > similarity_threshold:
                potential_matches.append((i, j, similarity_matrix[i][j]))
    potential_matches.sort(key=lambda x: x[2], reverse=True)
    used_gt = set()
    for i, j, score in potential_matches:
        if i not in used_gt and j not in used_preds:
            matches.append((i, j, score))
            used_gt.add(i)
            used_preds.add(j)
    return matches

def calculate_optimal_match_metrics(gt_data, pred_data, similarity_threshold=0.8):
    total_gt = sum(len(entities) for entities in gt_data.values())
    total_pred = sum(len(entities) for entities in pred_data.values())
    true_positives = 0
    match_details = []
    for doc_id, gt_entities in gt_data.items():
        if doc_id not in pred_data:
            continue
        pred_entities = pred_data[doc_id]
        matches = optimal_entity_matching(gt_entities, pred_entities, similarity_threshold)
        for gt_idx, pred_idx, similarity in matches:
            true_positives += 1
            match_details.append({
                'doc_id': doc_id,
                'gt_text': gt_entities[gt_idx]['text'],
                'pred_text': pred_entities[pred_idx]['text'],
                'similarity': similarity,
                'match_type': 'optimal'
            })
    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': total_pred - true_positives,
        'false_negatives': total_gt - true_positives,
        'match_details': match_details
    }

def analyze_document_id_mismatches(gt_data, pred_data):
    gt_docs = set(gt_data.keys())
    pred_docs = set(pred_data.keys())
    common_docs = gt_docs.intersection(pred_docs)
    only_in_gt = gt_docs - pred_docs
    only_in_pred = pred_docs - gt_docs
    return {
        'common_docs': common_docs,
        'only_in_gt': only_in_gt,
        'only_in_pred': only_in_pred
    }

def analyze_errors(gt_data, pred_data, use_enhanced_matching=True):
    missed_entities = []
    false_positives = []
    for doc_id, gt_entities in gt_data.items():
        doc_preds = pred_data.get(doc_id, [])
        for gt_entity in gt_entities:
            gt_text = str(gt_entity['text']).lower().strip() if pd.notnull(gt_entity['text']) else ""
            found = False
            for pred_entity in doc_preds:
                pred_text = str(pred_entity['text']).lower().strip() if pd.notnull(pred_entity['text']) else ""
                if use_enhanced_matching:
                    exact_match = gt_text == pred_text
                    substring_match = (gt_text in pred_text or pred_text in gt_text)
                    fuzzy_match = fuzz.ratio(gt_text, pred_text) / 100.0 >= 0.8
                    if exact_match or substring_match or fuzzy_match:
                        found = True
                        break
                else:
                    if (gt_text in pred_text or pred_text in gt_text):
                        found = True
                        break
            if not found:
                missed_entities.append({
                    'doc_id': doc_id,
                    'text': gt_entity['text']
                })
    for doc_id, pred_entities in pred_data.items():
        doc_gt = gt_data.get(doc_id, [])
        for pred_entity in pred_entities:
            pred_text = str(pred_entity['text']).lower().strip() if pd.notnull(pred_entity['text']) else ""
            found = False
            for gt_entity in doc_gt:
                gt_text = str(gt_entity['text']).lower().strip() if pd.notnull(gt_entity['text']) else ""
                if use_enhanced_matching:
                    exact_match = gt_text == pred_text
                    substring_match = (gt_text in pred_text or pred_text in gt_text)
                    fuzzy_match = fuzz.ratio(gt_text, pred_text) / 100.0 >= 0.8
                    if exact_match or substring_match or fuzzy_match:
                        found = True
                        break
                else:
                    if (gt_text in pred_text or pred_text in gt_text):
                        found = True
                        break
            if not found:
                false_positives.append({
                    'doc_id': doc_id,
                    'text': pred_entity['text']
                })
    return {
        'missed_entities': missed_entities[:20],
        'false_positives': false_positives[:50]
    }

def analyze_match_quality(match_details):
    if not match_details:
        return {}
    match_types = {}
    similarity_stats = []
    for match in match_details:
        match_type = match.get('match_type', 'unknown')
        similarity = match.get('similarity', 1.0)
        if match_type not in match_types:
            match_types[match_type] = 0
        match_types[match_type] += 1
        similarity_stats.append(similarity)
    if similarity_stats:
        return {
            'match_type_distribution': match_types,
            'avg_similarity': np.mean(similarity_stats),
            'min_similarity': np.min(similarity_stats),
            'max_similarity': np.max(similarity_stats),
            'similarity_std': np.std(similarity_stats) if len(similarity_stats) > 1 else 0
        }
    else:
        return {
            'match_type_distribution': match_types,
            'avg_similarity': 0,
            'min_similarity': 0,
            'max_similarity': 0,
            'similarity_std': 0
        }

def create_evaluation_summary(results):
    summary = {
        'dataset_info': {
            'total_gt_entities': sum(len(entities) for entities in results['gt_data'].values()),
            'total_pred_entities': sum(len(entities) for entities in results['pred_data'].values()),
            'total_documents': len(results['gt_data']),
            'documents_with_predictions': len(results['pred_data']),
            'document_coverage': len(results['doc_analysis']['common_docs']) / len(results['gt_data'])
        },
        'performance_metrics': {
            'exact_match': results['exact_metrics'],
            'partial_match': results['partial_metrics']
        }
    }
    if 'enhanced_metrics' in results:
        summary['performance_metrics']['enhanced_match'] = results['enhanced_metrics']
        summary['match_quality_analysis'] = analyze_match_quality(results['enhanced_metrics']['match_details'])
    if 'optimal_metrics' in results:
        summary['performance_metrics']['optimal_match'] = results['optimal_metrics']
    return summary

def print_detailed_results(results):
    print("COMPREHENSIVE NER EVALUATION RESULTS")
    total_gt = sum(len(entities) for entities in results['gt_data'].values())
    total_pred = sum(len(entities) for entities in results['pred_data'].values())
    print(f"\nDataset Information:")
    print(f" Ground truth entities: {total_gt}")
    print(f" Predicted entities: {total_pred}")
    print(f" Total documents: {len(results['gt_data'])}")
    print(f"Documents with predictions: {len(results['pred_data'])}")
    print(f" Document coverage: {len(results['doc_analysis']['common_docs'])/len(results['gt_data'])*100:.1f}%")
    print(f"\nPerformance Metrics:")
    metrics_to_show = [
        ('Exact Match', results['exact_metrics']),
        ('Partial Match (Substring)', results['partial_metrics'])
    ]
    if 'enhanced_metrics' in results:
        metrics_to_show.append(('Enhanced Match (Multi-strategy)', results['enhanced_metrics']))
    if 'optimal_metrics' in results:
        metrics_to_show.append(('Optimal Match', results['optimal_metrics']))
    for name, metrics in metrics_to_show:
        print(f"\n{name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
    if 'enhanced_metrics' in results:
        quality = analyze_match_quality(results['enhanced_metrics']['match_details'])
        if quality:
            print(f"\nMatch Quality Analysis:")
            print(f"  Match type distribution: {quality['match_type_distribution']}")
            print(f"  Average similarity: {quality['avg_similarity']:.3f}")
            print(f"  Similarity range: [{quality['min_similarity']:.3f}, {quality['max_similarity']:.3f}]")
    print(f"\nError Analysis:")
    print("Top Missed Entities (False Negatives):")
    for i, entity in enumerate(results['error_analysis']['missed_entities'][:10], 1):
        print(f"  {i:2d}. {entity['text']} ({entity['doc_id']})")
    print("\nTop False Positives:")
    for i, entity in enumerate(results['error_analysis']['false_positives'][:10], 1):
        print(f"  {i:2d}. {entity['text']} ({entity['doc_id']})")

def evaluate_ner_pipeline(predictions_file, ground_truth_file, use_enhanced_matching=True, use_optimal_matching=False, fuzzy_threshold=0.8, position_tolerance=5, verbose=False):
    gt_data = load_ground_truth_from_csv(ground_truth_file)
    results_df = pd.read_csv(predictions_file)
    pred_data, fixed_df = prepare_predictions(results_df)
    doc_analysis = analyze_document_id_mismatches(gt_data, pred_data)
    exact_metrics = calculate_exact_match_metrics(gt_data, pred_data, position_tolerance)
    partial_metrics = calculate_partial_match_metrics(gt_data, pred_data, verbose)
    results = {
        'fixed_df': fixed_df,
        'gt_data': gt_data,
        'pred_data': pred_data,
        'exact_metrics': exact_metrics,
        'partial_metrics': partial_metrics,
        'doc_analysis': doc_analysis
    }
    if use_enhanced_matching:
        enhanced_metrics = calculate_enhanced_partial_match_metrics(
            gt_data, pred_data, 
            use_fuzzy=True, 
            fuzzy_threshold=fuzzy_threshold,
            use_word_overlap=True,
            verbose=verbose
        )
        results['enhanced_metrics'] = enhanced_metrics
    if use_optimal_matching:
        optimal_metrics = calculate_optimal_match_metrics(gt_data, pred_data, fuzzy_threshold)
        results['optimal_metrics'] = optimal_metrics
    error_analysis = analyze_errors(gt_data, pred_data, use_enhanced_matching)
    results['error_analysis'] = error_analysis
    print_detailed_results(results)
    results['summary'] = create_evaluation_summary(results)
    return results

def plot_performance_metrics_bar(results):
    """
    Plot precision, recall, and F1-score for different matching strategies.
    Input: results from evaluate_ner_pipeline(...)
    """
    strategies = []
    precisions = []
    recalls = []
    f1s = []
    if 'exact_metrics' in results:
        strategies.append("Exact")
        precisions.append(results['exact_metrics']['precision'])
        recalls.append(results['exact_metrics']['recall'])
        f1s.append(results['exact_metrics']['f1'])
    if 'partial_metrics' in results:
        strategies.append("Partial")
        precisions.append(results['partial_metrics']['precision'])
        recalls.append(results['partial_metrics']['recall'])
        f1s.append(results['partial_metrics']['f1'])
    if 'enhanced_metrics' in results:
        strategies.append("Enhanced")
        precisions.append(results['enhanced_metrics']['precision'])
        recalls.append(results['enhanced_metrics']['recall'])
        f1s.append(results['enhanced_metrics']['f1'])
    if 'optimal_metrics' in results:
        strategies.append("Optimal")
        precisions.append(results['optimal_metrics']['precision'])
        recalls.append(results['optimal_metrics']['recall'])
        f1s.append(results['optimal_metrics']['f1'])
    df = pd.DataFrame({
        'Strategy': strategies * 3,
        'Metric': ['Precision'] * len(strategies) + ['Recall'] * len(strategies) + ['F1 Score'] * len(strategies),
        'Score': precisions + recalls + f1s
    })
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='Strategy', y='Score', hue='Metric')
    plt.title("NER Performance Metrics by Matching Strategy")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     # Example: Post-process predictions
#     cleaned_results = simple_post_process(
#         input_csv_path="path/to/raw_predictions.csv",
#         output_csv_path="path/to/cleaned_predictions.csv",
#         taxonomic_terms_path="taxonomic_terms_list.txt"
#     )
#     # Example: Evaluate NER pipeline
#     results = evaluate_ner_pipeline(
#         predictions_file="path/to/cleaned_predictions.csv",
#         ground_truth_file="path/to/ground_truth.csv",
#         use_enhanced_matching=True,
#         use_optimal_matching=False,
#         fuzzy_threshold=0.8,
#         position_tolerance=5,
#         verbose=False
#     )
#     # Example: Plot performance
#     plot_performance_metrics_bar(results) 
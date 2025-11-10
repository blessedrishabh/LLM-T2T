# evaluation/evaluate_lotnlg.py

import json
import re
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import pipeline
from tqdm import tqdm

def clean_prediction_text(pred_text):
    """
    Clean prediction text by:
    1. Removing markdown headers (##)
    2. Extracting actual claims from structured outputs
    3. Removing reasoning/meta-text
    """
    # Remove markdown headers
    pred_text = re.sub(r'##\s+', '', pred_text)
    
    # Extract claims from "Claim X:" format if present
    claims = re.findall(r'Claim \d+[^:]*:\s*([^\.]+\.)', pred_text)
    if claims:
        pred_text = ' '.join(claims)
    
    # Remove "Reasoning:" sections
    pred_text = re.sub(r'Reasoning[^:]*:[^\.]+\.', '', pred_text)
    
    # Remove markdown bold **text**
    pred_text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', pred_text)
    
    # Clean up extra whitespace
    pred_text = ' '.join(pred_text.split())
    
    return pred_text

def evaluate_lotnlg(pred_file, gold_file):
    """
    Evaluation for LogicNLG/LoTNLG benchmarking
    Works with both datasets (with or without logical_labels)
    Handles verbose/structured predictions from advanced models
    """
    print("Loading files...")
    with open(pred_file, encoding='utf-8') as f:
        predictions = json.load(f)
    
    with open(gold_file, encoding='utf-8') as f:
        gold_data = json.load(f)
    
    # Create mapping from csv_id to gold data
    csv_id_to_gold = {}
    for idx, gold_item in gold_data.items():
        csv_id = gold_item.get('csv_id')
        if csv_id:
            csv_id_to_gold[csv_id] = gold_item
    
    # Check if dataset has logical labels (LoTNLG) or not (LogicNLG)
    sample_item = next(iter(gold_data.values()))
    has_logical_labels = 'logical_labels' in sample_item
    
    print(f"Found {len(predictions)} predictions")
    print(f"Mapped {len(csv_id_to_gold)} gold examples")
    print(f"Dataset type: {'LoTNLG (with logical labels)' if has_logical_labels else 'LogicNLG (basic)'}\n")
    
    # Initialize NLI model for faithfulness
    print("Loading NLI model (this may take a while)...")
    nli_model = pipeline("text-classification", 
                        model="roberta-large-mnli",
                        device=-1)  # CPU
    
    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    results = {
        'nli_acc': 0,
        'rouge_l': 0,
        'bleu': 0,
    }
    
    if has_logical_labels:
        results['type_em'] = 0
    
    total = 0
    not_found = 0
    
    print("\nEvaluating predictions...")
    for csv_id, pred_data in tqdm(predictions.items()):
        # Match by csv_id
        if csv_id not in csv_id_to_gold:
            not_found += 1
            continue
        
        # Handle both list and string predictions
        if isinstance(pred_data, list):
            pred_text = ' '.join(pred_data)
        else:
            pred_text = str(pred_data)
        
        # Clean verbose/structured predictions
        pred_text_clean = clean_prediction_text(pred_text)
            
        gold_item = csv_id_to_gold[csv_id]
        table_str = gold_item['table_text']
        
        # Get first gold sentence
        gold_sentences = gold_item['sentences']
        if isinstance(gold_sentences, list):
            gold_text = gold_sentences[0]
        else:
            gold_text = str(gold_sentences)
        
        try:
            # NLI Faithfulness Check (use original pred_text for better context)
            nli_input = f"{table_str[:1000]} [SEP] {pred_text[:500]}"
            nli_result = nli_model(nli_input)[0]
            if nli_result['label'] == 'ENTAILMENT':
                results['nli_acc'] += 1
            
            # ROUGE-L Score (use cleaned text)
            rouge_score = rouge.score(gold_text, pred_text_clean)
            results['rouge_l'] += rouge_score['rougeL'].fmeasure
            
            # BLEU Score (use cleaned text)
            reference = [gold_text.split()]
            candidate = pred_text_clean.split()
            bleu = sentence_bleu(reference, candidate)
            results['bleu'] += bleu
            
            # Type matching (only if logical_labels exist)
            if has_logical_labels:
                gold_labels = gold_item['logical_labels']
                if isinstance(gold_labels, list):
                    gold_type = gold_labels[0]
                else:
                    gold_type = str(gold_labels)
                
                # Enhanced type detection for structured outputs
                type_keywords = {
                    'aggregation': ['total', 'sum', 'average', 'mean', 'aggregate'],
                    'superlative': ['most', 'highest', 'lowest', 'best', 'worst', 'maximum', 'minimum', 'largest', 'smallest', 'superlative'],
                    'count': ['number of', 'count', 'how many', 'there are', 'count:'],
                    'comparative': ['more than', 'less than', 'greater', 'higher', 'lower', 'comparative:', 'compared'],
                    'ordinal': ['first', 'second', 'third', 'last', 'ordinal:'],
                    'unique': ['different', 'unique', 'distinct', 'unique:'],
                    'negation': ['not', 'no', 'never', 'did not', 'negation:'],
                }
                
                # Check for explicit label in parentheses (e.g., "Claim 1 (count):")
                explicit_label = re.search(r'\(([^)]+)\):', pred_text)
                if explicit_label:
                    pred_type = explicit_label.group(1).lower().strip()
                else:
                    pred_type = 'surface-level'
                    for type_name, keywords in type_keywords.items():
                        if any(kw in pred_text.lower() for kw in keywords):
                            pred_type = type_name
                            break
                
                if pred_type.lower() in gold_type.lower() or gold_type.lower() in pred_type.lower():
                    results['type_em'] += 1
            
            total += 1
            
        except Exception as e:
            print(f"\nError processing {csv_id}: {e}")
            continue
    
    # Calculate percentages
    if total > 0:
        for key in results:
            results[key] = (results[key] / total) * 100
    
    results['total'] = total
    results['not_found'] = not_found
    results['has_type_em'] = has_logical_labels
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--gold_file", required=True)
    args = parser.parse_args()
    
    scores = evaluate_lotnlg(args.pred_file, args.gold_file)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total evaluated: {scores['total']}")
    print(f"Not found: {scores['not_found']}")
    print(f"\nNLI-Acc: {scores['nli_acc']:.1f}%")
    print(f"ROUGE-L: {scores['rouge_l']:.1f}%")
    print(f"BLEU: {scores['bleu']:.1f}%")
    if scores['has_type_em']:
        print(f"Type EM: {scores['type_em']:.1f}%")
    print("="*50)

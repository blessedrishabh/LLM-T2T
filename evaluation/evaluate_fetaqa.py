# evaluation/evaluate_fetaqa.py

import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm

def evaluate_fetaqa(pred_file, gold_file):
    """
    Evaluate FeTaQA predictions (question answering task)
    """
    print("Loading files...")
    with open(pred_file, encoding='utf-8') as f:
        predictions = json.load(f)
    
    with open(gold_file, encoding='utf-8') as f:
        gold_data = json.load(f)
    
    # Create mapping by feta_id
    feta_id_to_gold = {}
    for idx, gold_item in gold_data.items():
        feta_id = gold_item.get('feta_id')
        if feta_id:
            feta_id_to_gold[str(feta_id)] = gold_item
    
    print(f"Found {len(predictions)} predictions")
    print(f"Mapped {len(feta_id_to_gold)} gold examples\n")
    
    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction()
    
    results = {
        'rouge1': 0,
        'rouge2': 0,
        'rougeL': 0,
        'bleu': 0,
    }
    
    total = 0
    not_found = 0
    
    print("Evaluating predictions...")
    for table_id, pred_text in tqdm(predictions.items()):
        # FeTaQA uses feta_id as key
        if table_id not in feta_id_to_gold:
            not_found += 1
            continue
        
        # Handle list/string predictions
        if isinstance(pred_text, list):
            pred_text = ' '.join(pred_text)
        else:
            pred_text = str(pred_text)
        
        gold_item = feta_id_to_gold[table_id]
        gold_answer = gold_item['answer']
        
        try:
            # ROUGE Scores
            rouge_scores = rouge.score(gold_answer, pred_text)
            results['rouge1'] += rouge_scores['rouge1'].fmeasure
            results['rouge2'] += rouge_scores['rouge2'].fmeasure
            results['rougeL'] += rouge_scores['rougeL'].fmeasure
            
            # BLEU Score with smoothing
            reference = [gold_answer.split()]
            candidate = pred_text.split()
            bleu = sentence_bleu(reference, candidate, smoothing_function=smooth.method1)
            results['bleu'] += bleu
            
            total += 1
            
        except Exception as e:
            print(f"\nError processing {table_id}: {e}")
            continue
    
    # Calculate averages (as percentages)
    if total > 0:
        for key in results:
            results[key] = (results[key] / total) * 100
    
    results['total'] = total
    results['not_found'] = not_found
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--gold_file", required=True)
    args = parser.parse_args()
    
    scores = evaluate_fetaqa(args.pred_file, args.gold_file)
    
    print("\n" + "="*50)
    print("FeTaQA EVALUATION RESULTS")
    print("="*50)
    print(f"Total evaluated: {scores['total']}")
    print(f"Not found: {scores['not_found']}")
    print(f"\nROUGE-1: {scores['rouge1']:.1f}%")
    print(f"ROUGE-2: {scores['rouge2']:.1f}%")
    print(f"ROUGE-L: {scores['rougeL']:.1f}%")
    print(f"BLEU: {scores['bleu']:.1f}%")
    print("="*50)

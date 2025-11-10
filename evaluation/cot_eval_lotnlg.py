# evaluation/cot_eval_lotnlg.py

import json
import asyncio
import openai
from tqdm import tqdm

# Configure API
openai.api_base = "https://api.perplexity.ai"
openai.api_key = "your_api_key"

COT_EVAL_PROMPT_LOTNLG = """Given a table and a generated statement about the table, determine if the statement is factually correct and faithful to the table data.

Table:
{table}

Generated Statement:
{statement}

Please reason step-by-step:
1. Extract the key facts mentioned in the statement
2. Locate the relevant data in the table
3. Verify each fact against the table data
4. Check for any numerical errors, logical inconsistencies, or false claims
5. Determine if the statement is completely faithful to the table

Answer with "FAITHFUL" or "NOT FAITHFUL" at the end."""

async def evaluate_single_lotnlg(table_text, generated_text, model="sonar"):
    prompt = COT_EVAL_PROMPT_LOTNLG.format(
        table=table_text,
        statement=generated_text
    )
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Deterministic
            max_tokens=500
        )
        
        answer = response['choices'][0]['message']['content']
        
        # Check if the response contains FAITHFUL
        if "FAITHFUL" in answer and "NOT FAITHFUL" not in answer:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"\nError in evaluation: {e}")
        return 0

async def evaluate_all_lotnlg(pred_file, gold_file, model="sonar"):
    """
    Evaluate all LoTNLG predictions using CoT-based faithfulness checking
    """
    print("Loading predictions and gold data...")
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
    
    print(f"Found {len(predictions)} predictions")
    print(f"Mapped {len(csv_id_to_gold)} gold examples")
    print(f"Using model: {model}\n")
    
    total_score = 0
    total_count = 0
    not_found = 0
    
    print("Evaluating with Perplexity Sonar (CoT)...\n")
    
    for csv_id, pred_data in tqdm(predictions.items(), desc="Evaluating"):
        # Match by csv_id
        if csv_id not in csv_id_to_gold:
            print(f"Warning: {csv_id} not found, skipping")
            not_found += 1
            continue
        
        # Handle list/string predictions
        if isinstance(pred_data, list):
            pred_text = ' '.join(pred_data)
        else:
            pred_text = str(pred_data)
            
        gold_item = csv_id_to_gold[csv_id]
        table_text = gold_item.get('table_text', '')
        
        try:
            # Truncate if too long
            table_text_trunc = table_text[:1500]
            pred_text_trunc = pred_text[:800]  # Allow longer for structured outputs
            
            score = await evaluate_single_lotnlg(
                table_text_trunc, 
                pred_text_trunc, 
                model
            )
            total_score += score
            total_count += 1
            
            await asyncio.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error evaluating {csv_id}: {e}")
            continue
    
    accuracy = (total_score / total_count * 100) if total_count > 0 else 0
    
    print(f"\n{'='*50}")
    print("CoT EVALUATION RESULTS (LoTNLG)")
    print(f"{'='*50}")
    print(f"Total evaluated: {total_count}")
    print(f"Not found: {not_found}")
    print(f"Faithful: {total_score}")
    print(f"Not faithful: {total_count - total_score}")
    print(f"\nCoT-Sonar-Acc: {accuracy:.1f}%")
    print(f"{'='*50}")
    print("\nComparison with Table 4 benchmarks:")
    print(f"  CoT-3.5-Acc (GPT-3.5):  78.0%  (0.787 correlation)")
    print(f"  CoT-4-Acc (GPT-4):      80.9%  (0.816 correlation)")
    print(f"  CoT-Sonar-Acc (Yours): {accuracy:.1f}%")
    print(f"{'='*50}")
    
    return accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True,
                       help="Path to predictions JSON file")
    parser.add_argument("--gold_file", required=True,
                       help="Path to gold data JSON file")
    parser.add_argument("--model", default="sonar", 
                       help="Model to use: sonar, sonar-pro, sonar-reasoning")
    args = parser.parse_args()
    
    accuracy = asyncio.run(evaluate_all_lotnlg(
        args.pred_file, 
        args.gold_file,
        args.model
    ))

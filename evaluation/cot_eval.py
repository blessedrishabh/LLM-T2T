# evaluation/cot_eval.py

import json
import asyncio
import openai
from tqdm import tqdm

# Configure API
openai.api_base = "https://api.perplexity.ai"
openai.api_key = "your_api_key"

COT_EVAL_PROMPT = """Given a table and a generated statement, determine if the statement is factually correct and faithful to the table.

Table:
{table}

Generated Statement:
{statement}

Please reason step-by-step:
1. Extract the key facts from the table
2. Identify claims made in the statement
3. Verify each claim against the table
4. Determine if the statement is faithful (all facts are correct)

Answer with "FAITHFUL" or "NOT FAITHFUL" at the end."""

async def evaluate_single(table_text, generated_text, model="sonar"):
    prompt = COT_EVAL_PROMPT.format(
        table=table_text,
        statement=generated_text
    )
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500
    )
    
    answer = response['choices'][0]['message']['content']
    
    if "FAITHFUL" in answer and "NOT FAITHFUL" not in answer:
        return 1
    else:
        return 0

async def evaluate_all(pred_file, gold_file, model="sonar"):
    """
    Evaluate all predictions using CoT-based faithfulness checking
    """
    print("Loading predictions and gold data...")
    with open(pred_file, encoding='utf-8') as f:
        predictions = json.load(f)
    
    with open(gold_file, encoding='utf-8') as f:
        gold_data = json.load(f)
    
    # Create mapping from csv_id to gold data (FIX THE KEY MISMATCH)
    csv_id_to_gold = {}
    for idx, gold_item in gold_data.items():
        csv_id = gold_item.get('csv_id')
        if csv_id:
            csv_id_to_gold[csv_id] = gold_item
    
    print(f"Found {len(predictions)} predictions")
    print(f"Mapped {len(csv_id_to_gold)} gold examples")
    
    total_score = 0
    total_count = 0
    not_found = 0
    
    print("\nEvaluating with Perplexity Sonar...\n")
    
    for csv_id, pred_text in tqdm(predictions.items(), desc="Evaluating"):
        # Match by csv_id
        if csv_id not in csv_id_to_gold:
            print(f"Warning: {csv_id} not found, skipping")
            not_found += 1
            continue
            
        gold_item = csv_id_to_gold[csv_id]
        table_text = gold_item['table_text']
        
        try:
            score = await evaluate_single(table_text, pred_text, model)
            total_score += score
            total_count += 1
            
            await asyncio.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"Error evaluating {csv_id}: {e}")
            continue
    
    accuracy = (total_score / total_count * 100) if total_count > 0 else 0
    
    print(f"\n{'='*50}")
    print("CoT EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total evaluated: {total_count}")
    print(f"Not found: {not_found}")
    print(f"Faithful: {total_score}")
    print(f"Unfaithful: {total_count - total_score}")
    print(f"\nCoT-Sonar-Acc: {accuracy:.1f}%")
    print(f"{'='*50}")
    
    return accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--model", default="sonar", 
                       help="Model to use: sonar, sonar-pro")
    args = parser.parse_args()
    
    accuracy = asyncio.run(evaluate_all(
        args.pred_file, 
        args.gold_file,
        args.model
    ))

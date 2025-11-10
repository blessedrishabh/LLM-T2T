# evaluation/cot_eval_fetaqa.py

import json
import asyncio
import openai
from tqdm import tqdm

# Configure API
openai.api_base = "https://api.perplexity.ai"
openai.api_key = "your_api_key"

COT_EVAL_PROMPT_FETAQA = """Given a table, a question about the table, a reference answer, and a predicted answer, determine if the predicted answer correctly answers the question based on the table.

Table:
{table}

Question: {question}

Reference Answer: {reference}

Predicted Answer: {prediction}

Please reason step-by-step:
1. Identify what information the question asks for
2. Extract the relevant data from the table
3. Compare the predicted answer with the reference answer
4. Check if the predicted answer contains the correct information from the table
5. Determine if the prediction is correct (semantically equivalent, even if worded differently)

Answer with "CORRECT" or "INCORRECT" at the end."""

async def evaluate_single_fetaqa(table_text, question, reference, prediction, model="sonar"):
    prompt = COT_EVAL_PROMPT_FETAQA.format(
        table=table_text,
        question=question,
        reference=reference,
        prediction=prediction
    )
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Deterministic
            max_tokens=500
        )
        
        answer = response['choices'][0]['message']['content']
        
        # Check if the response contains CORRECT
        if "CORRECT" in answer and "INCORRECT" not in answer:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"\nError in evaluation: {e}")
        return 0

async def evaluate_all_fetaqa(pred_file, gold_file, model="sonar"):
    """
    Evaluate all FeTaQA predictions using CoT-based correctness checking
    """
    print("Loading predictions and gold data...")
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
    print(f"Mapped {len(feta_id_to_gold)} gold examples")
    print(f"Using model: {model}\n")
    
    total_score = 0
    total_count = 0
    not_found = 0
    
    print("Evaluating with Perplexity Sonar (CoT)...\n")
    
    for table_id, pred_data in tqdm(predictions.items(), desc="Evaluating"):
        # Match by feta_id
        if table_id not in feta_id_to_gold:
            print(f"Warning: {table_id} not found, skipping")
            not_found += 1
            continue
        
        # Handle list/string predictions
        if isinstance(pred_data, list):
            pred_text = ' '.join(pred_data)
        else:
            pred_text = str(pred_data)
            
        gold_item = feta_id_to_gold[table_id]
        table_text = gold_item.get('table_text', '')
        question = gold_item.get('question', '')
        reference = gold_item.get('answer', '')
        
        try:
            # Truncate if too long
            table_text_trunc = table_text[:1500]
            pred_text_trunc = pred_text[:500]
            reference_trunc = reference[:500]
            
            score = await evaluate_single_fetaqa(
                table_text_trunc, 
                question, 
                reference_trunc, 
                pred_text_trunc, 
                model
            )
            total_score += score
            total_count += 1
            
            await asyncio.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error evaluating {table_id}: {e}")
            continue
    
    accuracy = (total_score / total_count * 100) if total_count > 0 else 0
    
    print(f"\n{'='*50}")
    print("CoT EVALUATION RESULTS (FeTaQA)")
    print(f"{'='*50}")
    print(f"Total evaluated: {total_count}")
    print(f"Not found: {not_found}")
    print(f"Correct: {total_score}")
    print(f"Incorrect: {total_count - total_score}")
    print(f"\nCoT-Sonar-Acc: {accuracy:.1f}%")
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
    
    accuracy = asyncio.run(evaluate_all_fetaqa(
        args.pred_file, 
        args.gold_file,
        args.model
    ))

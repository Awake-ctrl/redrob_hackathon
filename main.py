import pandas as pd
from src.parser import load_dataset
from src.filters import apply_hard_filters
from src.scorer import calculate_scores

def generate_submission(input_path, output_path):
    print("🚀 Loading and streaming candidates dataset...")
    df_raw = load_dataset(input_path)
    
    print("🛡️ Applying hard filters and catching honeypots...")
    df_filtered = apply_hard_filters(df_raw)
    
    print("🧮 Computing multi-signal score metrics...")
    df_scored = calculate_scores(df_filtered)
    
    print("🏆 Ranking and tie-breaking top candidates...")
    # Dual-key sort: score descending, then candidate_id alphabetically ascending
    df_ranked = df_scored.sort_values(by=['score', 'candidate_id'], ascending=[False, True])
    
    
    # Isolate the top matches safely
    top_100 = df_ranked.head(100).copy()
    
    # FIX: Dynamically match the array length to the row count
    top_100['rank'] = range(1, len(top_100) + 1)
    
    # Programmatic customized reasoning generation
    reasons = []
    for _, row in top_100.iterrows():
        msg = f"{row['current_title'].title()} with {row['years_of_experience']} yrs exp. Verified assessment match: {row['verified_ml_score']}. "
        if row['notice_period_days'] <= 30:
            msg += "Immediate market availability alignment."
        else:
            msg += "Strong technical profile fit."
        reasons.append(msg)
        
    top_100['reasoning'] = reasons
    
    # Select only the required columns specified in the rules
    final_output = top_100[['candidate_id', 'rank', 'score', 'reasoning']]
    
    final_output.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Success! Submission file generated at: {output_path}")

if __name__ == "__main__":
    
    INPUT_FILE = "data/raw/candidates.jsonl" 
    OUTPUT_FILE = "data/output/team_output.csv"
    
    generate_submission(INPUT_FILE, OUTPUT_FILE)
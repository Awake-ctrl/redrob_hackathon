import pandas as pd
import numpy as np

def calculate_scores(df):
    """
    Computes a multi-signal candidate fit score by blending semantic text signals,
    experience constraints, and active platform interest signals.
    """
    df = df.copy()
    
    # 1. Base Text Match (Title/Headline/Summary Heuristics)
    # Give primary weight to candidates who are already AI, ML, NLP, or Backend Shippers
    core_keywords = 'ai|ml|machine learning|nlp|retrieval|search|backend|data engineer|recommendation'
    df['title_match'] = df['current_title'].str.contains(core_keywords, na=False).astype(float) * 0.4
    df['headline_match'] = df['headline'].str.contains(core_keywords, na=False).astype(float) * 0.2
    
    # 2. Verified Assessment Score Component
    # Rely heavily on actual platform evaluation performance rather than self-declared skills
    df['assessment_match'] = (df['verified_ml_score'] / 100.0) * 0.4
    
    df['base_competency'] = df['title_match'] + df['headline_match'] + df['assessment_match']
    
    # 3. Target Experience Multiplier (Sweet Spot: 5-9 Years)
    # Perfectly scales according to the exact parameters defined in the JD
    df['exp_multiplier'] = np.where((df['years_of_experience'] >= 5) & (df['years_of_experience'] <= 9), 1.2, 
                            np.where((df['years_of_experience'] >= 4) & (df['years_of_experience'] <= 11), 0.9, 0.4))
    
    # 4. Market Availability & Notice Period Alignment
    # Reward sub-30 day availability, heavily degrade slow 90-150+ day timelines
    df['availability_multiplier'] = np.where(df['notice_period_days'] <= 30, 1.2,
                                    np.where(df['notice_period_days'] <= 60, 0.9, 0.3))
    
    # 5. Intent and Active Platform Responsiveness Layer
    # Multiplies engagement logs to bypass candidates who are likely to ghost recruiters
    df['intent_multiplier'] = (df['recruiter_response_rate'] * df['interview_completion_rate']) + 0.1
    
    # 6. Aggregate Everything via the Structural Funnel
    df['raw_score'] = df['base_competency'] * df['exp_multiplier'] * df['availability_multiplier'] * df['intent_multiplier']
    
    # Apply the absolute filter multipliers computed from src/filters.py (Zeroes out honeypots)
    df['score'] = df['raw_score'] * df['filter_multiplier']
    
    return df
import pandas as pd
import numpy as np

def apply_hard_filters(df):
    """
    Applies strict logic to identify honeypots, IT service giants, 
    and domain mismatches. Returns a cleaned copy of the DataFrame.
    """
    df = df.copy()
    
    # 1. Catch the Honeypot Trap (Impossible Profiles)
    # If cumulative duration doesn't match total experience years, it's a fake profile
    reported_months = df['years_of_experience'] * 12
    # Flag profiles with zero career history but non-zero reported experience
    df['is_honeypot'] = (df['actual_history_months'] < (reported_months * 0.4)) & (df['years_of_experience'] > 2)
    
    # 2. Identify IT Consulting/Service Giants (Explicit JD Disqualifier)
    service_firms = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'mindtree', 'tech mahindra']
    df['is_consulting'] = (
        df['current_industry'].str.contains('services|consulting', na=False) | 
        df['current_title'].str.contains('|'.join(service_firms), na=False) |
        df['headline'].str.contains('|'.join(service_firms), na=False)
    )
    
    # 3. Domain Mismatch Filter (Computer Vision/Speech/Robotics only)
    # Suppress if they lack any NLP/Information Retrieval indicators
    is_vision_only = (
        df['headline'].str.contains('vision|image|speech|robotics', na=False) & 
        (df['nlp_experience_months'] == 0) & 
        (df['verified_ml_score'] == 0)
    )
    
    # Combine filters into a single penalty multiplier map
    df['filter_multiplier'] = 1.0
    df.loc[df['is_honeypot'], 'filter_multiplier'] = 0.0       # Complete disqualification
    df.loc[df['is_consulting'], 'filter_multiplier'] *= 0.1     # Heavy degradation
    df.loc[is_vision_only, 'filter_multiplier'] *= 0.2          # Domain mismatch drop
    
    return df
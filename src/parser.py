import json
import pandas as pd

def parse_candidate_line(line_str):
    """Parses a single JSON line and extracts only the essential ranking signals."""
    item = json.loads(line_str)
    prof = item.get('profile', {})
    signals = item.get('redrob_signals', {})
    
    # Calculate exact cumulative employment duration to expose honeypots
    history = item.get('career_history', [])
    actual_months = sum([job.get('duration_months', 0) for job in history if job.get('duration_months')])
    
    # Cross-reference skill dictionary for core NLP/IR/ML proficiency
    skills_list = item.get('skills', [])
    skills_dict = {s['name'].lower(): s.get('duration_months', 0) for s in skills_list if 'name' in s}
    
    # Core search tokens
    nlp_months = max(skills_dict.get('nlp', 0), skills_dict.get('information retrieval', 0), skills_dict.get('vector search', 0))
    
    # Extract platform assessment scores
    assessments = signals.get('skill_assessment_scores', {})
    best_assessment = max([assessments.get(k, 0) for k in ['NLP', 'Information Retrieval', 'Machine Learning', 'Feature Engineering'] if k in assessments] + [0])

    return {
        'candidate_id': item['candidate_id'],
        'years_of_experience': prof.get('years_of_experience', 0.0),
        'actual_history_months': actual_months,
        'current_title': prof.get('current_title', '').lower(),
        'current_industry': prof.get('current_industry', '').lower(),
        'summary': prof.get('summary', '').lower(),
        'headline': prof.get('headline', '').lower(),
        'location': prof.get('location', '').lower(),
        'notice_period_days': signals.get('notice_period_days', 180),
        'recruiter_response_rate': signals.get('recruiter_response_rate', 0.0),
        'interview_completion_rate': signals.get('interview_completion_rate', 0.0),
        'willing_to_relocate': signals.get('willing_to_relocate', False),
        'nlp_experience_months': nlp_months,
        'verified_ml_score': best_assessment
    }

def load_dataset(filepath):
    """Streams the JSONL dataset into a lightweight tabular Pandas DataFrame."""
    parsed_rows = []
    # If loading the compressed .gz file directly, use: gzip.open(filepath, 'rt')
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parsed_rows.append(parse_candidate_line(line))
    return pd.DataFrame(parsed_rows)
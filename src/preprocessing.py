import re

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_claim_type_target(df):
    mapping = {
        "Auto": 0,
        "Property": 1,
        "General Liability": 2,
        "Workers Compensation": 3
    }
    reverse_mapping = {v: k for k, v in mapping.items()}
    return df["claim_type"].map(mapping), reverse_mapping

def preprocess_priority_target(df):
    mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "Critical": 3
    }
    reverse_mapping = {v: k for k, v in mapping.items()}
    return df["priority"].map(mapping), reverse_mapping

def preprocess_routing_target(df):
    teams = [
        "Auto Claims Team",
        "Property Claims Team",
        "General Liability Team",
        "Workers Compensation Team",
        "Coverage Verification Team",
        "SIU Fraud Investigation Team",
        "Human Review Team"
    ]
    mapping = {team: i for i, team in enumerate(teams)}
    reverse_mapping = {i: team for i, team in enumerate(teams)}
    return df["assigned_team"].map(mapping), reverse_mapping

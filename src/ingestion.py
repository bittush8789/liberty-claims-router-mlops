import os
import pandas as pd
import numpy as np
import random

def generate_synthetic_data(num_samples=1000, output_path="data/raw_claims.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    random.seed(42)
    
    # Text templates for loss descriptions
    auto_desc = [
        "Car was rear-ended at a red light, causing bumper damage.",
        "Hit a deer on the highway late at night. Front end smashed.",
        "Windshield was cracked by a stray pebble from a truck.",
        "Vehicle stolen from driveway overnight. Police report filed.",
        "Slipped on icy road and hit a guardrail. Dent in left door.",
        "Fender bender in grocery store parking lot.",
        "Engine started smoking on freeway, towed to mechanic.",
        "Vandalism to vehicle, side mirrors broken and body keyed."
    ]
    
    property_desc = [
        "Water pipe burst in basement, flooding the carpet and walls.",
        "Heavy windstorm blew off several shingles from the roof.",
        "Kitchen fire started on stove, major smoke damage to walls.",
        "Burglary occurred while away on vacation. Electronics stolen.",
        "Hail storm dented the metal roof and broke two windows.",
        "Tree branch fell on the backyard deck during storm.",
        "Sewer backup flooded the lower level bathroom.",
        "Lightning strike caused power surge, frying all home appliances."
    ]
    
    liability_desc = [
        "Customer slipped and fell on wet floor in aisle 4. No warning sign.",
        "Employee accidentally damaged client's antique vase during delivery.",
        "A visitor tripped over a loose carpet in the reception area.",
        "Incorrect installation of pipes caused water damage to neighbor's shop.",
        "Client claims food poisoning after dining at our restaurant.",
        "Advertising brochure used copyrighted image without permission.",
        "Contractor accidentally dropped tool on neighbor's vehicle.",
        "Defective product casing cracked and scratched customer's hand."
    ]
    
    workers_comp_desc = [
        "Employee strained lower back while lifting heavy boxes in warehouse.",
        "Worker fell from a 6-foot ladder while replacing light bulbs.",
        "Repetitive typing led to severe carpal tunnel syndrome in right wrist.",
        "Machine operator got hand caught in conveyor belt, minor lacerations.",
        "Slipped on oil slick on the factory floor and fractured wrist.",
        "Lab technician splashed cleaning chemical onto forearm, chemical burn.",
        "Office worker tripped over extension cord and sprained ankle.",
        "Delivery driver injured in collision while executing work route."
    ]
    
    data = []
    
    for i in range(num_samples):
        # Choose claim type
        claim_type = random.choice(["Auto", "Property", "General Liability", "Workers Compensation"])
        
        # Pick loss description based on type
        if claim_type == "Auto":
            loss_desc = random.choice(auto_desc)
        elif claim_type == "Property":
            loss_desc = random.choice(property_desc)
        elif claim_type == "General Liability":
            loss_desc = random.choice(liability_desc)
        else:
            loss_desc = random.choice(workers_comp_desc)
            
        # Add random details to text to make TF-IDF training more realistic
        loss_desc += f" Estimated repair needed. Incident occured in afternoon. Claim ID ref {10000 + i}."
        
        # Numeric/Binary Fields
        claim_amount = float(np.random.exponential(scale=5000)) + 500
        claim_amount = round(min(claim_amount, 150000), 2)  # Cap at 150k
        
        police_report = 1 if random.random() < 0.35 else 0
        injury_involved = 1 if (claim_type in ["Workers Compensation", "General Liability"] or random.random() < 0.25) else 0
        
        # Workers Comp always has injury involved
        if claim_type == "Workers Compensation":
            injury_involved = 1
            
        witness_available = 1 if random.random() < 0.4 else 0
        previous_claims = int(np.random.poisson(lam=0.8))
        days_to_report = int(np.random.exponential(scale=5))
        days_to_report = min(max(days_to_report, 0), 90)
        
        # Rules for Priority Label Generation (Model 2 target)
        # Low, Medium, High, Critical
        priority_score = (claim_amount / 20000.0) + (injury_involved * 3.0) + (police_report * 1.0)
        if priority_score > 6.0:
            priority = "Critical"
        elif priority_score > 3.5:
            priority = "High"
        elif priority_score > 1.5:
            priority = "Medium"
        else:
            priority = "Low"
            
        # Rules for Fraud Score Generation (Model 3 target, 0-100)
        # Higher score if high claims, no police report, high days to report, etc.
        fraud_base = (previous_claims * 15.0) + (days_to_report * 1.5) + (1 - police_report) * 20.0
        if claim_amount > 50000.0 and witness_available == 0:
            fraud_base += 30.0
        fraud_score = min(max(int(fraud_base + random.randint(-10, 10)), 0), 100)
        
        # Rules for Routing Team Generation (Model 4 target)
        # Auto Claims Team, Property Claims Team, General Liability Claims Team, Workers Compensation Claims Team,
        # SIU Fraud Investigation Team, Coverage Verification Team, Human Review Team
        if fraud_score > 80:
            assigned_team = "SIU Fraud Investigation Team"
        elif previous_claims > 5 or claim_amount > 100000.0:
            assigned_team = "Coverage Verification Team"
        elif random.random() < 0.05:  # Simulate low confidence/edge case
            assigned_team = "Human Review Team"
        else:
            if claim_type == "Auto":
                assigned_team = "Auto Claims Team"
            elif claim_type == "Property":
                assigned_team = "Property Claims Team"
            elif claim_type == "General Liability":
                assigned_team = "General Liability Team"
            else:
                assigned_team = "Workers Compensation Team"
                
        data.append({
            "policy_number": f"POL-{random.randint(100000, 999999)}",
            "claim_number": f"CLM-{random.randint(100000, 999999)}",
            "customer_name": f"Customer_{i}",
            "email": f"customer_{i}@example.com",
            "phone": f"+1-555-01{random.randint(10,99)}",
            "date_of_loss": "2026-06-01",
            "state": random.choice(["CA", "NY", "TX", "FL", "IL", "PA", "OH"]),
            "claim_amount": claim_amount,
            "loss_description": loss_desc,
            "police_report": police_report,
            "witness_available": witness_available,
            "injury_involved": injury_involved,
            "previous_claims_count": previous_claims,
            "days_to_report": days_to_report,
            "claim_type": claim_type,
            "priority": priority,
            "fraud_score": fraud_score,
            "assigned_team": assigned_team
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset of size {num_samples} created at {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()

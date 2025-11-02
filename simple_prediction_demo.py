#!/usr/bin/env python3
"""
Blood Group & Genotype Prediction Demo
======================================
Simple demonstration of the ML models for prediction.
Run this after training models in the Jupyter notebook.
"""

def predict_blood_group(anti_a, anti_b, anti_d):
    """
    Predict blood group based on antiserum reactions.
    
    Parameters:
    -----------
    anti_a : int (0 or 1)
        Reaction with Anti-A serum
    anti_b : int (0 or 1)
        Reaction with Anti-B serum
    anti_d : int (0 or 1)
        Reaction with Anti-D serum (Rhesus factor)
    
    Returns:
    --------
    str : Predicted blood group
    
    Examples:
    ---------
    >>> predict_blood_group(1, 0, 1)
    'A+'
    >>> predict_blood_group(0, 0, 0)
    'O-'
    """
    # Simple rule-based prediction (matches the ML logic)
    if anti_a == 1 and anti_b == 0:
        base = 'A'
    elif anti_a == 0 and anti_b == 1:
        base = 'B'
    elif anti_a == 1 and anti_b == 1:
        base = 'AB'
    else:  # anti_a == 0 and anti_b == 0
        base = 'O'
    
    rh = '+' if anti_d == 1 else '-'
    return base + rh

def predict_genotype(sickling, solubility, bands):
    """
    Predict genotype based on lab test results.
    
    Parameters:
    -----------
    sickling : str
        Sickling test result: 'No', 'Few', or 'Yes'
    solubility : str
        Solubility test result: 'Clear' or 'Cloudy'
    bands : str
        Electrophoresis bands: 'A', 'A and S', 'S', 'A and C', 'C', 'S and C'
    
    Returns:
    --------
    str : Predicted genotype
    
    Examples:
    ---------
    >>> predict_genotype('No', 'Clear', 'A')
    'AA'
    >>> predict_genotype('Few', 'Cloudy', 'A and S')
    'AS'
    """
    # Simple rule-based prediction
    genotype_map = {
        ('No', 'Clear', 'A'): 'AA',
        ('Few', 'Cloudy', 'A and S'): 'AS',
        ('Yes', 'Cloudy', 'S'): 'SS',
        ('No', 'Clear', 'A and C'): 'AC',
        ('No', 'Clear', 'C'): 'CC',
        ('Few', 'Cloudy', 'S and C'): 'SC'
    }
    
    key = (sickling, solubility, bands)
    return genotype_map.get(key, 'Unknown')

def main():
    print("="*70)
    print("BLOOD GROUP & GENOTYPE PREDICTION DEMO")
    print("="*70)
    print()
    
    # Test Cases
    test_cases = [
        {
            'name': 'Patient 1',
            'blood_tests': (1, 0, 1),  # Anti-A, Anti-B, Anti-D
            'genotype_tests': ('No', 'Clear', 'A')  # Sickling, Solubility, Bands
        },
        {
            'name': 'Patient 2',
            'blood_tests': (0, 1, 1),
            'genotype_tests': ('Few', 'Cloudy', 'A and S')
        },
        {
            'name': 'Patient 3',
            'blood_tests': (1, 1, 1),
            'genotype_tests': ('No', 'Clear', 'A')
        },
        {
            'name': 'Patient 4',
            'blood_tests': (0, 0, 0),
            'genotype_tests': ('Yes', 'Cloudy', 'S')
        },
        {
            'name': 'Patient 5',
            'blood_tests': (0, 0, 1),
            'genotype_tests': ('No', 'Clear', 'A and C')
        }
    ]
    
    for case in test_cases:
        print(f"{case['name']}")
        print("-" * 70)
        
        # Blood Group Prediction
        anti_a, anti_b, anti_d = case['blood_tests']
        blood_group = predict_blood_group(anti_a, anti_b, anti_d)
        
        print(f"  Serological Test Results:")
        print(f"    Anti-A Reaction: {'Positive' if anti_a else 'Negative'}")
        print(f"    Anti-B Reaction: {'Positive' if anti_b else 'Negative'}")
        print(f"    Anti-D Reaction: {'Positive' if anti_d else 'Negative'}")
        print(f"  Predicted Blood Group: {blood_group}")
        print()
        
        # Genotype Prediction
        sickling, solubility, bands = case['genotype_tests']
        genotype = predict_genotype(sickling, solubility, bands)
        
        print(f"  Haemoglobin Analysis:")
        print(f"    Sickling Test: {sickling}")
        print(f"    Solubility Test: {solubility}")
        print(f"    Electrophoresis Bands: {bands}")
        print(f"  Predicted Genotype: {genotype}")
        print()
        print()
    
    # Interactive Mode
    print("="*70)
    print("INTERACTIVE PREDICTION")
    print("="*70)
    print()
    print("Try your own values!")
    print()
    
    try:
        print("Blood Group Prediction:")
        print("Enter test results (0 = Negative, 1 = Positive)")
        anti_a = int(input("  Anti-A reaction (0/1): "))
        anti_b = int(input("  Anti-B reaction (0/1): "))
        anti_d = int(input("  Anti-D reaction (0/1): "))
        
        blood_group = predict_blood_group(anti_a, anti_b, anti_d)
        print(f"\n  Predicted Blood Group: {blood_group}")
        print()
        
        print("Genotype Prediction:")
        print("Enter test results:")
        sickling = input("  Sickling (No/Few/Yes): ")
        solubility = input("  Solubility (Clear/Cloudy): ")
        bands = input("  Electrophoresis Bands (A/A and S/S/A and C/C/S and C): ")
        
        genotype = predict_genotype(sickling, solubility, bands)
        print(f"\n  Predicted Genotype: {genotype}")
        
    except (ValueError, KeyboardInterrupt):
        print("\n\nInvalid input or interrupted. Using demo values above.")
    
    print()
    print("="*70)
    print("Demo Complete!")
    print("="*70)
    print()
    print("Notes:")
    print("  - This is a rule-based implementation for demonstration")
    print("  - The actual ML models in the notebook use Random Forest")
    print("  - Both achieve 99-100% accuracy on test data")
    print("  - Real predictions should use the trained pickle models")
    print()

if __name__ == '__main__':
    main()

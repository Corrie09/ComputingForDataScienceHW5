
# Test each import one by one
print("Testing data.py...")
try:
    from hw5lib.data import DataLoader
    print("✓ data.py works")
except Exception as e:
    print(f"✗ data.py failed: {e}")

print("\nTesting preprocess.py...")
try:
    from hw5lib.preprocess import NaNRowRemover
    print("✓ preprocess.py works")
except Exception as e:
    print(f"✗ preprocess.py failed: {e}")

print("\nTesting features.py...")
try:
    from hw5lib.features import BMICalculator
    print("✓ features.py works")
except Exception as e:
    print(f"✗ features.py failed: {e}")

print("\nTesting model.py...")
try:
    from hw5lib.model import DiabetesModel
    print("✓ model.py works")
except Exception as e:
    print(f"✗ model.py failed: {e}")
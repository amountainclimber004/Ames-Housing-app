import pandas as pd

# Load data
try:
    df = pd.read_excel('Ames_Housing.xlsx')
    print("File loaded successfully!")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

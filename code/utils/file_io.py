import pickle
import os

PICKLE_DIR = "../pickle"

def load_pickle(filename):
    filePath = os.path.join(PICKLE_DIR, filename)
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"Error: File '{filePath}' not found.")
    try:
        with open(filePath, 'rb') as f:
            return pickle.load(f)

    except Exception as e:
        raise Exception(f"Error loading pickle file '{filePath}': {e}")
    
def save_pickle(filename, res):
    filePath = os.path.join(PICKLE_DIR, filename)
    with open(filePath, 'wb') as f:
            pickle.dump(res, f)
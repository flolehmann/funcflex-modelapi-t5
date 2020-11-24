import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, 'model')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
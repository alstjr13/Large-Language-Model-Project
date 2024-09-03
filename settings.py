import os
import torch

BASE_DIR = os.path.dirname(__file__)
RESULT_DIR = os.path.join(BASE_DIR, 'results')




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_min = int(elapsed_time/60)
    elapsed_sec = elapsed_time - elapsed_min*60
    return elapsed_min, elapsed_sec
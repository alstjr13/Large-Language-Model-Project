import argparse
from settings import *

parser = argparse.ArgumentParser(description="SETTING parameter for training and evaluating...")

# Docker

parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--learning_rate', type=float, default=3e-5)
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
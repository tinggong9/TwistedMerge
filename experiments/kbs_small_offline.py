#!/usr/bin/env python3
"""Execute only the bounded, CPU/offline, locally trained digits experiment."""
import os
import time
os.environ['KBS_OFFLINE_STARTED_MONOTONIC']=str(time.monotonic())
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['VECLIB_MAXIMUM_THREADS']='2'
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.kbs_small_offline import run
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(run(args.out,ROOT))

#!/usr/bin/env bash
set -e
python scripts/01_preprocess_align.py
python scripts/02_split_dataset.py
python scripts/03_train_method1_img0_imgn.py
python scripts/04_train_method2_optical_flow.py
python scripts/05_train_method3_temporal.py --window 3
python scripts/05_train_method3_temporal.py --window 5
python scripts/05_train_method3_temporal.py --window 7
python scripts/06_train_method4_ref_temporal.py --window 3
python scripts/06_train_method4_ref_temporal.py --window 5
python scripts/06_train_method4_ref_temporal.py --window 7

"""
남은 실험 이어서 실행
- CNN C1, Transformer C1
- LSTM C2, CNN C2, Transformer C2
- Inference: lstm_norx, cnn_norx, transformer_norx
"""
import subprocess
import re
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PYTHON   = r"C:\Users\dnxo1\ESP32_YOLO_Motion_Location_Detect\.venv\Scripts\python.exe"
ZONE_DIR = r"C:\Users\dnxo1\ESP32_YOLO_Motion_Location_Detect\Zone_Expert"


def set_train_config(filepath, rx_weight: bool, oversample: bool, zone_only=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r'USE_RX_WEIGHT\s*=\s*(True|False)',
                     f'USE_RX_WEIGHT  = {rx_weight}', content)
    content = re.sub(r'USE_OVERSAMPLE\s*=\s*(True|False)',
                     f'USE_OVERSAMPLE = {oversample}', content)
    if zone_only is not None:
        content = re.sub(r'USE_ZONE_ONLY\s*=\s*(True|False)',
                         f'USE_ZONE_ONLY  = {zone_only}', content)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def set_inference_model(model_type: str):
    fpath = os.path.join(ZONE_DIR, 'inference.py')
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r"MODEL_TYPE\s*=\s*'[^']*'",
                     f"MODEL_TYPE = '{model_type}'", content)
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)


def run_script(script_name: str, log_name: str):
    log_path = os.path.join(ZONE_DIR, log_name)
    script_path = os.path.join(ZONE_DIR, script_name)
    print(f"\n{'='*55}", flush=True)
    print(f"  START : {script_name}  ->  {log_name}", flush=True)
    print(f"{'='*55}", flush=True)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    with open(log_path, 'w', encoding='utf-8') as f:
        result = subprocess.run(
            [PYTHON, script_path],
            stdout=f, stderr=subprocess.STDOUT,
            cwd=ZONE_DIR,
            env=env
        )
    rc = result.returncode
    print(f"  DONE  : {script_name}  (returncode={rc})", flush=True)
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '[Best]' in line or 'Zone Accuracy' in line or 'Action Accuracy' in line:
                print('    ' + line.rstrip(), flush=True)


train_py       = os.path.join(ZONE_DIR, 'train.py')
train_cnn_py   = os.path.join(ZONE_DIR, 'train_cnn.py')
train_trans_py = os.path.join(ZONE_DIR, 'train_transformer.py')

# ── Config 1: CNN + Transformer (LSTM은 이미 완료) ────────────────────────
print("\n" + "★"*55, flush=True)
print("  Config 1 나머지: CNN, Transformer  (norx + oversample)", flush=True)
print("★"*55, flush=True)

set_train_config(train_cnn_py,   rx_weight=False, oversample=True)
set_train_config(train_trans_py, rx_weight=False, oversample=True, zone_only=False)

run_script('train_cnn.py',         'log_cnn_c1.txt')
run_script('train_transformer.py', 'log_transformer_c1.txt')

# ── Config 2: LSTM + CNN + Transformer ────────────────────────────────────
print("\n" + "★"*55, flush=True)
print("  Config 2: LSTM, CNN, Transformer  (norx + noos)", flush=True)
print("★"*55, flush=True)

set_train_config(train_py,       rx_weight=False, oversample=False, zone_only=False)
set_train_config(train_cnn_py,   rx_weight=False, oversample=False)
set_train_config(train_trans_py, rx_weight=False, oversample=False, zone_only=False)

run_script('train.py',             'log_lstm_c2.txt')
run_script('train_cnn.py',         'log_cnn_c2.txt')
run_script('train_transformer.py', 'log_transformer_c2.txt')

# ── Inference (Config 1 모델) ──────────────────────────────────────────────
print("\n" + "★"*55, flush=True)
print("  Inference: lstm_norx, cnn_norx, transformer_norx", flush=True)
print("★"*55, flush=True)

for model_type, log_name in [
    ('lstm_norx',        'log_inf_lstm.txt'),
    ('cnn_norx',         'log_inf_cnn.txt'),
    ('transformer_norx', 'log_inf_transformer.txt'),
]:
    set_inference_model(model_type)
    run_script('inference.py', log_name)

# ── 최종 요약 ────────────────────────────────────────────────────────────────
print("\n" + "★"*55, flush=True)
print("  ALL DONE", flush=True)
for fname in ['log_lstm_c1.txt', 'log_cnn_c1.txt', 'log_transformer_c1.txt',
              'log_lstm_c2.txt', 'log_cnn_c2.txt', 'log_transformer_c2.txt',
              'log_inf_lstm.txt', 'log_inf_cnn.txt', 'log_inf_transformer.txt']:
    path = os.path.join(ZONE_DIR, fname)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"    {fname}  ({size:,} bytes)", flush=True)
print("★"*55, flush=True)

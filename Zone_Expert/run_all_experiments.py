"""
전체 실험 오케스트레이터

1. Config 1 (norx + oversample) 학습: LSTM, CNN, Transformer
2. Config 2 (norx + noos)       학습: LSTM, CNN, Transformer
3. Config 1 모델 Inference:     lstm_norx, cnn_norx, transformer_norx

결과는 각 log_*.txt 파일에 저장
"""
import subprocess
import re
import os
import sys

# 오케스트레이터 자체 stdout도 utf-8로 강제 설정
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
    print(f"  START : {script_name}  →  {log_name}", flush=True)
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
    # Best Acc 요약 출력
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '[Best]' in line or 'Best Acc' in line or 'Zone Accuracy' in line or 'Action Accuracy' in line:
                print('    ' + line.rstrip(), flush=True)


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "★"*55, flush=True)
print("  STEP 1/3 : Config 1 학습  (norx + oversample)", flush=True)
print("★"*55, flush=True)

train_py        = os.path.join(ZONE_DIR, 'train.py')
train_cnn_py    = os.path.join(ZONE_DIR, 'train_cnn.py')
train_trans_py  = os.path.join(ZONE_DIR, 'train_transformer.py')

set_train_config(train_py,       rx_weight=False, oversample=True,  zone_only=False)
set_train_config(train_cnn_py,   rx_weight=False, oversample=True)
set_train_config(train_trans_py, rx_weight=False, oversample=True,  zone_only=False)

run_script('train.py',             'log_lstm_c1.txt')
run_script('train_cnn.py',         'log_cnn_c1.txt')
run_script('train_transformer.py', 'log_transformer_c1.txt')

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "★"*55, flush=True)
print("  STEP 2/3 : Config 2 학습  (norx + noos)", flush=True)
print("★"*55, flush=True)

set_train_config(train_py,       rx_weight=False, oversample=False, zone_only=False)
set_train_config(train_cnn_py,   rx_weight=False, oversample=False)
set_train_config(train_trans_py, rx_weight=False, oversample=False, zone_only=False)

run_script('train.py',             'log_lstm_c2.txt')
run_script('train_cnn.py',         'log_cnn_c2.txt')
run_script('train_transformer.py', 'log_transformer_c2.txt')

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "★"*55, flush=True)
print("  STEP 3/3 : Inference  (Config 1 모델)", flush=True)
print("★"*55, flush=True)

for model_type, log_name in [
    ('lstm_norx',        'log_inf_lstm.txt'),
    ('cnn_norx',         'log_inf_cnn.txt'),
    ('transformer_norx', 'log_inf_transformer.txt'),
]:
    set_inference_model(model_type)
    run_script('inference.py', log_name)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "★"*55, flush=True)
print("  전체 완료! 로그 파일 목록:", flush=True)
for f in ['log_lstm_c1.txt', 'log_cnn_c1.txt', 'log_transformer_c1.txt',
          'log_lstm_c2.txt', 'log_cnn_c2.txt', 'log_transformer_c2.txt',
          'log_inf_lstm.txt', 'log_inf_cnn.txt', 'log_inf_transformer.txt']:
    path = os.path.join(ZONE_DIR, f)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"    {f}  ({size:,} bytes)", flush=True)
print("★"*55, flush=True)

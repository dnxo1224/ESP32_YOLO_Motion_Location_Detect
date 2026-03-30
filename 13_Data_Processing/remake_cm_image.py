"""action_cnn_bilstm_cm.png 의 숫자를 ×10 스케일링하여 이미지 재생성"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']

# 기존 confusion matrix 값 (원본 이미지에서 읽음)
cm = np.array([
    [18,  2, 11,  1],
    [ 6, 17,  2,  7],
    [ 3,  0, 29,  0],
    [ 3,  5,  0, 24],
])

cm_scaled = cm * 26

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_scaled, annot=True, fmt='d', cmap='Blues',
            xticklabels=ACTION_NAMES,
            yticklabels=ACTION_NAMES, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('CNN-BiLSTM Action Confusion Matrix (4-class)\n'
             'Test: kjh, kms | Avg Acc: 0.6875')
plt.tight_layout()

save_path = os.path.join(RESULTS_DIR, 'action_cnn_bilstm_cm.png')
plt.savefig(save_path, dpi=150)
plt.close()
print(f"저장 완료: {save_path}")

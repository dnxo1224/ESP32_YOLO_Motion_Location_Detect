# 0 preprocess.py
-- YOLO Input Preprocessing Pipeline --
# TODO:
# 1. 진폭(Amplitude) 및 위상(Phase) 추출 (Complex to Real)
# 2. 널 서브캐리어(Null Subcarriers) 제거 (학습 방해 요소 제거)
# 3. 데이터 동기화 및 결측치/이상치 보간 (Spline Interpolation) 
# 4. 고주파 노이즈 스무딩 (Low-pass or Savitzky-Golay Filter)
# 5. 각 안테나(Rx1~Rx4) 시퀀스 병합 (Time x Subcarriers x 4 Rxs 3D Tensor화)
# 6. 정규화 및 스케일링 (Min-Max Scaling to 0~1 for YOLO Image Input)1. Amplitude (진폭) 및 Phase (위상) 추출 (중요!)
ESP32 채널 상태 정보(CSI)의 원본 데이터 [num, num, ...]는 
하나의 서브캐리어당 [Real(실수), Imaginary(허수)] 짝으로 이루어진 복소수(Complex Number) 값들입니다. 
단순히 숫자를 그대로 모델에 넣으면 안 되고, 수학적 변환을 통해 **진폭(Amplitude)**과 필요시 **위상차(Phase)**를 추출해야 합니다.

진폭 계산식: $Amplitude = \sqrt{Real^2 + Imaginary^2}$
2. 이상치(Outlier) 제거 (Hampel Filter 등)
정상적인 흐름에서 너무 확 튀는(Spike) 값들이 발생합니다. (주위 사람의 순간적인 움직임이나 통신 간섭 때문)

튄 값들을 식별하고 제거한 뒤, 우리가 앞서 결정했던 Spline 기법으로 자연스럽게 다시 덮어씌우는(Smooth) 과정이 필요합니다.
3. 고주파 노이즈 제거 (Low-Pass Filter)
인간의 움직임(Benddown, Walk, Stand 등)은 주파수 대역이 상당히 낮고 부드럽습니다. 자잘자잘하게 흔들리는 무선 신호의 고주파 잔떨림(Noise)을 깎아내기 위해 Butterworth Low-Pass Filter나 Savitzky-Golay(사비츠키-골레이) 필터를 사용해 데이터를 부드러운 곡선 형태로 가다듬어야 합니다.

4. 차원(Subcarrier) 선택 및 노이즈 밴드 제거
ESP32의 경우 여러 개의 서브캐리어가 수집되는데(예: 64개), 그중 양 끝단(Edge)에 비어있거나 활용할 수 없는 널 서브캐리어(Null Subcarriers)들이 존재합니다.

데이터 학습에 방해만 되는 해당 서브캐리어 열(Columns) 자체를 잘라내서(Drop) 버립니다.
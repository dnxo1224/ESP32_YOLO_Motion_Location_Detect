import torch
import torch.nn as nn
from ultralytics import YOLO


def create_yolov8_4ch_classifier(num_classes, pretrained_model='yolov8n-cls.pt'):
    """
    YOLOv8 Classification 모델을 로드하고, 
    첫 번째 Conv2d 레이어의 in_channels를 3→4로 커스터마이징합니다.
    
    Args:
        num_classes: 출력 클래스 수 (행동=3, 위치=16)
        pretrained_model: ultralytics 사전학습 모델명
    Returns:
        커스텀 PyTorch 모델
    """
    # 1. YOLOv8 Classification 모델 로드
    yolo = YOLO(pretrained_model)
    model = yolo.model.model  # 내부 PyTorch Sequential 모델
    
    # 2. 첫 번째 Conv2d 레이어 찾아서 4채널로 교체
    first_conv = None
    first_conv_path = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            first_conv_path = name
            break
    
    if first_conv is None:
        raise RuntimeError("Cannot find the first Conv2d layer in YOLOv8 model.")
    
    print(f"Found first Conv2d at '{first_conv_path}': "
          f"in_channels={first_conv.in_channels}, out_channels={first_conv.out_channels}")
    
    # 새로운 4채널 Conv2d 생성
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None
    )
    
    # 기존 3채널 가중치를 4채널로 복사 (4번째 채널은 기존 3채널의 평균으로 초기화)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = first_conv.weight
        new_conv.weight[:, 3:4, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
        if first_conv.bias is not None:
            new_conv.bias = first_conv.bias
    
    # 첫 번째 Conv2d를 새 4채널 레이어로 교체
    # YOLOv8의 구조에 따라 path를 파싱하여 교체
    parts = first_conv_path.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    
    last_attr = parts[-1]
    if last_attr.isdigit():
        parent[int(last_attr)] = new_conv
    else:
        setattr(parent, last_attr, new_conv)
    
    # 3. 마지막 분류 헤드의 출력 클래스 수 조정
    # YOLOv8-cls의 마지막 레이어는 보통 Linear
    last_linear = None
    last_linear_path = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = module
            last_linear_path = name
    
    if last_linear is not None and last_linear.out_features != num_classes:
        print(f"Replacing final Linear: {last_linear.out_features} → {num_classes} classes")
        
        new_linear = nn.Linear(last_linear.in_features, num_classes)
        
        parts = last_linear_path.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        last_attr = parts[-1]
        if last_attr.isdigit():
            parent[int(last_attr)] = new_linear
        else:
            setattr(parent, last_attr, new_linear)
    
    print(f"YOLOv8 4-channel model ready. Classes: {num_classes}")
    return model


class CSIClassifier(nn.Module):
    """
    YOLOv8 백본을 감싸는 래퍼 모델.
    입력: (B, 4, 900, 166)
    출력: (B, num_classes)
    """
    def __init__(self, num_classes, pretrained_model='yolov8n-cls.pt'):
        super().__init__()
        self.backbone = create_yolov8_4ch_classifier(num_classes, pretrained_model)
    
    def forward(self, x):
        out = self.backbone(x)
        # YOLOv8 내부 모델이 tuple을 반환할 수 있음
        if isinstance(out, tuple):
            out = out[0]
        return out


if __name__ == "__main__":
    # 테스트: 4채널 입력 모델 생성
    print("=== Action Model (3 classes) ===")
    action_model = CSIClassifier(num_classes=3)
    dummy_input = torch.randn(2, 4, 900, 166)
    output = action_model(dummy_input)
    print(f"Input: {dummy_input.shape} → Output: {output.shape}")
    
    print("\n=== Position Model (16 classes) ===")
    pos_model = CSIClassifier(num_classes=16)
    output = pos_model(dummy_input)
    print(f"Input: {dummy_input.shape} → Output: {output.shape}")

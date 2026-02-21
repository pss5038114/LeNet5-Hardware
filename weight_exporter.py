import os
import torch
import numpy as np
from LeNet_5_quantization import Net

class WeightExporter:
    def __init__(self, model_path):
        # 방금 학습한 구조와 동일하게 q=False (FP32) 로 모델 생성
        self.model = Net(q=False)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
    def _create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def export_fp32(self, save_dir='weights_fp32'):
        """2주 차 과제용: 순수 32비트 부동소수점 추출"""
        self._create_dir(save_dir)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name: # bias는 없으므로 weight만 추출
                    layer_name = name.split('.')[0]
                    file_path = f"{save_dir}/{layer_name}_weight.npy"
                    np.save(file_path, param.numpy())
        print(f"✅ FP32 가중치가 '{save_dir}' 폴더에 성공적으로 저장되었습니다!")

    def export_quantized(self, bit=8, save_dir='weights_int8'):
        """4주 차 과제용: 양자화 추출 (현재는 뼈대만)"""
        self._create_dir(save_dir)
        print(f"⚠️ {bit}-bit 양자화 로직은 4주 차에 이곳에 구현될 예정입니다.")

if __name__ == '__main__':
    # 방금 학습이 완료된 모델 파일명 지정
    exporter = WeightExporter('path_to_save_model.pth')
    
    # FP32 추출 실행
    exporter.export_fp32(save_dir='weights_fp32')
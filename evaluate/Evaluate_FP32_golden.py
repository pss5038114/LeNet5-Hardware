import os
import sys
import glob
import time
import numpy as np
from PIL import Image

# 현재 폴더(evaluate)의 부모 폴더(루트)에 있는 모델 클래스를 불러오기 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 부모 폴더의 LeNet_5_FP32_Golden.py 에서 클래스 가져오기
from LeNet_5_FP32_Golden import LeNet5_FP32

def evaluate_from_images():
    image_dir = os.path.join(current_dir, "test_images")
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_paths:
        print("❌ 테스트 이미지를 찾을 수 없습니다. extract_test_images.py를 먼저 실행해주세요.")
        return

    # 가중치 파일 경로 설정 (부모 폴더의 weights_fp32)
    weight_path = os.path.join(parent_dir, 'weights_fp32')
    model = LeNet5_FP32(weight_dir=weight_path)

    correct = 0
    total = len(image_paths)
    
    print(f"\n디렉토리에서 총 {total}장의 이미지 파일 로드 및 FP32 검증 시작...")
    start_time = time.time()
    
    for i, img_path in enumerate(image_paths):
        # 1. 파일 이름에서 실제 정답(Ground Truth) 추출 (예: "00000_label_7.png" -> 7)
        filename = os.path.basename(img_path)
        gt_label = int(filename.split('_label_')[1].split('.png')[0])
        
        # 2. 이미지 파일 로드 및 전처리
        # PyTorch의 transforms.ToTensor() 와 완전히 동일한 스케일링 (0.0 ~ 1.0)
        img = Image.open(img_path).convert('L')
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        # NumPy 추론기 입력 규격에 맞게 형태 변환 (1, 28, 28)
        ifmap = img_np.reshape(1, 28, 28) 
        
        # 3. FP32 추론 수행
        output = model.forward(ifmap)
        pred = np.argmax(output)
        
        # 4. 정답 확인
        if pred == gt_label:
            correct += 1
            
        # 1000장마다 진행 상황 출력
        if (i + 1) % 1000 == 0:
            current_acc = 100.0 * correct / (i + 1)
            print(f"[{i + 1:>5} / {total}] 실시간 정확도: {current_acc:.2f}%")
            
    end_time = time.time()
    final_accuracy = 100.0 * correct / total
    
    print("\n====================================")
    print(f"✅ 이미지 기반 FP32 최종 정확도 : {final_accuracy:.2f}%")
    print(f"⏱️ 총 검증 소요 시간            : {end_time - start_time:.2f} 초")
    print("====================================")

if __name__ == '__main__':
    evaluate_from_images()
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# 방금 전에 만든 순수 FP32 NumPy 추론기 클래스를 불러옵니다.
from LeNet_5_FP32_Golden import LeNet5_FP32

def evaluate_golden_model():
    print("MNIST 테스트 데이터셋 로딩 중...")
    # 1. 데이터셋 로드 (PyTorch의 기능을 빌려서 10,000장만 가져옵니다)
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    
    # 우리가 만든 모델이 1장씩 처리하도록 설계되었으므로 batch_size=1 로 설정
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # 2. FP32 추론기 초기화
    model = LeNet5_FP32(weight_dir='weights_fp32')

    correct = 0
    total = len(testset)
    
    print(f"\n총 {total}장 이미지에 대한 FP32 검증을 시작합니다...")
    start_time = time.time()
    
    # 3. 10,000장 반복 추론 및 채점
    for i, (image, label) in enumerate(testloader):
        # PyTorch Tensor (1, 1, 28, 28) 형태를 NumPy 배열 (1, 28, 28)로 변환
        ifmap = image.numpy()[0] 
        gt_label = label.numpy()[0]  # 실제 정답 (Ground Truth)
        
        # 순수 NumPy 그라운드업 추론
        output = model.forward(ifmap)
        pred = np.argmax(output)
        
        if pred == gt_label:
            correct += 1
            
        # 1000장마다 진행 상황 출력
        if (i + 1) % 1000 == 0:
            current_acc = 100.0 * correct / (i + 1)
            print(f"[{i + 1:>5} / {total}] 실시간 정확도: {current_acc:.2f}%")
            
    end_time = time.time()
    final_accuracy = 100.0 * correct / total
    
    print("\n====================================")
    print(f"✅ FP32 Golden Model 최종 정확도 : {final_accuracy:.2f}%")
    print(f"⏱️ 총 검증 소요 시간             : {end_time - start_time:.2f} 초")
    print("====================================")

if __name__ == '__main__':
    evaluate_golden_model()
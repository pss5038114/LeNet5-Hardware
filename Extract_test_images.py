import os
import torchvision
from PIL import Image

def create_eval_dataset():
    # evaluate 폴더 및 하위 test_images 폴더 생성
    output_dir = os.path.join("evaluate", "test_images")
    os.makedirs(output_dir, exist_ok=True)
    
    print("MNIST 테스트 데이터셋 로딩 중...")
    # PyTorch 기능을 빌려 이미지 다운로드 및 로드
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    total_images = len(testset)
    print(f"총 {total_images}장의 이미지를 '{output_dir}' 폴더에 PNG 파일로 저장합니다.")
    print("이 작업은 약 10~20초 정도 소요됩니다...")
    
    for i in range(total_images):
        image, label = testset[i]
        
        # 파일 이름에 인덱스와 실제 정답(Label)을 포함시켜 저장 (예: 00000_label_7.png)
        # 이렇게 하면 나중에 정답지를 따로 텍스트 파일로 안 만들어도 파일명만 보고 채점이 가능합니다.
        file_name = f"{i:05d}_label_{label}.png"
        file_path = os.path.join(output_dir, file_name)
        
        image.save(file_path)
        
        if (i + 1) % 2000 == 0:
            print(f"... {i + 1} / {total_images} 장 저장 완료")
            
    print("✅ 10,000장 이미지 추출 및 저장 완료!")

if __name__ == '__main__':
    create_eval_dataset()
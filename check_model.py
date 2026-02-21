import torch
from LeNet_5_quantization import Net  # 기존에 작성된 모델 구조를 불러옴

def check_model(model_path):
    print(f"--- '{model_path}' 파일 검증 시작 ---")
    
    # 1. FP32 빈 모델 생성
    model = Net(q=False) 
    
    try:
        # 2. 가중치 덮어씌우기
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print("✅ 성공: 정상적인 LeNet-5 FP32 모델입니다!\n")
        
        # 3. 레이어 정보 확인
        for name, param in model.named_parameters():
            print(f"Layer: {name:10} | Shape: {str(list(param.shape)):20} | Type: {param.dtype}")
            
    except Exception as e:
        print("❌ 실패: 모델 구조가 다르거나 손상된 파일입니다.")
        print("에러 내용:", e)

if __name__ == '__main__':
    check_model('model.pt')
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import time 

# ==========================================
# 1. NumPy 기반 기본 연산 함수 (FP32 그라운드업)
# ==========================================
def relu(x):
    return np.maximum(0, x)

def max_pool2d(input_array, kernel_size=2, stride=2):
    channels, in_height, in_width = input_array.shape
    out_height = (in_height - kernel_size) // stride + 1
    out_width = (in_width - kernel_size) // stride + 1
    
    output = np.zeros((channels, out_height, out_width), dtype=np.float32)
    
    for c in range(channels):
        for y in range(out_height):
            for x in range(out_width):
                output[c, y, x] = np.max(
                    input_array[c, y*stride:y*stride+kernel_size, x*stride:x*stride+kernel_size]
                )
    return output

def im2col(input_data, kernel_size, stride=1, padding=0):
    """
    하드웨어의 MAC 유닛(GEMM) 연산을 모사하기 위해
    입력 피처맵을 2차원 행렬로 전개하는 함수입니다.
    """
    channels, in_height, in_width = input_data.shape
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1

    if padding > 0:
        input_data = np.pad(input_data, ((0,0), (padding,padding), (padding,padding)), mode='constant')

    col = np.zeros((channels * kernel_size * kernel_size, out_height * out_width), dtype=np.float32)
    
    col_idx = 0
    for y in range(out_height):
        for x in range(out_width):
            window = input_data[:, y*stride:y*stride+kernel_size, x*stride:x*stride+kernel_size]
            col[:, col_idx] = window.flatten()
            col_idx += 1
            
    return col, out_height, out_width

# ==========================================
# 2. LeNet-5 모델 클래스 (PyTorch 없이 NumPy로만 구현)
# ==========================================
class LeNet5_FP32:
    def __init__(self, weight_dir='weights_fp32'):
        # 추출해둔 FP32 가중치(.npy) 로드
        self.conv1_w = np.load(f'{weight_dir}/conv1_weight.npy') # (6, 1, 5, 5)
        self.conv2_w = np.load(f'{weight_dir}/conv2_weight.npy') # (16, 6, 5, 5)
        self.fc1_w = np.load(f'{weight_dir}/fc1_weight.npy')     # (120, 256)
        self.fc2_w = np.load(f'{weight_dir}/fc2_weight.npy')     # (84, 120)
        self.fc3_w = np.load(f'{weight_dir}/fc3_weight.npy')     # (10, 84)

    def forward(self, ifmap):
        # 1. Conv1 -> ReLU -> Pool1
        out_c_1 = self.conv1_w.shape[0]
        col_1, out_h_1, out_w_1 = im2col(ifmap, kernel_size=5)
        w_1_col = self.conv1_w.reshape(out_c_1, -1)
        conv1_out = np.dot(w_1_col, col_1).reshape(out_c_1, out_h_1, out_w_1)
        
        relu1_out = relu(conv1_out)
        pool1_out = max_pool2d(relu1_out)

        # 2. Conv2 -> ReLU -> Pool2
        out_c_2 = self.conv2_w.shape[0]
        col_2, out_h_2, out_w_2 = im2col(pool1_out, kernel_size=5)
        w_2_col = self.conv2_w.reshape(out_c_2, -1)
        conv2_out = np.dot(w_2_col, col_2).reshape(out_c_2, out_h_2, out_w_2)

        relu2_out = relu(conv2_out)
        pool2_out = max_pool2d(relu2_out)

        # 3. Flatten
        flattened = pool2_out.flatten()

        # 4. FC1 -> ReLU
        fc1_out = np.dot(self.fc1_w, flattened)
        relu3_out = relu(fc1_out)

        # 5. FC2 -> ReLU
        fc2_out = np.dot(self.fc2_w, relu3_out)
        relu4_out = relu(fc2_out)

        # 6. FC3
        fc3_out = np.dot(self.fc3_w, relu4_out)
        return fc3_out

# ==========================================
# 3. Tkinter UI 기반 손글씨 인식 애플리케이션
# ==========================================
class App(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.title("LeNet-5 FP32 Golden 추론기")
        self.canvas = tk.Canvas(self, width=280, height=280, bg='black')
        self.canvas.pack(padx=10, pady=10)

        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.brush_size = 12 # 선명한 인식을 위해 브러시 크기 약간 상향
        self.brush_color = 'white'

        self.setup()

        self.predict_button = Button(self, text='Predict', command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.clear_button = Button(self, text='Clear', command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10, pady=10)

    def setup(self):
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.brush_color, outline='')
        self.draw.ellipse([x1, y1, x2, y2], fill=self.brush_color)

    def reset(self, event):
        self.canvas.bind('<B1-Motion>', self.paint)

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def preprocess(self):
        # 1. 28x28 사이즈로 축소
        # Image.LANCZOS (기존 ANTIALIAS) 방식으로 부드럽게 리사이징
        try:
            resample_method = Image.Resampling.LANCZOS
        except AttributeError:
            resample_method = Image.ANTIALIAS
            
        img = self.image.resize((28, 28), resample_method)
        img = ImageOps.invert(img)
        img = img.convert("L")

        # 2. NumPy 배열 변환 및 스케일링
        img_np = np.array(img, dtype=np.float32)
        img_np = 255.0 - img_np
        
        # PyTorch의 transforms.ToTensor()와 완벽히 동일한 스케일링 (0 ~ 1.0)
        img_np = img_np / 255.0 
        
        # 3. 모델 입력 형식 (Channels, Height, Width) 에 맞춤
        img_np = img_np.reshape(1, 28, 28)
        return img_np

    def predict(self):
        ifmap = self.preprocess()

        start_time = time.time()
        
        # 추론 수행 (FP32)
        output = self.model.forward(ifmap)
        prediction = np.argmax(output)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        print(f"====================================")
        print(f"FP32 Output Logits : \n{np.round(output, 4)}")
        print(f'Inference time     : {inference_time:.4f} seconds')
        print(f'Predicted digit    : {prediction}')
        print(f"====================================\n")

if __name__ == '__main__':
    # 모델 초기화 및 UI 실행
    lenet_fp32 = LeNet5_FP32(weight_dir='weights_fp32')
    app = App(lenet_fp32)
    app.mainloop()
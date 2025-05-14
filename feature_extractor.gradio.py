import gradio as gr
import paddle
import numpy as np
from PIL import Image
import base64
import io
import json
import time
import hashlib

print("初始化ChainGuard图像特征提取服务...")

# 定义特征提取器模型
class FeatureExtractor(paddle.nn.Layer):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        try:
            resnet = paddle.vision.models.resnet50(pretrained=True)
            self.features = paddle.nn.Sequential(*list(resnet.children())[:-1])
            print("成功加载ResNet50预训练模型")
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            # 简化版备用模型
            from paddle.nn import Conv2D, MaxPool2D, Flatten
            self.features = paddle.nn.Sequential(
                Conv2D(3, 64, kernel_size=7, stride=2, padding=3),
                MaxPool2D(kernel_size=3, stride=2, padding=1),
                Conv2D(64, 128, kernel_size=3, padding=1),
                MaxPool2D(kernel_size=2, stride=2),
                Conv2D(128, 256, kernel_size=3, padding=1),
                MaxPool2D(kernel_size=2, stride=2),
                Conv2D(256, 512, kernel_size=3, padding=1),
                MaxPool2D(kernel_size=2, stride=2),
                Conv2D(512, 1024, kernel_size=3, padding=1),
                MaxPool2D(kernel_size=2, stride=2),
                Conv2D(1024, 2048, kernel_size=3, padding=1),
                paddle.nn.AdaptiveAvgPool2D(1),
                Flatten()
            )
            print("已创建备用特征提取模型")
        
    def forward(self, x):
        return self.features(x)

# 图像预处理函数 - 确保使用float32类型
def preprocess_image(image):
    """预处理图像"""
    if image is None:
        return None
        
    try:
        # 调整大小为224x224
        img = image.resize((224, 224))
        # 转换为RGB
        img = img.convert('RGB')
        # 归一化到[0,1] - 显式指定float32类型
        img = np.array(img, dtype='float32') / 255.0
        # 调整通道顺序
        img = img.transpose((2, 0, 1))
        # 标准化处理 - 确保使用float32类型的数组
        mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape((3, 1, 1))
        std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape((3, 1, 1))
        img = (img - mean) / std
        # 添加批次维度
        img = np.expand_dims(img, axis=0)
        return paddle.to_tensor(img, dtype='float32')  # 明确指定数据类型
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None

# 计算图像指纹函数
def calculate_fingerprint(features, title, username):
    try:
        feature_bytes = json.dumps(features).encode('utf-8')
        salt = f"{title}:{username}:{int(time.time())}".encode('utf-8')
        fingerprint = hashlib.sha256(feature_bytes + salt).hexdigest()
        return fingerprint
    except Exception as e:
        print(f"计算指纹失败: {e}")
        return "计算失败"

# 特征提取函数 - UI版本
def extract_features_ui(image, title="未命名", username="匿名用户"):
    if image is None:
        return {"error": "请先上传图像"}, "请先上传图像"
    
    try:
        print(f"处理图像: 标题={title}, 用户={username}")
        model = FeatureExtractor()
        model.eval()
        
        input_tensor = preprocess_image(image)
        if input_tensor is None:
            return {"error": "图像预处理失败"}, "预处理失败"
        
        # 确保使用float32类型处理
        with paddle.no_grad():
            features = model(input_tensor)
        
        # 确保转换为列表时保持数据类型一致
        features_list = features.numpy().astype('float32').tolist()[0]
        fingerprint = calculate_fingerprint(features_list, title, username)
        
        display_features = {
            "features_preview": features_list[:10] + ["..."] + features_list[-10:],
            "features_length": len(features_list),
            "fingerprint": fingerprint,
            "timestamp": time.time()
        }
        
        print(f"特征提取成功: 长度={len(features_list)}")
        return display_features, fingerprint
        
    except Exception as e:
        error_msg = f"特征提取失败: {str(e)}"
        print(error_msg)
        return {"error": error_msg}, "处理出错"

# API函数 - 处理JSON请求
def extract_features_api(json_input):
    try:
        data = json.loads(json_input) if isinstance(json_input, str) else json_input
        
        if "image" not in data:
            return {"status": "error", "error": "缺少image字段"}
            
        image_b64 = data["image"]
        title = data.get("title", "未命名")
        username = data.get("username", "匿名用户")
        
        print(f"API请求: 标题={title}")
        
        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return {"status": "error", "error": f"图像解码失败: {str(e)}"}
            
        model = FeatureExtractor()
        model.eval()
        
        input_tensor = preprocess_image(image)
        if input_tensor is None:
            return {"status": "error", "error": "图像预处理失败"}
        
        with paddle.no_grad():
            features = model(input_tensor)
        
        features_list = features.numpy().tolist()[0]
        fingerprint = calculate_fingerprint(features_list, title, username)
        
        return {
            "status": "success",
            "features": features_list,
            "features_length": len(features_list),
            "fingerprint": fingerprint,
            "timestamp": time.time()
        }
    
    except Exception as e:
        return {"status": "error", "error": f"处理失败: {str(e)}"}

# 生成示例请求
def generate_api_request(image, title, username):
    if image is None:
        return {"error": "请先上传图像"}
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return json.dumps({
        "image": img_str,
        "title": title,
        "username": username
    }, ensure_ascii=False)

# 测试API调用
def test_api_call(json_request):
    try:
        # 处理已经是字典的情况
        if isinstance(json_request, dict):
            data = json_request
        # 如果是字符串，尝试解析JSON
        elif isinstance(json_request, str):
            # 移除开头的空格和换行符
            json_request = json_request.strip()
            data = json.loads(json_request)
        else:
            return {"error": "请提供有效的JSON请求"}
        
        # 调用API处理函数    
        return extract_features_api(data)
    except json.JSONDecodeError as e:
        return {"error": f"JSON解析错误: {str(e)}"}
    except Exception as e:
        return {"error": f"API测试失败: {str(e)}"}

# 创建Gradio界面 - 使用单一Blocks避免组件重复
with gr.Blocks(title="ChainGuard 图像特征提取服务") as demo:
    gr.Markdown("# ChainGuard 图像特征提取服务")
    gr.Markdown("上传图像并提取特征，生成独特的版权指纹")
    
    with gr.Tabs():
        # 第一标签页: 图像特征提取
        with gr.TabItem("图像特征提取"):
            with gr.Row():
                with gr.Column(scale=1):
                    title_input = gr.Textbox(label="版权标题", placeholder="输入作品标题", value="未命名作品")
                    username_input = gr.Textbox(label="用户名", placeholder="输入您的用户名", value="匿名用户")
                    image_input = gr.Image(type="pil", label="上传图像", sources=["upload"])
                    extract_btn = gr.Button("提取特征", variant="primary")
                
                with gr.Column(scale=1):
                    features_output = gr.JSON(label="特征详情")
                    fingerprint_output = gr.Textbox(label="版权指纹", lines=2)
            
            extract_btn.click(
                fn=extract_features_ui,
                inputs=[image_input, title_input, username_input],
                outputs=[features_output, fingerprint_output]
            )
        
        # 第二标签页: API测试
        with gr.TabItem("API测试"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 测试API请求")
                    test_image = gr.Image(type="pil", label="测试图像", sources=["upload"])
                    test_title = gr.Textbox(label="标题", value="API测试图像")
                    test_username = gr.Textbox(label="用户名", value="API测试用户")
                    
                    with gr.Row():
                        generate_btn = gr.Button("生成请求JSON", variant="primary")
                        test_btn = gr.Button("测试API调用", variant="secondary")
                    
                    api_request = gr.JSON(label="API请求JSON")
                    api_response = gr.JSON(label="API响应结果")
                    
            # 事件绑定
            generate_btn.click(
                fn=generate_api_request, 
                inputs=[test_image, test_title, test_username], 
                outputs=api_request
            )
            
            test_btn.click(
                fn=test_api_call,
                inputs=api_request,
                outputs=api_response
            )
        
        # 第三标签页: API文档
        with gr.TabItem("API文档"):
            gr.Markdown("""
            # ChainGuard 图像特征提取API

            ## 概述
            
            ChainGuard API提供图像特征提取服务，可生成用于版权保护和图像识别的特征向量和唯一指纹。

            ## API端点
            
            ### 特征提取
            ```
            POST /extract_features_ui
            Content-Type: application/json
            ```
            
            ### 健康检查
            ```
            GET /health
            ```
            
            ## 请求格式
            
            ```json
            {
                "image": "base64编码的图像数据",
                "title": "作品标题(可选，默认为'未命名')",
                "username": "用户名(可选，默认为'匿名用户')"
            }
            ```
            
            ## 响应格式
            
            ### 成功响应
            ```json
            {
                "status": "success",
                "features": [...],  // 特征向量(2048维)
                "features_length": 2048,
                "fingerprint": "sha256指纹哈希",
                "timestamp": 1620000000.123
            }
            ```
            
            ### 错误响应
            ```json
            {
                "status": "error",
                "error": "错误描述"
            }
            ```
            
            ## 常见错误
            
            | 错误信息 | 描述 |
            |---------|------|
            | "图像解码失败" | Base64图像数据无效或格式不支持 |
            | "图像预处理失败" | 图像无法被正确处理 |
            | "处理失败" | 服务器内部错误 |
            
            ## 健康检查响应
            
            ```json
            {
                "status": "ok",
                "service": "ChainGuard",
                "version": "1.0.0"
            }
            ```
            
            ## 客户端示例代码
            
            ### JavaScript
            ```javascript
            async function extractFeatures(imageFile, title, username) {
                // 将图像转换为Base64
                const reader = new FileReader();
                reader.readAsDataURL(imageFile);
                const base64 = await new Promise(resolve => {
                    reader.onload = () => resolve(reader.result);
                });
                
                // 提取base64数据部分
                const base64Data = base64.split(',')[1];
                
                // 构造请求数据
                const requestData = {
                    image: base64Data,
                    title: title || '未命名',
                    username: username || '匿名用户'
                };
                
                // 发送API请求
                const response = await fetch('https://您的部署地址/extract_features_ui', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestData)
                });
                
                return await response.json();
            }
            ```
            
            ### Python
            ```python
            import requests
            import base64
            from PIL import Image
            import io

            def extract_features(image_path, title="未命名", username="匿名用户"):
                # 读取并编码图像
                with open(image_path, "rb") as image_file:
                    image_bytes = image_file.read()
                
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                
                # 构造请求数据
                request_data = {
                    "image": image_b64,
                    "title": title,
                    "username": username
                }
                
                # 发送API请求
                response = requests.post(
                    "https://您的部署地址/extract_features_ui",
                    json=request_data
                )
                
                return response.json()
            ```
            
            ## 实现说明
            
            - 特征向量由ResNet50模型生成，维度为2048
            - 版权指纹使用SHA-256算法基于特征向量、标题和用户名生成
            - 支持常见图像格式：JPEG、PNG、BMP等
            - 建议图像大小不低于224x224像素
            - API请求大小限制为20MB
            """)
# 添加一个专门的RESTful API端点
def create_restful_api():
    from pydantic import BaseModel
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    from typing import Dict, Any, Optional
    
    # 创建API应用
    app = FastAPI(title="ChainGuard API")
    
    # 添加CORS支持
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # API请求模型
    class ExtractFeaturesRequest(BaseModel):
        image: str
        title: Optional[str] = "未命名"
        username: Optional[str] = "匿名用户"
        
    # 特征提取API端点
    @app.post("/extract_features_ui")
    async def api_extract_features(request: ExtractFeaturesRequest):
        try:
            # 解析Base64图像
            try:
                image_bytes = base64.b64decode(request.image)
                image = Image.open(io.BytesIO(image_bytes))
                print(f"API图像解析成功: {image.size}")
            except Exception as e:
                print(f"图像解码失败: {e}")
                return {"status": "error", "error": f"图像解码失败: {str(e)}"}
            
            # 创建模型并提取特征
            model = FeatureExtractor()
            model.eval()
            
            input_tensor = preprocess_image(image)
            if input_tensor is None:
                return {"status": "error", "error": "图像预处理失败"}
            
            with paddle.no_grad():
                features = model(input_tensor)
                
            # 验证特征向量非空
            if features is None or features.size == 0:
                return {"status": "error", "error": "模型生成了空特征向量"}
                
            # 确保转换为列表时保持数据类型一致
            features_list = features.numpy().astype('float32').tolist()[0]
            
            # 验证特征向量有效性
            if len(features_list) == 0:
                return {"status": "error", "error": "无法提取有效特征向量"}
                
            # 计算指纹
            fingerprint = calculate_fingerprint(features_list, request.title, request.username)
            
            # 返回结果
            return {
                "status": "success",
                "features": features_list,
                "features_length": len(features_list),
                "fingerprint": fingerprint,
                "timestamp": time.time()
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": f"处理失败: {str(e)}"}
    
    # 返回应用实例和启动函数
    return app

# 主程序入口点
if __name__ == "__main__":
    # 添加命令行参数支持
    import argparse
    parser = argparse.ArgumentParser(description="ChainGuard 图像特征提取服务")
    parser.add_argument("--port", type=int, default=8686, help="服务端口")
    args = parser.parse_args()
    
    # 预热模型以确保正常工作
    print("预热模型中...")
    try:
        dummy_model = FeatureExtractor()
        dummy_model.eval()
        dummy_input = paddle.randn([1, 3, 224, 224])
        with paddle.no_grad():
            dummy_output = dummy_model(dummy_input)
        print(f"模型预热成功，特征维度: {dummy_output.shape}")
    except Exception as e:
        print(f"模型预热失败: {e}")
    
    # 打印环境信息
    print(f"PaddlePaddle版本: {paddle.__version__}")
    print(f"NumPy版本: {np.__version__}")
    print(f"CUDA是否可用: {paddle.device.is_compiled_with_cuda()}")
    paddle.set_device('gpu' if paddle.device.is_compiled_with_cuda() else 'cpu')
    print(f"使用设备: {paddle.device.get_device()}")

    # 创建FastAPI应用
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    from fastapi.responses import RedirectResponse

    # 创建API应用
    app = create_restful_api()

    # 添加健康检查端点
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "service": "ChainGuard", "version": "1.0.0"}
    
    # 准备Gradio应用
    demo.queue()
    
    # 将Gradio挂载到根路径，但先定义API路由
    app = gr.mount_gradio_app(app, demo, path="")
        
    # 启动服务    
    print(f"启动服务在端口 {args.port}...")
    print(f"访问UI: http://localhost:{args.port}")
    print(f"访问API: http://localhost:{args.port}/extract_features_ui")
    print(f"健康检查: http://localhost:{args.port}/health")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
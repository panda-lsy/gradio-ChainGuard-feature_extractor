# ChainGuard 图像特征提取服务

<p align="center">
  <img src="https://img.shields.io/badge/PaddlePaddle-2.5%2B-blue" alt="PaddlePaddle"/>
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/Gradio-4.16%2B-orange" alt="Gradio"/>
  <img src="https://img.shields.io/badge/FastAPI-0.95%2B-green" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License"/>
</p>

<p align="center">
  <b>基于飞桨深度学习框架的图像特征提取与版权指纹服务</b>
</p>

<p align="center">
  🔗 <b><a href="https://app-s8zcofjc4dj6j3u4.aistudio-app.com/" target="_blank">在线演示</a></b>
</p>

## 📋 目录

- 系统概述
- 核心功能
- 技术架构
- 安装指南
- 使用方法
- API文档
- 部署指南
- 常见问题
- 贡献指南
- 许可证
- 鸣谢

## 🔍 系统概述

ChainGuard是一套专业的图像特征提取与版权保护方案，基于飞桨(PaddlePaddle)深度学习框架开发。系统利用ResNet50深度神经网络提取图像的高维特征向量，结合作品信息生成独特的版权指纹，可用于图像版权管理、溯源追踪、相似度比对和内容认证。

🌟 **在线演示**: [https://app-s8zcofjc4dj6j3u4.aistudio-app.com/](https://app-s8zcofjc4dj6j3u4.aistudio-app.com/)

### 应用场景

- 🖼️ **数字艺术品版权保护**：为艺术创作提供独特指纹，方便版权追踪
- 📸 **摄影作品溯源认证**：确保摄影作品可追溯并防止未授权使用
- 🎨 **NFT内容特征提取**：为数字藏品提供唯一性验证
- 🔎 **图像相似度比对**：检测内容重复或抄袭行为
- 🛡️ **内容安全与防伪验证**：防止图像伪造和盗用

## ✨ 核心功能

- **先进的特征提取**：基于ResNet50提取2048维特征向量，支持细粒度图像特征识别
- **安全版权指纹**：结合图像特征、作品标题和用户信息生成SHA256加密指纹
- **统一Web服务**：集成Gradio交互界面与FastAPI接口于同一端口（8686）
- **优雅降级机制**：无法加载预训练模型时自动启用备用特征提取网络
- **数据类型安全**：全面确保跨平台一致性，解决float32/float64兼容问题
- **自动检测硬件**：智能识别CUDA加速，在CPU和GPU环境均可高效运行
- **完整调试信息**：提供详细日志与健康检查端点，易于问题诊断

## 🔧 技术架构

ChainGuard采用现代化多层架构设计：

```
┌──────────────────────────────────────────────┐
│                  用户层                       │
│  ┌────────────┐       ┌───────────────────┐  │
│  │ Gradio UI  │       │ 第三方客户端应用   │  │
│  └────────────┘       └───────────────────┘  │
└──────────────────────────────────────────────┘
                   │              │
                   ▼              ▼
┌──────────────────────────────────────────────┐
│                  接口层                       │
│  ┌────────────┐       ┌───────────────────┐  │
│  │ Web界面    │       │ REST API接口      │  │
│  │ (/)        │       │ (/extract_features)│  │
│  └────────────┘       └───────────────────┘  │
└──────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                  业务层                       │
│  ┌────────────┐   ┌─────────┐   ┌─────────┐  │
│  │ 图像预处理 │   │特征提取 │   │指纹生成 │  │
│  └────────────┘   └─────────┘   └─────────┘  │
└──────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                  模型层                       │
│  ┌────────────────────────────────────────┐  │
│  │      ResNet50 预训练模型               │  │
│  │   (备用: 自定义CNN特征提取网络)         │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

## 📥 安装指南

### 环境要求

- Python 3.7+
- CUDA兼容GPU（可选，有助于加速处理）
- 2GB+ RAM

### 依赖组件

- PaddlePaddle 2.5+：深度学习框架
- Gradio 4.16+：Web界面组件
- FastAPI 0.95+：REST API框架
- Uvicorn：ASGI服务器
- NumPy, Pillow：数据处理库

### 安装步骤

1. **克隆代码库**

```bash
git clone https://github.com/yourusername/chainguard-feature-extractor.git
cd chainguard-feature-extractor
```

2. **创建虚拟环境**（推荐）

```bash
# 使用venv
python -m venv venv

# Windows激活
venv\Scripts\activate

# Linux/Mac激活
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **验证安装**

```bash
python -c "import paddle; print(f'PaddlePaddle版本: {paddle.__version__}')"
python -c "import gradio; print(f'Gradio版本: {gradio.__version__}')"
```

## 🚀 使用方法

### 在线演示

您可以访问我们的在线演示服务，无需安装任何环境：
[https://app-s8zcofjc4dj6j3u4.aistudio-app.com/](https://app-s8zcofjc4dj6j3u4.aistudio-app.com/)

### 启动本地服务

ChainGuard采用统一服务模式，在单一端口提供Web界面和API：

```bash
# 默认端口启动(8686)
python feature_extractor.gradio.py

# 指定自定义端口
python feature_extractor.gradio.py --port 7860
```

### Web界面交互

启动服务后，通过以下地址访问Web界面：

```
http://localhost:8686/
```

Web界面包含三个功能页面：

#### 1. 图像特征提取

- 输入作品标题和作者信息
- 上传待分析图像
- 点击"提取特征"按钮
- 查看生成的特征向量和版权指纹

#### 2. API测试

- 上传测试图像
- 填写标题和用户名
- 生成API请求JSON
- 执行API调用并查看结果

#### 3. API文档

- 查看API调用规范
- 获取示例代码
- 了解响应格式

### 健康检查

系统提供健康检查接口，用于监控服务状态：

```
GET http://localhost:8686/health
```

响应示例：

```json
{
  "status": "ok",
  "service": "ChainGuard",
  "version": "1.0.0"
}
```

## 📡 API文档

### API端点

ChainGuard提供了一个主要的特征提取API端点：

```
POST http://localhost:8686/extract_features_ui
Content-Type: application/json
```

线上服务API端点：

```
POST https://app-s8zcofjc4dj6j3u4.aistudio-app.com/extract_features_ui
Content-Type: application/json
```

### 请求格式

```json
{
  "image": "base64编码的图像数据",
  "title": "作品标题",
  "username": "用户名"
}
```

### 响应格式

成功响应：

```json
{
  "status": "success",
  "features": [0.123, -0.456, ...],  // 2048维特征向量
  "features_length": 2048,
  "fingerprint": "a1b2c3d4e5...",  // SHA256指纹
  "timestamp": 1620000000.123
}
```

错误响应：

```json
{
  "status": "error",
  "error": "错误描述"
}
```

### API使用示例(Python)

```python
import requests
import base64
import json

def extract_features(image_path, title="未命名", username="匿名用户"):
    # 读取并编码图像
    with open(image_path, "rb") as f:
        image_bytes = f.read()
  
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
  
    # 构造请求
    payload = {
        "image": image_b64,
        "title": title,
        "username": username
    }
  
    # 发送请求 - 使用在线服务地址
    response = requests.post(
        "https://app-s8zcofjc4dj6j3u4.aistudio-app.com/extract_features_ui",
        json=payload
    )
  
    return response.json()

# 使用示例
result = extract_features("path/to/image.jpg", "我的作品", "作者名")
print(f"指纹: {result['fingerprint']}")
print(f"特征向量长度: {result['features_length']}")
```

### API使用示例(JavaScript)

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
  
    // 发送API请求（使用绝对路径）
    const response = await fetch('https://app-s8zcofjc4dj6j3u4.aistudio-app.com/extract_features_ui', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    });
  
    return await response.json();
}
```

## 🌐 部署指南

### 本地部署

单命令启动完整服务：

```bash
python feature_extractor.gradio.py --port 8686
```

### Docker部署

1. **创建Dockerfile**:

```dockerfile
FROM python:3.9

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制应用文件
COPY feature_extractor.gradio.py .

# 暴露端口
EXPOSE 8686

# 运行应用
CMD ["python", "feature_extractor.gradio.py", "--port", "8686"]
```

2. **构建并运行Docker镜像**:

```bash
docker build -t chainguard-feature-extractor .
docker run -p 8686:8686 chainguard-feature-extractor
```

### 云平台部署

#### AIStudio部署

当前服务已部署于百度AIStudio，访问地址：
```
https://app-s8zcofjc4dj6j3u4.aistudio-app.com
```

访问路径：
- Web界面：`https://app-s8zcofjc4dj6j3u4.aistudio-app.com/`
- API接口：`https://app-s8zcofjc4dj6j3u4.aistudio-app.com/extract_features_ui`
- 健康检查：`https://app-s8zcofjc4dj6j3u4.aistudio-app.com/health`

部署步骤：
1. 在AIStudio创建新项目
2. 上传代码文件
3. 配置应用部署并选择对应资源
4. 启动服务，系统将自动分配域名

#### Hugging Face Spaces

1. 在Hugging Face创建新Space
2. 选择Gradio作为SDK
3. 上传feature_extractor.gradio.py和requirements.txt文件
4. 设置环境变量PYTHON_ENABLE_FASTAPI=1

## ❓ 常见问题

### 安装问题

| 问题                 | 解决方案                                                                           |
| -------------------- | ---------------------------------------------------------------------------------- |
| 安装PaddlePaddle失败 | 根据[官方安装指南](https://www.paddlepaddle.org.cn/install/quick)选择适合您系统的版本 |
| 找不到CUDA           | 确保安装了兼容的NVIDIA驱动和CUDA工具包，或使用CPU版本的PaddlePaddle                |
| 依赖冲突             | 创建新的虚拟环境并按顺序安装依赖                                                   |

### 运行问题

| 问题                                    | 解决方案                                                 |
| --------------------------------------- | -------------------------------------------------------- |
| 加载预训练模型失败                      | 系统会自动使用备用模型。如需预训练模型，确保网络连接正常 |
| 类型错误(`float32`/`float64`不匹配) | 已在代码中通过 `astype('float32')`解决此问题           |
| 内存不足                                | 增加系统内存或减小批处理大小                             |
| 摄像头功能无法使用                      | 安装 `opencv-python`并确保浏览器授予摄像头权限         |

### API问题

| 问题                  | 解决方案                                   |
| --------------------- | ------------------------------------------ |
| CORS错误              | 服务已配置允许跨域请求，检查客户端调用方式 |
| 请求超时              | 确认网络连接稳定，考虑增加客户端超时设置   |
| 空特征向量            | 确保图像正确上传且格式支持，检查图像质量   |
| 413错误(请求体积过大) | 压缩图像或调整服务器配置                   |

## 🤝 贡献指南

我们欢迎社区贡献代码、报告问题或提出新功能建议。请遵循以下步骤：

1. **Fork项目**到您的GitHub账户
2. **创建功能分支**：`git checkout -b feature/amazing-feature`
3. **提交更改**：`git commit -m 'Add amazing feature'`
4. **推送分支**：`git push origin feature/amazing-feature`
5. **提交Pull Request**

### 代码规范

- 遵循PEP 8编码风格
- 为新功能添加适当的文档
- 保持代码简洁明了
- 添加单元测试（如适用）

## 📄 许可证

本项目采用MIT许可证，详细信息请参见LICENSE文件。

## 🙏 鸣谢

- [PaddlePaddle](https://www.paddlepaddle.org.cn/) - 提供深度学习框架
- [Gradio](https://gradio.app/) - 提供交互式界面支持
- [FastAPI](https://fastapi.tiangolo.com/) - 提供高性能API框架
- [ResNet](https://arxiv.org/abs/1512.03385) - 提供图像特征提取网络架构
- [百度AIStudio](https://aistudio.baidu.com/) - 提供在线部署环境

---

<p align="center">
  <a href="https://app-s8zcofjc4dj6j3u4.aistudio-app.com/">立即体验在线演示</a>
</p>

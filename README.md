# sagemaker-llama3

在 Amazon SageMaker 上部署 Llama 3 / Qwen 系列大语言模型的两种实践方案，基于 [SGLang](https://sgl-project.github.io/) 推理引擎，提供 OpenAI 兼容的 Chat Completions API。

本仓库包含两套独立、可运行的部署示例，分别对应两种典型场景：

| 方案 | 路径 | 部署方式 | 适用场景 |
|------|------|----------|----------|
| **AWS DLC** | [`awsdlc/`](./awsdlc) | 直接使用 AWS 官方 SGLang Deep Learning Container | 快速上线、无需自建镜像，开箱即用 |
| **BYOC** | [`byoc/`](./byoc) | Bring Your Own Container，自行构建镜像推送到 ECR | 需要自定义 SGLang 版本、系统依赖或启动逻辑 |

两种方案均使用 SGLang 作为推理后端，暴露 **OpenAI 兼容的 Chat Completions API**，支持同步与流式推理。

---

## 方案一：AWS Deep Learning Container（推荐快速上手）

📂 [`awsdlc/`](./awsdlc) · 📓 `deploy_llama3_8b_dlc.ipynb`

使用 AWS 官方维护的 SGLang DLC 镜像（例如 `0.5.10-gpu-py312-cu129-ubuntu24.04-sagemaker`）部署 Llama 3 8B，**纯 boto3 调用，无需 sagemaker SDK**。

### 核心特点

- ✅ 镜像由 AWS 维护，无需自行构建
- ✅ 仅依赖 `boto3`，环境要求极简
- ✅ Notebook 从 S3 模型 → 创建 Model → EndpointConfig → 部署 → 推理 → 清理，端到端演示
- ✅ 同时演示非流式和流式推理

### 前置条件

1. **模型文件已上传到 S3**（HuggingFace 格式，与 Endpoint 同 Region）：

   ```
   s3://<your-bucket>/models/llama3-8b/
   ├── config.json
   ├── tokenizer.json / tokenizer_config.json / vocab.json
   ├── model-0000{1..4}-of-00004.safetensors
   └── model.safetensors.index.json
   ```

2. **IAM 权限**：执行 Notebook 的身份需要
   - `sts:GetCallerIdentity`
   - `sagemaker:CreateModel` / `CreateEndpointConfig` / `CreateEndpoint` / `DescribeEndpoint`
   - `sagemaker:InvokeEndpoint` / `InvokeEndpointWithResponseStream`
   - `sagemaker:DeleteEndpoint` / `DeleteEndpointConfig` / `DeleteModel`
   - `ecr:GetAuthorizationToken` / `ecr:BatchGetImage`

   > 💡 附加 `AmazonSageMakerFullAccess` 可覆盖绝大部分权限。

3. **SageMaker Execution Role** 需要：信任 `sagemaker.amazonaws.com`，并对模型 S3 bucket 有 `s3:GetObject` / `s3:ListBucket` 权限。

4. **实例配额**：默认 `ml.g6e.2xlarge`（1× L40S 48GB），需在 Service Quotas 中确认配额 > 0。

### 快速开始

打开 [`awsdlc/deploy_llama3_8b_dlc.ipynb`](./awsdlc/deploy_llama3_8b_dlc.ipynb)，按 Cell 顺序执行：

| Cell | 功能 |
|------|------|
| 1 | Setup & Configuration（Region / Role / S3 路径 / 实例类型） |
| 2 | Create SageMaker Model（S3 模型 + SGLang DLC 镜像 + 环境变量） |
| 3 | Create Endpoint Config & Deploy（含 `InferenceAmiVersion`） |
| 4 | Wait for Endpoint to be InService（约 10–15 分钟） |
| 5 | Test Inference（OpenAI 兼容，单次请求） |
| 6 | Test Streaming Inference（流式输出） |
| 7 | **Cleanup**（删除 Endpoint / EndpointConfig / Model）⚠️ 用完务必执行 |

### 推荐实例选型

| 实例类型 | GPU | 显存 | 适用场景 |
|----------|-----|------|----------|
| `ml.g5.2xlarge` | 1× A10G | 24GB | 7B FP16 / 8B INT8，TP=1 |
| `ml.g5.12xlarge` | 4× A10G | 96GB | 7B FP16 TP=4，或 13B |
| `ml.g6e.2xlarge` | 1× L40S | 48GB | **7B/8B FP16 单卡，余量充足（默认）** |
| `ml.p4d.24xlarge` | 8× A100 | 320GB | 70B 模型 |

### 已知坑点

- **Endpoint Config 必须指定 GPU 推理 AMI**，否则报 `CannotStartContainerError` 且无 CloudWatch 日志：

  ```python
  "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1"
  ```

- **S3 bucket 必须与 Endpoint 同 Region**，否则 `Could not access model data`。跨区先同步：

  ```bash
  aws s3 sync s3://source-bucket/models/ s3://target-bucket/models/ \
    --source-region us-east-1 --region us-east-2
  ```

- **SGLang 参数通过 `SM_SGLANG_*` 环境变量注入**，DLC 入口脚本会自动转成 CLI 参数：

  | 环境变量 | → CLI 参数 |
  |----------|-----------|
  | `SM_SGLANG_TP_SIZE=1` | `--tp-size 1` |
  | `SM_SGLANG_DTYPE=float16` | `--dtype float16` |
  | `SM_SGLANG_CONTEXT_LENGTH=8192` | `--context-length 8192` |

---

## 方案二：BYOC（Bring Your Own Container）

📂 [`byoc/`](./byoc) · 📓 `deploy_and_test.ipynb`

自行构建 SGLang 镜像推送到 ECR，再部署到 SageMaker 端点。利用 **SGLang v0.2.13+ 原生的 SageMaker API 支持**（内置 `/ping` 和 `/invocations` 端点），**无需额外代理层**。

### 核心特点

- ✅ **原生支持**：直接使用 SGLang 内置的 SageMaker 接口
- ✅ **简单部署**：单进程，无代理层
- ✅ **快速启动**：架构简化
- ✅ **易于维护**：代码清晰，调试方便
- ⚠️ `/invocations` 端点**仅支持 Chat Completions API**（`messages` 格式）

### 架构

```
┌─────────────────────────────────────┐
│     SageMaker 请求 (端口 8080)      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     SGLang 服务器 (端口 8080)       │
│  • GET  /ping            (原生)     │
│  • POST /invocations     (原生)     │
│  • OpenAI 兼容 API                  │
└─────────────────────────────────────┘
```

### 默认配置

```python
MODEL_ID        = "Qwen/Qwen3.5-0.8B"
INSTANCE_TYPE   = "ml.g6.2xlarge"
SGLANG_VERSION  = "v0.5.9"
```

### Notebook 步骤

打开 [`byoc/deploy_and_test.ipynb`](./byoc/deploy_and_test.ipynb)：

| Cell | 功能 |
|------|------|
| 4 | 构建镜像并推送到 ECR（只需执行一次） |
| 12–17 | 下载模型并上传到 S3 |
| 19–20 | 创建启动脚本 |
| 22–24 | 部署 SageMaker 端点 |
| 28 | Message API（非流式） |
| 30 | Message API（流式） |

---

## 推理示例（两方案通用）

两种方案部署后都暴露 **OpenAI 兼容的 Chat Completions API**，通过 SageMaker Runtime `invoke_endpoint` 调用。

### 非流式

```python
import boto3, json

smr = boto3.client("sagemaker-runtime", region_name="us-east-1")

payload = {
    "model": "llama3-8b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is the capital of France?"},
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
}

resp = smr.invoke_endpoint(
    EndpointName="llama3-8b-sglang-ep-xxxxxxxxxx",
    ContentType="application/json",
    Body=json.dumps(payload),
)
result = json.loads(resp["Body"].read())
print(result["choices"][0]["message"]["content"])
```

### 流式

```python
payload["stream"] = True

resp = smr.invoke_endpoint_with_response_stream(
    EndpointName="llama3-8b-sglang-ep-xxxxxxxxxx",
    ContentType="application/json",
    Body=json.dumps(payload),
)
for event in resp["Body"]:
    chunk = event.get("PayloadPart", {}).get("Bytes", b"")
    if chunk:
        print(chunk.decode("utf-8"), end="", flush=True)
```

### 支持的参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | string | 需与 `SM_SGLANG_SERVED_MODEL_NAME` 一致 |
| `messages` | array | `system` / `user` / `assistant` |
| `max_tokens` | int | 最大生成 token 数 |
| `temperature` | float | 0 为贪心，越高越随机 |
| `top_p` | float | nucleus sampling |
| `stream` | bool | 是否流式 |
| `stop` | string/array | 停止 token |
| `frequency_penalty` | float | 减少重复 |
| `presence_penalty` | float | 鼓励新话题 |

---

## 常见错误排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `CannotStartContainerError`（无日志） | 缺少 `InferenceAmiVersion` | EndpointConfig 中添加 `"InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1"` |
| `Could not access model data` | S3 与 Endpoint 跨 Region，或 Role 无 S3 权限 | 确认 Region 一致；补齐 S3 读取权限 |
| `ModelError: ... OOM` | 显存不足 | 换更大实例或使用量化 |
| `ResourceLimitExceeded` | 实例配额不足 | Service Quotas 中申请提升 |

查看 CloudWatch 日志：

```bash
aws logs get-log-events \
  --log-group-name /aws/sagemaker/Endpoints/<endpoint-name> \
  --log-stream-name AllTraffic/<instance-id> \
  --region <region>
```

---

## 选型建议

- **只想快速验证 / 生产直接上线 AWS 官方支持的版本** → 用 [`awsdlc/`](./awsdlc)
- **需要特定 SGLang 版本、加入自定义依赖或启动逻辑** → 用 [`byoc/`](./byoc)

---

## 参考资料

- [AWS DLC Available Images](https://aws.github.io/deep-learning-containers/reference/available_images/)
- [SGLang DLC Source & Entrypoint](https://github.com/aws/deep-learning-containers/tree/master/sglang)
- [SGLang DLC Changelog](https://github.com/aws/deep-learning-containers/blob/master/sglang/CHANGELOG.md)
- [SGLang Documentation](https://sgl-project.github.io/)
- [SageMaker Python SDK Docs](https://sagemaker.readthedocs.io/)

---

## License

_(按需添加 LICENSE 文件)_

# Deploy Llama-3-8B on SageMaker with SGLang DLC

使用 AWS SGLang Deep Learning Container 在 SageMaker 上部署 llama3-8B 模型，纯 boto3 调用，无需 sagemaker SDK。

## 目录结构

```
.
├── README.md                       # 本文档
└── deploy_llama3_8b_dlc.ipynb      # 部署 Notebook
```

## 前置条件

### 1. 运行环境

在以下任一环境中运行 Notebook：

- **SageMaker Studio / Notebook Instance**
- **本地机器** — 需要配置 AWS CLI credentials (`aws configure`)

仅依赖 `boto3`，无需安装 `sagemaker` SDK。

### 2. IAM 权限

执行 Notebook 的 IAM Role / User 需要以下权限：

| 权限 | 用途 |
|------|------|
| `sts:GetCallerIdentity` | 获取 Account ID |
| `sagemaker:CreateModel` | 创建 SageMaker Model |
| `sagemaker:CreateEndpointConfig` | 创建 Endpoint Config |
| `sagemaker:CreateEndpoint` | 部署 Endpoint |
| `sagemaker:DescribeEndpoint` | 等待 Endpoint 就绪 |
| `sagemaker:InvokeEndpoint` | 调用推理接口 |
| `sagemaker:InvokeEndpointWithResponseStream` | 流式推理 |
| `sagemaker:DeleteEndpoint`, `DeleteEndpointConfig`, `DeleteModel` | 清理资源 |
| `ecr:GetAuthorizationToken`, `ecr:BatchGetImage` | 拉取 SGLang DLC 镜像 |

> **快捷方式：** 附加 `AmazonSageMakerFullAccess` 托管策略即可覆盖以上大部分权限。

### 3. SageMaker Execution Role

Notebook 中需要指定一个 SageMaker Execution Role ARN（`role` 变量），该 Role 需要：

- **信任策略**：允许 `sagemaker.amazonaws.com` assume
- **S3 权限**：对模型所在的 S3 bucket 有 `s3:GetObject` 和 `s3:ListBucket` 权限

### 4. 模型文件准备

将模型文件（HuggingFace 格式）上传到 S3，需与 SageMaker Endpoint 在**同一 Region**：

```bash
s3://<your-bucket>/models/llama3-8b/
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
└── model.safetensors.index.json
```

### 5. Service Quota

确认目标 Region 有对应实例类型的 SageMaker Endpoint 配额：

- 默认实例类型：`ml.g6e.2xlarge`（1x L40S 48GB）
- 在 AWS Console → Service Quotas → SageMaker 中搜索 `ml.g6e.2xlarge for endpoint usage`
- 如配额为 0，需提交申请

---

## 需要修改的参数

Notebook **Cell 1 (Setup & Configuration)** 中需要根据实际情况修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `REGION` | `"us-east-1"` | 必须与 S3 模型数据在同一 Region |
| `role` | 当前账号的 SageMaker Execution Role ARN | 需有 S3 读取权限 |
| `S3_MODEL_URI` | `"s3://salunchbucket/models/llama3-8b/"` | 模型文件的 S3 路径 |
| `INSTANCE_TYPE` | `"ml.g6e.2xlarge"` | SageMaker 实例类型 |
| `TENSOR_PARALLEL_DEGREE` | `1` | 张量并行度，多卡实例改为对应 GPU 数 |

以下 SGLang 参数通常不需要修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SGLANG_IMAGE_TAG` | `"0.5.10-gpu-py312-cu129-ubuntu24.04-sagemaker"` | SGLang DLC 镜像版本 |
| `SM_SGLANG_CONTEXT_LENGTH` | `"4096"` | 最大上下文长度 |
| `SM_SGLANG_DTYPE` | `"float16"` | 数据类型 |

### 实例选型参考

| 实例类型 | GPU | 显存 | 适用场景 |
|----------|-----|------|----------|
| `ml.g5.2xlarge` | 1x A10G | 24GB | 7B FP16 / 8B INT8，TP=1 |
| `ml.g5.12xlarge` | 4x A10G | 96GB | 7B FP16 TP=4，或 13B 模型 |
| `ml.g6e.2xlarge` | 1x L40S | 48GB | 7B FP16 单卡，余量充足 |
| `ml.p4d.24xlarge` | 8x A100 | 320GB | 70B 模型 |

---

## 部署流程概览

```
Cell 1  → Setup & Configuration（Region / Role / S3 路径 / 实例类型）
Cell 2  → Create SageMaker Model（S3 模型 + SGLang DLC 镜像 + 环境变量）
Cell 3  → Create Endpoint Config & Deploy（含 InferenceAmiVersion）
Cell 4  → Wait for Endpoint to be InService（约 10-15 分钟）
Cell 5  → Test Inference（OpenAI 兼容格式，单次请求）
Cell 6  → Test Streaming Inference（流式输出）
Cell 7  → Cleanup（删除 Endpoint / EndpointConfig / Model）  ← 用完务必执行
```

---

## 关键配置说明

### InferenceAmiVersion（必须）

Endpoint Config 中必须指定 GPU inference AMI：

```python
"InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1"
```

缺少此参数会导致 `CannotStartContainerError`，容器无法启动且无 CloudWatch 日志。

### SM_SGLANG_* 环境变量

SGLang DLC 的入口脚本（`sagemaker_entrypoint.sh`）会自动扫描所有 `SM_SGLANG_*` 环境变量，转换为 SGLang CLI 参数：

```
SM_SGLANG_TP_SIZE=1         →  --tp-size 1
SM_SGLANG_DTYPE=float16     →  --dtype float16
SM_SGLANG_CONTEXT_LENGTH=8192  →  --context-length 8192
```

容器默认设置：`--model-path /opt/ml/model --host 0.0.0.0 --port 8080`

### S3 模型数据与 Region

S3 bucket 必须与 SageMaker Endpoint 在**同一 Region**。跨 Region 会报 `Could not access model data` 错误。如需跨区域，先复制模型：

```bash
aws s3 sync s3://source-bucket/models/ s3://target-bucket/models/ \
  --source-region us-east-1 --region us-east-2
```

---

## 推理接口

SGLang DLC 暴露 **OpenAI 兼容的 Chat Completions API**，通过 SageMaker Runtime 的 `invoke_endpoint` 调用。

### 单次推理

```python
import boto3
import json

REGION = "us-east-1"
endpoint_name = "llama3-8b-sglang-ep-1777267479"  # 替换为实际 endpoint 名称

smr_client = boto3.client("sagemaker-runtime", region_name=REGION)

payload = {
    "model": "llama3-8b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
}

response = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload),
)

result = json.loads(response["Body"].read().decode("utf-8"))
print(result["choices"][0]["message"]["content"])
```

返回格式（OpenAI 兼容）：

```json
{
  "id": "715c1556d62d4047a44b9f27e5ca72cf",
  "object": "chat.completion",
  "model": "llama3-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Paris. The capital of France is Paris..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 26,
    "total_tokens": 100,
    "completion_tokens": 74
  }
}
```

### 流式推理

```python
payload_stream = {
    "model": "llama3-8b",
    "messages": [
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": True,
}

response_stream = smr_client.invoke_endpoint_with_response_stream(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload_stream),
)

event_stream = response_stream["Body"]
for event in event_stream:
    chunk = event.get("PayloadPart", {}).get("Bytes", b"")
    if chunk:
        text = chunk.decode("utf-8")
        print(text, end="", flush=True)
print()
```

### 多轮对话

```python
messages = [
    {"role": "system", "content": "你是一个中文助手。"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是人工智能的一个子领域，通过数据训练模型来进行预测或决策。"},
    {"role": "user", "content": "它和深度学习有什么区别？"},
]

payload = {
    "model": "llama3-8b",
    "messages": messages,
    "max_tokens": 512,
    "temperature": 0.7,
}

response = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload),
)

result = json.loads(response["Body"].read().decode("utf-8"))
print(result["choices"][0]["message"]["content"])
```

### 支持的请求参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | string | 模型名称，需与 `SM_SGLANG_SERVED_MODEL_NAME` 一致 |
| `messages` | array | 对话消息列表，支持 `system` / `user` / `assistant` 角色 |
| `max_tokens` | int | 最大生成 token 数 |
| `temperature` | float | 采样温度，0 为贪心解码，越高越随机 |
| `top_p` | float | nucleus sampling 参数 |
| `stream` | bool | 是否流式输出 |
| `stop` | string/array | 停止生成的 token 或字符串 |
| `frequency_penalty` | float | 频率惩罚，减少重复 |
| `presence_penalty` | float | 存在惩罚，鼓励新话题 |

---

## 排障

### Endpoint 创建失败

查看 CloudWatch 日志：

```bash
aws logs get-log-events \
  --log-group-name /aws/sagemaker/Endpoints/<endpoint-name> \
  --log-stream-name AllTraffic/<instance-id> \
  --region <region>
```

### 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `CannotStartContainerError`（无日志） | 缺少 `InferenceAmiVersion` | 在 EndpointConfig 中添加 `"InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1"` |
| `Could not access model data` | S3 bucket 与 Endpoint 不在同一 Region，或 Role 无 S3 权限 | 确认 Region 一致，给 Role 添加 S3 读取权限 |
| `ModelError: ... OOM` | 显存不足 | 换更大实例或使用量化 |
| `ResourceLimitExceeded` | 实例配额不足 | 在 Service Quotas 中申请提升 |

---

## 参考资料

- [AWS DLC Available Images](https://aws.github.io/deep-learning-containers/reference/available_images/)
- [SGLang DLC Source & Entrypoint](https://github.com/aws/deep-learning-containers/tree/master/sglang)
- [SGLang DLC Changelog](https://github.com/aws/deep-learning-containers/blob/master/sglang/CHANGELOG.md)
- [SGLang Documentation](https://sgl-project.github.io/)

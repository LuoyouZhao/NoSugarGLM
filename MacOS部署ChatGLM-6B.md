# MacOS部署ChatGLM-6B

> 经过测试，由于ChatGLM2-6B无法在mac上运行，改为尝试部署ChatGLM-6B
> 

## 下载仓库以及模型

### 下载仓库

```bash
git clone https://github.com/THUDM/ChatGLM2-6B
cd ChatGLM2-6B
```

### 下载模型

由于mac只能从本地加载模型，我们需要先将模型下载到本地；从 Hugging Face Hub 下载模型需要先安装Git LFS，请参考

[https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

然后我们就可以下载模型

```bash
git clone https://huggingface.co/THUDM/chatglm-6b
```

## 安装依赖

作者提供的依赖文件

```bash
pip install -r requirements.txt
```

安装 openmp 依赖

```bash
curl -O https://mac.r-project.org/openmp/openmp-14.0.6-darwin20-Release.tar.gz
sudo tar fvxz openmp-14.0.6-darwin20-Release.tar.gz -C /
```

### PyTorch-Nightly

安装 PyTorch-Nightly

```bash
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

验证PyTorch-Nightly

```python
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
```

正确输出应该是

```python
tensor([1.], device='mps:0')
```

运行模型

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("your local path", trust_remote_code=True)
model = AutoModel.from_pretrained("your local path", trust_remote_code=True).half().to('mps')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
```

`注意：mac并不支持模型量化`
# CogvideoX_IP

## 配置同CogvideX
### 安装依赖
```bash
# clone story repo
cd CogvideoX_IP
# create conda env
conda create --name cogip python=3.10
# activate env
conda activate cogip
# Install dependent packages
pip install -r requirements.txt
```
### 下载模型权重（复制CogvideoX）
首先，前往 SAT 镜像下载模型权重。

对于 CogVideoX-2B 模型，请按照如下方式下载:

```shell
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
unzip vae.zip
wget https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1
mv 'index.html?dl=1' transformer.zip
unzip transformer.zip
```

请按如下链接方式下载 CogVideoX-5B 模型的 `transformers` 文件（VAE 文件与 2B 相同）：

+ [CogVideoX-5B](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX-5B-I2V](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)

接着，你需要将模型文件排版成如下格式：

```
.
├── transformer
│   ├── 1000 (or 1)
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```

由于模型的权重档案较大，建议使用`git lfs`。`git lfs`
安装参见[这里](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing)

```shell
git lfs install
```

接着，克隆 T5 模型，该模型不用做训练和微调，但是必须使用。
> 克隆模型的时候也可以使用[Modelscope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b)上的模型文件位置。

```shell
git clone https://huggingface.co/THUDM/CogVideoX-2b.git #从huggingface下载模型
# git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-2b.git #从modelscope下载模型
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

通过上述方案，你将会得到一个 safetensor 格式的T5文件，确保在 Deepspeed微调过程中读入的时候不会报错。

```
├── added_tokens.json
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── spiece.model
└── tokenizer_config.json

0 directories, 8 files
```

## 训练
### 准备数据集

数据集格式应该如下：

```
.
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── ref_images
│   └── 1
│   │   ├──1.jpg
│   │   └──1_masked.jpg
│   └── 2
│   │   ├──2.jpg
│   │   └──2_masked.jpg
│   └── ...  
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...  
```

视频与标签,参考图像应该一一对应。通常情况下，不使用一个视频对应多个标签和多张参考图像。

### 修改配置文件
微调方式都仅仅对 `transformer` 部分进行微调。不改动 `VAE` 部分。`T5`仅作为Encoder 使用。

请按照以下方式修改`configs/sft.yaml`(全量微调) 中的文件。

```yaml
  checkpoint_activations: True ## using gradient checkpointing (配置文件中的两个checkpoint_activations都需要设置为True)
  model_parallel_size: 1 # 模型并行大小
  experiment_name: cross_attention   # 实验名称
  mode: finetune # 模式(不要改动)
  load: "{your_CogVideoX-2b-sat_path}/transformer" ## Transformer 模型路径
  no_load_rng: True # 是否加载随机数种子
  epochs: 50 
#   train_iters: 1000 # 训练迭代次数
  eval_iters: 1 # 验证迭代次数
  eval_interval: 100    # 验证间隔
  eval_batch_size: 1  # 验证集 batch size
  save: ckpts # 模型保存路径 
  save_interval: 100 # 模型保存间隔
  log_interval: 20 # 日志输出间隔
  train_data: [ "your train data path" ]
  valid_data: [ "your val data path" ] # 训练集和验证集可以相同
  split: 1,0,0 # 训练集，验证集，测试集比例
  num_workers: 8 # 数据加载器的工作线程数
  force_train: True # 在加载checkpoint时允许missing keys (T5 和 VAE 单独加载)
  only_log_video_latents: True # 避免VAE decode带来的显存开销
  deepspeed:
    bf16:
      enabled: False # For CogVideoX-2B Turn to False and For CogVideoX-5B Turn to True
    fp16:
      enabled: True  # For CogVideoX-2B Turn to True and For CogVideoX-5B Turn to False
```

请按照以下方式修改`configs/cogvideox_2b_ip_attention.yaml`中的文件。

```yaml
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  not_trainable_prefixes: [ 'all' ] ## 解除注释
  log_keys:
    - txt'

transformer_args:
    checkpoint_activations: True ## using gradient checkpointing
    ...
    is_decoder: false
    image_encoder: true #True for image_encoder

conditioner_config:
target: sgm.modules.GeneralConditioner
params:
    emb_models:
    - is_trainable: false 
        input_key: txt
        ucg_rate: 0.1
        target: sgm.modules.encoders.modules.FrozenT5Embedder
        params:
        model_dir: "./t5-v1_1-xxl"
        max_length: 226

    # ref_image encoder
    - is_trainable: true # True for encoder_train , false for inference
        input_key: ref_image
        target: sgm.modules.encoders.modules.ImageEncoder
        params:
        image_hidden_size: 1024
        hidden_size: 1920

```

### 修改运行脚本

编辑`finetune_single_gpu.sh` 或者 `finetune_multi_gpus.sh`，选择配置文件。下面是多卡训练的例子:

```
run_cmd="torchrun --standalone --nproc_per_node=3 train_video.py --base configs/cogvideox_2b_ip_attention.yaml configs/sft.yaml --seed $RANDOM"
```
### 微调

运行bash代码,即可开始微调。

```shell
bash finetune_single_gpu.sh # Single GPU
bash finetune_multi_gpus.sh # Multi GPUs
```

### 验证

### 验证数据格式如下：
```
.
└── dog_inference
    ├── image0.jpg
    ├── image1.jpg
    ├── ... 
    ├── prompt.txt 
    ├── word_prompt.txt
```
### 修改配置文件
`configs/cogvideox_2b_ip_attention.yaml`：
```yaml
# ref_image encoder
    - is_trainable: false # True for encoder_train , false for inference
        input_key: ref_image
        target: sgm.modules.encoders.modules.ImageEncoder
        params:
        image_hidden_size: 1024
        hidden_size: 1920
```
`configs/inference.yaml`：

```yaml
args:
  ...
  # load: "../transformer" # This is for Full model without image encoder
  load: "./ckpts_2b_ip/Cross_attention-11-08-11-02" # This is for Full model with image encoder
  batch_size: 1
  input_type: txt
  # input_file: configs/test.txt # For CogvideoX_2b
  input_file: dog_inference # For CogvideX_2b_attention
  ...
  output_dir: outputs_attention/ # results
```

修改推理配置文件 `inference.sh`

```
run_cmd="$environs python sample_video_ip_attention.py --base configs/cogvideox_2b_ip_attention.yaml configs/inference.yaml --seed 42"
```

然后，执行代码:

```
bash inference.sh 
```
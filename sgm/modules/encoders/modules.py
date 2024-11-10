import math
from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import kornia
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    CLIPVisionModel
)
from torchvision import transforms
from PIL import Image

from ...util import (
    append_dims,
    autocast,
    count_params,
    default,
    disabled_train,
    expand_dims_like,
    instantiate_from_config,
)


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1,"ref_image":0}

    def __init__(self, emb_models: Union[List, ListConfig], cor_embs=[], cor_p=[]):
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models): 
            embedder = instantiate_from_config(embconfig)
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0) # ucg_rate参数的作用  
            # # 文本编码器训练设置
            # if not embedder.is_trainable:
            #     embedder.train = disabled_train
            #     for param in embedder.parameters():
            #         param.requires_grad = False
            #     embedder.eval()
            # print(
            #     f"Initialized embedder #{n}: {embedder.__class__.__name__} "
            #     f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            # )

            # cross_attention 训练参数设置
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            else:
                embedder.train = disabled_train
                for name, param in embedder.named_parameters():
                    if 'image_proj' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                print(
                    f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                    f"with {count_params(embedder.image_proj, False)} params. Trainable: {embedder.is_trainable}"
                )

            # # T5 训练参数设置
            # if not embedder.is_trainable:
            #     embedder.train = disabled_train
            #     for param in embedder.parameters():
            #         param.requires_grad = False
            #     embedder.eval()
            # else:
            #     embedder.train = disabled_train
            #     for name, param in embedder.named_parameters():
            #         if 'mapper' in name:
            #             param.requires_grad = True
            #         else:
            #             param.requires_grad = False
            # print(
            #     f"Initialized embedder #{n}: {embedder.__class__.__name__} "
            #     f"with {count_params(embedder.mapper, False)} params. Trainable: {embedder.is_trainable}"
            # )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}")

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None) # legacy_ucg_value参数的作用
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

        if len(cor_embs) > 0:
            assert len(cor_p) == 2 ** len(cor_embs)
        self.cor_embs = cor_embs
        self.cor_p = cor_p

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def surely_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict, cond_or_not) -> Dict:
        assert embedder.legacy_ucg_val is not None
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if cond_or_not[i]:
                batch[embedder.input_key][i] = val
        return batch

    def get_single_embedding(
        self,
        embedder,
        batch,
        output,
        cond_or_not: Optional[np.ndarray] = None,
        force_zero_embeddings: Optional[List] = None,
    ):
        embedding_context = nullcontext if embedder.is_trainable else torch.no_grad

        with embedding_context():
            if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                if embedder.legacy_ucg_val is not None:
                    if cond_or_not is None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    else:
                        batch = self.surely_get_ucg_val(embedder, batch, cond_or_not)
                emb_out = embedder(batch[embedder.input_key])
            elif hasattr(embedder, "input_keys"):
                emb_out = embedder(*[batch[k] for k in embedder.input_keys])

        # # T5输入
        # with embedding_context():
        #     if hasattr(embedder, "input_key") and (embedder.input_key is not None):
        #         if embedder.legacy_ucg_val is not None:
        #             if cond_or_not is None:
        #                 batch = self.possibly_get_ucg_val(embedder, batch)
        #             else:
        #                 batch = self.surely_get_ucg_val(embedder, batch, cond_or_not)
        #         emb_out = embedder({"txt":batch[embedder.input_key]})
        #     elif hasattr(embedder, "input_keys"):
        #         emb_out = embedder({"txt":batch[embedder.input_keys[0]],
        #                             "word_prompt":batch[embedder.input_keys[1]],
        #                             "image":batch[embedder.input_keys[2]]
        #                             })
        assert isinstance(
            emb_out, (torch.Tensor, list, tuple)
        ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
        if not isinstance(emb_out, (list, tuple)):
            emb_out = [emb_out]
        for emb in emb_out:
            if embedder.input_key == 'txt':
                out_key = "crossattn"
            elif embedder.input_key == 'ref_image':
                out_key = "ref_image"
            # out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
            if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                if cond_or_not is None:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli((1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device)),
                            emb,
                        )
                        * emb
                    )
                else:
                    emb = (
                        expand_dims_like(
                            torch.tensor(1 - cond_or_not, dtype=emb.dtype, device=emb.device),
                            emb,
                        )
                        * emb
                    )
            if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
            # if hasattr(embedder, "input_keys") and embedder.input_keys[0] in force_zero_embeddings:
                emb = torch.zeros_like(emb)
            if out_key in output:
                output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
            else:
                output[out_key] = emb
        return output

    def forward(self, batch: Dict, force_zero_embeddings: Optional[List] = None) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []

        if len(self.cor_embs) > 0: # self.cor_embs=[]
            batch_size = len(batch[list(batch.keys())[0]])
            rand_idx = np.random.choice(len(self.cor_p), size=(batch_size,), p=self.cor_p)
            for emb_idx in self.cor_embs:
                cond_or_not = rand_idx % 2
                rand_idx //= 2
                output = self.get_single_embedding(
                    self.embedders[emb_idx],
                    batch,
                    output=output,
                    cond_or_not=cond_or_not,
                    force_zero_embeddings=force_zero_embeddings,
                )

        for i, embedder in enumerate(self.embedders):

            if i in self.cor_embs:
                continue
            output = self.get_single_embedding(
                embedder, batch, output=output, force_zero_embeddings=force_zero_embeddings
            )
        return output

    def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        cor_embs = self.cor_embs
        cor_p = self.cor_p
        self.cor_embs = []
        self.cor_p = []

        c = self(batch_c)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        self.cor_embs = cor_embs
        self.cor_p = cor_p

        return c, uc


class FrozenT5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self,
        model_dir="google/t5-v1_1-xxl",
        device="cuda",
        max_length=77,
        freeze=True,
        cache_dir=None,
    ):
        super().__init__()
        if model_dir is not "google/t5-v1_1-xxl":
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir, cache_dir=cache_dir)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    # @autocast
    def forward(self, text):
        # text =["A black and white animated steamboat sits on a serene body of water amidst a hilly or mountainous landscape under a cloudy sky. The steamboat's tall stack emits smoke in various stages, transforming from thick and dark to light and dispersed, eventually dissipating into billowing clouds. The boat remains stationary, with minor movements due to water currents, in a consistent environment with static lighting conditions, focusing solely on the dynamic smoke emission from the steamboat's stack.",
        # "A black and white character with a hat kneels beside a steering wheel, intensely gripping it, while a smaller character resembling Mickey Mouse stands nearby, watching with concern. The scene remains static, with the characters' expressions changing as they interact. The smaller character raises its arms in surprise, prompting the larger character to defend or express itself. The larger character leans forward, determined or aggressive, while the smaller character looks on, caught off guard. The fixed camera captures the characters within a maritime setting, showcasing their evolving emotions."]
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ) 
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

class ImageEncoder(AbstractEmbModel):
    def __init__(self,
                image_hidden_size:int,
                hidden_size:int,
                device="cuda",
                freeze=True,
                model_dir="openai/clip-vit-large-patch14",
    ):
        super(ImageEncoder,self).__init__()
        self.image_encoder = CLIPVisionModel.from_pretrained(model_dir)
        self.image_proj = nn.Linear(image_hidden_size, hidden_size)
        self.device = device

        if freeze:
            self.freeze()


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


    def forward(self,image_path_list):
        if image_path_list is not None:
            images = [Image.open(path) for path in image_path_list]
            image_features_list = [] 
            for image in images:
                preprocess = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),          
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])  
                ])
                image = preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.image_encoder(image, output_hidden_states=False)[0]
                image_features = self.image_proj(image_features)
                image_features_list.append(image_features)
            images_features = torch.cat(image_features_list, dim=0)
        else: 
            images_features = None
        return images_features

# if __name__ == "__main__":
#     encoder = ImageEncoder(1024,1920)
#     image_path = ["/home/tuyijing/CogVideo/SwissArmyTransformer/cogvideo/dog_dataset/ref_images/1/1_masked.jpg"]
#     image_feature = encoder(image_path)


# Mapper to transform the clip image embedding to T5 text embedding
class Mapper(nn.Module):
    def __init__(self,
        input_dim: int,#1024
        output_dim: int,#4096
    ):
        super(Mapper, self).__init__()

        for i in range(5):#一共5层,最后一层特征，以及隐藏层
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:]).mean(dim=1, keepdim=True) ##emb[1,257,1024],emb[:,:1]是图像patch前的cls token
            # print(hidden_state.size())
            hidden_states += (hidden_state, )
        hidden_states = torch.cat(hidden_states, dim=1)

        return hidden_states

class Inj_T5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self,
        model_dir="google/t5-v1_1-xxl",
        device="cuda",
        max_length=77,
        freeze=True,
        cache_dir=None,
        image=None
    ):
        super().__init__()
        if model_dir is not "google/t5-v1_1-xxl":
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir, cache_dir=cache_dir)
        self.device = device
        self.max_length = max_length
        # if freeze:
        #     self.freeze()

        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

        self.mapper=Mapper(input_dim=1024,output_dim=4096)

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    # @autocast
    def forward(self, inputs):
        # print(inputs)
        text = inputs['txt']
        if 'image' in inputs:
            image = inputs['image']
            word_prompt = inputs['word_prompt']
            # print(word_prompt)
        else:
            image = None
            word_prompt = None
        
        
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        original_ids = batch_encoding["input_ids"].to(self.device) #[1,77]

        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=original_ids)
        original_outputs = outputs.last_hidden_state #[1,226,4096]
        inj_outputs = original_outputs.clone()

        if image is not None:
            if str(image.device) == 'cpu':
                image = image.to(self.device)
            inj_index = self.tokenizer(word_prompt, add_special_tokens=False,return_tensors="pt")["input_ids"].to(self.device) #lion:[[3,7325]]
            # print(f"inj_index = {inj_index}")
            # get the image embeddings
            image_features = self.image_encoder(image, output_hidden_states=True)
            # print("success")
            image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16]]
            image_embeddings = [emb.detach() for emb in image_embeddings]
            inj_embedding = self.mapper(image_embeddings) # [1,5,4096]
            
            # img_embeddings_mean = torch.mean(torch.abs(inj_embedding),-1)
            # img_embeddings_var = torch.var(torch.abs(inj_embedding),-1,unbiased=False)
            # print(f"img_embedding_mean:{img_embeddings_mean}")
            # print(f"img_embedding_var:{img_embeddings_var}")


            emb_length = inj_embedding.shape[1]
            for bsz,idx_list in enumerate(inj_index):
                # print(idx_list)
                start_idx_list = []
                for i in range(len(original_ids[bsz]) - len(idx_list) + 1):
                    # print(original_ids[bsz][i:i + len(idx_list)])
                    if torch.equal(original_ids[bsz][i:i + len(idx_list)], idx_list):
                        start_idx_list.append(i) # 找到special token对应的起始位置
                # print(f"start idx list = {start_idx_list}")

                # word_start_idx = start_idx_list[0]
                # word_end_idx = word_start_idx + len(idx_list) -1
                # word_embedding = original_outputs[:,word_start_idx:word_end_idx+1,:]
                # word_embeddings_mean = torch.mean(torch.abs(word_embedding),-1)
                # word_embeddings_var = torch.var(torch.abs(word_embedding),-1,unbiased=False)
                # print(f"word_embedding_mean:{word_embeddings_mean}")
                # print(f"word_embedding_var:{word_embeddings_var}")


                for start_idx in start_idx_list:
                    end_idx = start_idx + len(idx_list) -1
                    if len(idx_list) > emb_length:
                        lll = original_outputs[bsz,end_idx+1:].shape[0]
                        try:
                            inj_outputs[bsz,start_idx+emb_length:] = torch.cat([original_outputs[bsz,end_idx+1:end_idx+1+lll],original_ids[bsz,-(len(idx_list)-emb_length):]],dim=0)
                        except:
                            print(f'Index Error: point1, {start_idx}, {end_idx}, {inj_outputs[bsz, start_idx+emb_length:].size()}, {original_outputs[bsz, end_idx+1:end_idx+1+lll].size()}, {original_outputs[bsz, -(len(idx_list) - emb_length):].size()}')
                    else:
                        lll = inj_outputs[bsz,start_idx+emb_length:].shape[0]
                        try:
                            inj_outputs[bsz,start_idx+emb_length:] = original_outputs[bsz,end_idx+1:end_idx+1+lll]
                        except:
                            print(f'Index Error: point2, {start_idx}, {end_idx}, {inj_outputs[bsz, start_idx+emb_length:].size()}, {original_outputs[bsz, end_idx+1:end_idx+1+lll].size()}')
                    try:
                        inj_outputs[bsz,start_idx:start_idx+emb_length] = inj_embedding[bsz]
                    except:
                        remain_length = inj_outputs[bsz,start_idx:start_idx+emb_length].size(0)
                        inj_outputs[bsz,start_idx:start_idx+emb_length] = inj_embedding[bsz,:remain_length]
            # print(f"original id = {original_ids}")
            # print(f"text embedding = {original_outputs}")
            # print(f"image embedding = {inj_embedding}")
            # print(f"text_image embedding = {inj_outputs}")
            # print(f"text embedding == text_image embedding : {torch.equal(original_outputs,inj_outputs)}")
            # print(f"image embedding == text_image embessing:{torch.equal(inj_embedding[bsz],inj_outputs[bsz,3:8])}")
        return inj_outputs

    def encode(self, inputs):
        return self(inputs)

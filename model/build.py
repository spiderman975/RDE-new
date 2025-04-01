from . import objectives

#from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn 
import torch.nn.functional as F

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        #self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        #self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    '''
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
    '''

    def encode_visual(self, visual_input):
        """
        根据输入维度判断调用图像编码或视频编码：
        - 如果输入为 5D 张量 [B, T, C, H, W]，则调用 encode_video；
        - 如果输入为 4D 张量 [B, C, H, W]，则调用 encode_image，并取 CLS token。
        """
        if visual_input.dim() == 5:  # 视频输入
            x, atten = self.base_model.encode_video(visual_input)
            return x.float(), atten
        else:  # 图像输入
            x, atten = self.base_model.encode_image(visual_input)
            # 假设图像模型返回序列，取第一个 token 作为全局表示
            return x[:, 0, :].float(), atten

    '''
    def encode_video_tse(self, video):
        x, atten = self.base_model.encode_video(video)
        v_tse_f = self.visul_emb_layer(x, atten)
        return v_tse_f.float()
    '''

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()


    '''
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()
    '''

    def compute_per_loss(self, batch):
        videos = batch['videos']
        caption_ids = batch['caption_ids']
        # 这里调用 base_model(videos, caption_ids) 即同时对视频和文本进行编码
        video_feats, atten_v, text_feats, atten_t = self.base_model(videos, caption_ids)
        #print("compute_per_loss - video_feats.shape:", video_feats.shape)
        #print("compute_per_loss - atten_v is None?", atten_v is None)
        #print("compute_per_loss - text_feats.shape:", text_feats.shape)
        # 检查 caption_ids 的最大值（用于索引）
        #print("caption_ids max:", caption_ids.max().item())
        # 因为视频版 CLIP 的 encode_video 已经返回全局表示，不再需要 [:,0,:]
        i_feats = video_feats.float()

        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        #i_tse_f = self.visul_emb_layer(video_feats, atten_v)
        #t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)

        '''
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        '''

        #return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB
        return lossA.detach().cpu(), simsA

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        # 用视频数据替换图像数据
        videos = batch['videos']
        caption_ids = batch['caption_ids']
        video_feats, atten_v, text_feats, atten_t = self.base_model(videos, caption_ids)
        i_feats = video_feats.float()

        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        #i_tse_f = self.visul_emb_layer(video_feats, atten_v)
        #t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device) 
     
        loss1 = objectives.compute_rbs(i_feats, t_feats, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1})
        #ret.update({'tse_loss':loss2})
  
        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model

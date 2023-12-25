import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from typing import Callable, Any, Optional, Tuple, List
from sentence_transformers import SentenceTransformer
import timm

# from fusion import OurFusion


class StickerClassify(nn.Module):
    def __init__(self, args):
        super(StickerClassify, self).__init__()
        self.sticker_height = args.sticker_height
        self.sticker_width = args.sticker_width
        self.batch_size = args.batch_size
        # image encoder
        self.sticker_encoder = timm.create_model(args.model, pretrained=True, num_classes=0, global_pool='avg').requires_grad_(True)
        for p in self.sticker_encoder.parameters():
            p.requires_grad = True
        self.hidden_dim = args.hidden_dim
        # self.our_fusion = OurFusion(hidden_dim=args.hidden_dim).requires_grad_(True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_head = ProjectionHead(384,self.hidden_dim).requires_grad_(True)
        feature_dim = self.sticker_encoder.feature_info[-1]['num_chs']
        self.sticker_head = ProjectionHead(feature_dim, self.hidden_dim).requires_grad_(True)
        self.emoji_head = ProjectionHead(feature_dim, 500).requires_grad_(True)
        self.loss = nn.CrossEntropyLoss()
    def forward(
        self,
        text_embeddings,  # str: [batch, 768]
        sticker_pix,  # [batch, cand_num, height, weight, 3]
    ):
        sticker_embeddings, emoji_logits = self.sticker_encode(sticker_pix)
        context_embeddings = self.context_head(text_embeddings)
        # sticker_encoded = F.normalize(sticker_encoded, p=2, dim=1)
        # enc_context = F.normalize(enc_context, p=2, dim=1)

        # sticker logits: calculate the cosine similarity between sticker and context

        # Calculating the Loss
        logits = (context_embeddings @ sticker_embeddings.T)
        
        # targets = torch.diag(torch.ones(logits.shape[0])).to(self.device)
        targets = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.loss(logits, targets)
    
        # images_similarity = sticker_embeddings @ sticker_embeddings.T
        # texts_similarity = context_embeddings @ context_embeddings.T
        # targets = F.softmax(
        #     (images_similarity + texts_similarity) / 2 , dim=-1
        # )
        # texts_loss = cross_entropy(logits, targets, reduction='none')
        # images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        # loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)

        return loss, sticker_embeddings, context_embeddings, emoji_logits
    
    def sticker_encode(self, sticker_pix):
        sticker_pix = sticker_pix.to(self.device)
        sticker_pix = sticker_pix.float()
        sticker_pix = sticker_pix.permute(0, 3, 1, 2)
        sticker_encoded = self.sticker_encoder(sticker_pix)
        # emoji_logits: [batch=32, emoji_vocab_size=500]
        # sticker_encoded: [batch=32, inception_hidden=2048]
        sticker_embeddings = self.sticker_head(sticker_encoded)
        emoji_logits = self.emoji_head(sticker_encoded)
        return sticker_embeddings, emoji_logits

    def emoji_loss(self, emoji_logits, sticker_emoji):
        # emoji_classification_loss
        emoji_logits = emoji_logits.float().to(self.device)
        sticker_emoji = sticker_emoji.long().to(self.device)
        emoji_loss = self.loss(emoji_logits, sticker_emoji)
        # no need to use regularization because we use Adam optimizer with weight_decay
        return emoji_loss.mean()
    
# def cross_entropy(preds, targets, reduction='none'):
#     log_softmax = nn.LogSoftmax(dim=-1)
#     loss = (-targets * log_softmax(preds)).sum(1)
#     if reduction == "none":
#         return loss
#     elif reduction == "mean":
#         return loss.mean()

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0.2
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
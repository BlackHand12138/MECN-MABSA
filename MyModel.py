import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import numpy as np
from pdb import set_trace as stop
import torch.nn.functional as F
from Gated_Fusion import gatedFusion
from RMT import VisionRetentionChunk, RetNetRelPos2d


class MultiHeadAttention(nn.Module):
    #实现跨模态的注意力交互（如文本与图像的融合）
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False, attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None, dec_self=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        if hasattr(self, 'dropout2'):
            q = self.dropout2(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        if hasattr(self, 'fc'):
            output = self.fc(output)

        if hasattr(self, 'dropout'):
            output = self.dropout(output)

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    #计算注意力分数并加权融合 Value 向量
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e6)

        if stop_sig:
            print('**')
            stop()

        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()


class SimpleBertModel(nn.Module):
    #定义模型各组件，包括 BERT 编码器、GCN 层、注意力模块和分类器
    def __init__(self, opt):
        super(SimpleBertModel, self).__init__()

        #加载预训练的 BERT 模型作为文本编码器
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.max_len = opt.MAX_LEN
        self.layers = 3

        # 消融实验开关：每次实验只开启一项，保持其他模块不变。
        self.without_bidirectional_mhca = getattr(opt, 'without_bidirectional_mhca', False)
        self.without_rmt = getattr(opt, 'without_rmt', False)
        self.without_gated = getattr(opt, 'without_gated', False)
        self.without_global_text = getattr(opt, 'without_global_text', False)
        self.without_text_gcn = getattr(opt, 'without_text_gcn', False)
        self.without_cross_gcn = getattr(opt, 'without_cross_gcn', False)

        # Layer Normalization 层
        self.layernorm = nn.LayerNorm(self.bert.config.hidden_size)

        if not self.without_bidirectional_mhca:
            self.mulitHead_text_img = MultiHeadAttention(3, self.bert.config.hidden_size, self.bert.config.hidden_size,
                                                        self.bert.config.hidden_size)
            self.mulitHead_img_text = MultiHeadAttention(3, self.bert.config.hidden_size, self.bert.config.hidden_size,
                                                        self.bert.config.hidden_size)

        # 添加视觉保留模块
        self.retention = VisionRetentionChunk(embed_dim=768, num_heads=8)
        self.retention_pos = RetNetRelPos2d(embed_dim=768, num_heads=8, initial_value=1, heads_range=8)

        # gcn layer GCN 的权重矩阵列表
        self.W1 = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert.config.hidden_size
            self.W1.append(nn.Linear(input_dim, input_dim))

        #另一个 GCN 权重矩阵列表，用于跨模态图卷积
        self.W2 = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert.config.hidden_size
            self.W2.append(nn.Linear(input_dim, input_dim))

        #Dropout 层
        self.bert_drop = nn.Dropout(0.1)
        self.gcn_drop = nn.Dropout(0.1)

        #全连接层
        self.linear_global = nn.Linear(768 * 2, 768)
        self.linear_local = nn.Linear(768 * 2, 768)

        self.gated_fusion = gatedFusion(dim=768)

        self.linear_temp = nn.Linear(768, 768)

        #outMLP|最终分类器
        self.outMLP = nn.Linear(768, opt.NUM_CLASSES)
        self.outMLP_temp = nn.Linear(768, opt.NUM_CLASSES)

    def _fuse_text_with_image(self, text_feat, vit_feature):
        """将图像信息注入文本特征。"""
        if self.without_bidirectional_mhca:
            image_global = vit_feature.mean(dim=1, keepdim=True)
            return self.layernorm(text_feat + image_global)
        return self.mulitHead_text_img(text_feat, vit_feature, vit_feature)[0]

    def _fuse_image_with_global_text(self, vit_feature, combined_pooled):
        """将全局文本信息注入图像特征。"""
        if self.without_bidirectional_mhca:
            return self.layernorm(vit_feature + combined_pooled)
        return self.mulitHead_img_text(vit_feature, combined_pooled, combined_pooled)[0]

    def _run_gcn(self, context_asp_adj_matrix, context_asp_adj_matrix_text_img,
                 tmps, tmps_text_img, denom_dep, denom_dep_text_img):
        """堆叠式 GCN 前向传播。

        修复点：
        1. 每一层 GCN 的输入为上一层的输出（原先每层都用原始 tmps，多层未真正堆叠）。
        2. 每层输出经过 gcn_drop 防止过拟合。
        3. 消融时直接返回原始特征，保持张量形状一致。
        """
        outputs_dep = tmps
        outputs_dep_text_img = tmps_text_img

        for l in range(self.layers):
            # ************GCN_text*************
            if not self.without_text_gcn:
                Ax_dep = context_asp_adj_matrix.bmm(outputs_dep)
                AxW_dep = self.W1[l](Ax_dep)
                AxW_dep = AxW_dep / denom_dep
                outputs_dep = self.gcn_drop(F.relu(AxW_dep))

            # ************GCN_text_img*************
            if not self.without_cross_gcn:
                Ax_dep_text_img = context_asp_adj_matrix_text_img.bmm(outputs_dep_text_img)
                AxW_dep_text_img = self.W2[l](Ax_dep_text_img)
                AxW_dep_text_img = AxW_dep_text_img / denom_dep_text_img
                outputs_dep_text_img = self.gcn_drop(F.relu(AxW_dep_text_img))

        return outputs_dep, outputs_dep_text_img

    def extract_gated_fusion_features(self, inputs):
        """提取门控融合后的特征（更有意义的可视化特征）"""
        input_ids, attention_mask, vit_feature, transformer_mask, target_input_ids, \
            target_attention_mask, target_mask, text_length, word_length, tran_indices, \
            context_asp_adj_matrix, globel_input_id, globel_mask, face_input_ids, face_mask = inputs

        # 分离 [CLS] 标记和图像特征
        cls_token = vit_feature[:, :1, :]
        image_tokens = vit_feature[:, 1:, :]

        # 处理图像特征：应用视觉保留机制
        batch_size, seq_len, dim = image_tokens.shape
        h = w = 14
        if self.without_rmt:
            vit_feature_processed = vit_feature
        elif seq_len == h * w:
            image_tokens_reshaped = image_tokens.view(batch_size, h, w, dim)
            rel_pos = self.retention_pos((h, w), chunkwise_recurrent=True)
            image_tokens_processed = self.retention(image_tokens_reshaped, rel_pos, chunkwise_recurrent=True)
            image_tokens_processed = image_tokens_processed.view(batch_size, h * w, dim)
            vit_feature_processed = torch.cat([cls_token, image_tokens_processed], dim=1)
        else:
            vit_feature_processed = vit_feature

        vit_feature = vit_feature_processed

        # 文本模态部分
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state
        text_feat = self.layernorm(text_feat)
        text_feat = self.bert_drop(text_feat)

        # 文本-图像跨模态融合
        text_include_img = self._fuse_text_with_image(text_feat, vit_feature)
        similarity = torch.cosine_similarity(text_include_img.unsqueeze(2), text_include_img.unsqueeze(1), dim=-1)

        if self.without_global_text:
            face_include_caption_avg = self.layernorm(vit_feature).mean(dim=1)
        else:
            batch_size = globel_input_id.size(0)
            device = globel_input_id.device
            cls_id = self.tokenizer.cls_token_id
            sep_id = self.tokenizer.sep_token_id
            new_input_ids = torch.cat([
                torch.full((batch_size, 1), cls_id, dtype=torch.long, device=device),
                globel_input_id,
                torch.full((batch_size, 1), sep_id, dtype=torch.long, device=device),
                face_input_ids,
                torch.full((batch_size, 1), sep_id, dtype=torch.long, device=device)
            ], dim=1)
            new_attention_mask = torch.ones_like(new_input_ids)

            combined_outputs = self.bert(input_ids=new_input_ids, attention_mask=new_attention_mask)
            combined_pooled = combined_outputs.pooler_output
            combined_pooled = self.layernorm(combined_pooled)
            combined_pooled = self.bert_drop(combined_pooled)

            combined_pooled = combined_pooled.unsqueeze(1)
            face_include_caption = self._fuse_image_with_global_text(vit_feature, combined_pooled)
            face_include_caption_avg = face_include_caption.mean(dim=1)

        # GCN处理
        gcn_text_feat = text_feat[:, 1:, :]
        gcn_text_img_feat = text_include_img[:, 1:, :]
        tmps = torch.zeros((input_ids.size(0), self.max_len, 768), dtype=torch.float32).to(input_ids.device)
        tmps_text_img = torch.zeros((input_ids.size(0), self.max_len, 768), dtype=torch.float32).to(input_ids.device)

        for i, spans in enumerate(tran_indices):
            true_len = word_length[i]
            nums = 0
            for j, span in enumerate(spans):
                if nums == true_len:
                    break
                nums += 1
                tmps[i, j + 1] = torch.sum(gcn_text_feat[i, span[0]:span[1]], 0)
                tmps_text_img[i, j + 1] = torch.sum(gcn_text_img_feat[i, span[0]:span[1]], 0)

        context_asp_adj_matrix_text_img = torch.mul(context_asp_adj_matrix, similarity)
        denom_dep = context_asp_adj_matrix.sum(2).unsqueeze(2) + 1
        denom_dep_text_img = context_asp_adj_matrix_text_img.sum(2).unsqueeze(2) + 1

        # 堆叠式 GCN 前向
        outputs_dep, outputs_dep_text_img = self._run_gcn(
            context_asp_adj_matrix, context_asp_adj_matrix_text_img,
            tmps, tmps_text_img, denom_dep, denom_dep_text_img
        )

        # 聚合方面词特征
        asp_wn = target_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = target_mask.unsqueeze(-1).repeat(1, 1, 768)
        gcn_out_text = (outputs_dep * aspect_mask).sum(dim=1) / asp_wn
        gcn_out_text_img = (outputs_dep_text_img * aspect_mask).sum(dim=1) / asp_wn

        # 局部特征融合
        local_feat = self.linear_local(torch.cat([gcn_out_text, gcn_out_text_img], dim=1))

        # 门控融合
        if self.without_gated:
            fused_feat = self.linear_global(torch.cat([face_include_caption_avg, local_feat], dim=1))
        else:
            fused_feat = self.gated_fusion(face_include_caption_avg, local_feat)
        fused_feat = fused_feat + local_feat
        fused_feat = torch.relu(fused_feat)

        return fused_feat

    def forward(self, inputs):
        input_ids, attention_mask, vit_feature, transformer_mask, target_input_ids, target_attention_mask, \
            target_mask, text_length, word_length, tran_indices, context_asp_adj_matrix, globel_input_id, \
            globel_mask, face_input_ids, face_mask = inputs

        # 分离 [CLS] 标记和图像特征
        cls_token = vit_feature[:, :1, :]
        image_tokens = vit_feature[:, 1:, :]

        # 处理图像特征：应用视觉保留机制
        batch_size, seq_len, dim = image_tokens.shape
        h = w = 14
        if self.without_rmt:
            vit_feature_processed = vit_feature
        elif seq_len == h * w:
            image_tokens_reshaped = image_tokens.view(batch_size, h, w, dim)
            rel_pos = self.retention_pos((h, w), chunkwise_recurrent=True)
            image_tokens_processed = self.retention(image_tokens_reshaped, rel_pos, chunkwise_recurrent=True)
            image_tokens_processed = image_tokens_processed.view(batch_size, h * w, dim)
            vit_feature_processed = torch.cat([cls_token, image_tokens_processed], dim=1)
        else:
            vit_feature_processed = vit_feature
            print('没有运行RMT')

        vit_feature = vit_feature_processed

        #文本模态部分 使用 BERT 编码输入文本
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state
        text_feat = self.layernorm(text_feat)
        text_feat = self.bert_drop(text_feat)

        # 文本-图像跨模态融合
        text_include_img = self._fuse_text_with_image(text_feat, vit_feature)
        similarity = torch.cosine_similarity(text_include_img.unsqueeze(2), text_include_img.unsqueeze(1), dim=-1)

        if self.without_global_text:
            face_include_caption_avg = self.layernorm(vit_feature).mean(dim=1)
        else:
            ###############################################################################################################
            #  构建[CLS] + 字幕 + [SEP] + 人脸描述 + [SEP]序列
            batch_size = globel_input_id.size(0)
            device = globel_input_id.device
            cls_id = self.tokenizer.cls_token_id
            sep_id = self.tokenizer.sep_token_id
            new_input_ids = torch.cat([
                torch.full((batch_size, 1), cls_id, dtype=torch.long, device=device),
                globel_input_id,
                torch.full((batch_size, 1), sep_id, dtype=torch.long, device=device),
                face_input_ids,
                torch.full((batch_size, 1), sep_id, dtype=torch.long, device=device)
            ], dim=1)
            new_attention_mask = torch.ones_like(new_input_ids)

            combined_outputs = self.bert(input_ids=new_input_ids, attention_mask=new_attention_mask)
            combined_pooled = combined_outputs.pooler_output
            combined_pooled = self.layernorm(combined_pooled)
            combined_pooled = self.bert_drop(combined_pooled)
            #################################################################################################################
            # 图像-全局文本跨模态融合
            combined_pooled = combined_pooled.unsqueeze(1)
            face_include_caption = self._fuse_image_with_global_text(vit_feature, combined_pooled)
            face_include_caption_avg = face_include_caption.mean(dim=1)

        #GCN准备步骤，将分词聚合在一起
        gcn_text_feat = text_feat[:, 1:, :]
        gcn_text_img_feat = text_include_img[:, 1:, :]
        tmps = torch.zeros((input_ids.size(0), self.max_len, 768), dtype=torch.float32).to(input_ids.device)
        tmps_text_img = torch.zeros((input_ids.size(0), self.max_len, 768), dtype=torch.float32).to(input_ids.device)

        for i, spans in enumerate(tran_indices):
            true_len = word_length[i]
            nums = 0
            for j, span in enumerate(spans):
                if nums == true_len:
                    break
                nums += 1
                tmps[i, j + 1] = torch.sum(gcn_text_feat[i, span[0]:span[1]], 0)
                tmps_text_img[i, j + 1] = torch.sum(gcn_text_img_feat[i, span[0]:span[1]], 0)

        #构建跨模态邻接矩阵
        context_asp_adj_matrix_text_img = torch.mul(context_asp_adj_matrix, similarity)
        denom_dep = context_asp_adj_matrix.sum(2).unsqueeze(2) + 1
        denom_dep_text_img = context_asp_adj_matrix_text_img.sum(2).unsqueeze(2) + 1

        # 堆叠式 GCN 前向（修复：每层输入为上一层输出）
        outputs_dep, outputs_dep_text_img = self._run_gcn(
            context_asp_adj_matrix, context_asp_adj_matrix_text_img,
            tmps, tmps_text_img, denom_dep, denom_dep_text_img
        )

        #聚合方面词特征
        asp_wn = target_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = target_mask.unsqueeze(-1).repeat(1, 1, 768)
        gcn_out_text = (outputs_dep * aspect_mask).sum(dim=1) / asp_wn
        gcn_out_text_img = (outputs_dep_text_img * aspect_mask).sum(dim=1) / asp_wn

        #拼接文本和跨模态 GCN 特征，通过全连接层
        local_feat = self.linear_local(torch.cat([gcn_out_text, gcn_out_text_img], dim=1))

        # 门控融合
        if self.without_gated:
            fused_feat = self.linear_global(torch.cat([face_include_caption_avg, local_feat], dim=1))
        else:
            fused_feat = self.gated_fusion(face_include_caption_avg, local_feat)

        fused_feat = fused_feat + local_feat

        fused_feat = torch.relu(fused_feat)

        return self.outMLP(fused_feat)

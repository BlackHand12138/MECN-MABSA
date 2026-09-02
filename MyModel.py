import torch
from torch import nn
from  transformers import BertModel,BertTokenizer
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
            # self.dropout2 = nn.Dropout(dropout2)
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
            # self.softmax = BottleSoftmax()
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            # attn = attn.masked_fill(attn_mask, -np.inf)
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
    def __init__(self,opt):
        super(SimpleBertModel, self).__init__()



        #加载预训练的 BERT 模型作为文本编码器
        self.bert = BertModel.from_pretrained("bert-base-uncased")              #用于编码输入文本
        #self.caption_text_bert = BertModel.from_pretrained("bert-base-uncased") #用于编码全局文本（如字幕）
        #self.face_text_bert = BertModel.from_pretrained("bert-base-uncased")    #用于编码人脸文本
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")     #新增tokenizer属性



        self.max_len = opt.MAX_LEN  #定义文本序列的最大长度
        self.layers = 3            #定义GCN（图卷积网络）的层数
        # 消融实验开关：每次实验只开启一项，保持其他模块不变。
        self.without_bidirectional_mhca = getattr(opt, 'without_bidirectional_mhca', False)
        self.without_rmt = getattr(opt, 'without_rmt', False)
        self.without_gated = getattr(opt, 'without_gated', False)
        self.without_global_text = getattr(opt, 'without_global_text', False)
        self.without_text_gcn = getattr(opt, 'without_text_gcn', False)
        self.without_cross_gcn = getattr(opt, 'without_cross_gcn', False)

        # Layer Normalization 层，用于标准化特征（维度为 BERT 的隐藏层大小，768）
        self.layernorm = nn.LayerNorm(self.bert.config.hidden_size)

        if not self.without_bidirectional_mhca:
            #文本指导图像多头注意力，定义跨模态多头注意力模块 【8个注意力头，输入维度：768（BERT 隐藏层大小），输出维度：768】
            self.mulitHead_text_img = MultiHeadAttention(3, self.bert.config.hidden_size, self.bert.config.hidden_size,
                                                self.bert.config.hidden_size)
            # 图像指导文本多头注意力，定义跨模态多头注意力模块 【8个注意力头，输入维度：768（BERT 隐藏层大小），输出维度：768】
            self.mulitHead_img_text = MultiHeadAttention(3, self.bert.config.hidden_size, self.bert.config.hidden_size,
                                                         self.bert.config.hidden_size)

        # 添加视觉保留模块
        self.retention = VisionRetentionChunk(embed_dim=768, num_heads=8)  # 图像特征维度为768，使用8个头
        self.retention_pos = RetNetRelPos2d(embed_dim=768, num_heads=8, initial_value=1, heads_range=8)

        # gcn layer GCN 的权重矩阵列表，用于图卷积操作，每个 GCN 层是一个线性变换（输入维度等于输出维度，768
        self.W1 = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert.config.hidden_size
            self.W1.append(nn.Linear(input_dim, input_dim))

        #另一个 GCN 权重矩阵列表，用于跨模态图卷积
        self.W2 = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert.config.hidden_size
            self.W2.append(nn.Linear(input_dim, input_dim))

        #Dropout 层，防止过拟合
        self.bert_drop = nn.Dropout(0.1)  #作用于 BERT 输出
        self.gcn_drop = nn.Dropout(0.1)   #作用于 GCN 输出

        #全连接层
        self.linear_global = nn.Linear(768 * 2,768)  #融合全局特征（如图像和文本）
        self.linear_local = nn.Linear(768 * 2, 768)  #融合局部特征（如方面词和跨模态特征）

        self.gated_fusion = gatedFusion(dim=768)  # 替代拼接的融合方式

        # TODO
        self.linear_temp = nn.Linear(768, 768) #临时全连接层

        #outMLP|最终分类器（输入维度 768，输出类别数）
        self.outMLP = nn.Linear(768 ,opt.NUM_CLASSES)



        # TODO
        self.outMLP_temp = nn.Linear(768, opt.NUM_CLASSES) #临时分类器（输入维度 768）

    def _fuse_text_with_image(self, text_feat, vit_feature):
        """将图像信息注入文本特征。

        完整模型使用文本查询图像的 MHCA；消融模型使用无参数的图像均值加法融合。
        """
        if self.without_bidirectional_mhca:
            image_global = vit_feature.mean(dim=1, keepdim=True)
            return self.layernorm(text_feat + image_global)
        return self.mulitHead_text_img(text_feat, vit_feature, vit_feature)[0]

    def _fuse_image_with_global_text(self, vit_feature, combined_pooled):
        """将全局文本信息注入图像特征。

        完整模型使用图像查询全局文本的 MHCA；消融模型使用无参数的加法融合。
        """
        if self.without_bidirectional_mhca:
            return self.layernorm(vit_feature + combined_pooled)
        return self.mulitHead_img_text(vit_feature, combined_pooled, combined_pooled)[0]

    def extract_gated_fusion_features(self, inputs):
        """提取门控融合后的特征（更有意义的可视化特征）"""
        input_ids, attention_mask, vit_feature, transformer_mask, target_input_ids, \
            target_attention_mask, target_mask, text_length, word_length, tran_indices, \
            context_asp_adj_matrix, globel_input_id, globel_mask, face_input_ids, face_mask = inputs

        # 运行完整的前向传播到门控融合阶段
        # 分离 [CLS] 标记和图像特征
        cls_token = vit_feature[:, :1, :]
        image_tokens = vit_feature[:, 1:, :]

        # 处理图像特征：应用视觉保留机制
        batch_size, seq_len, dim = image_tokens.shape
        h = w = 14
        if self.without_rmt:
            # w/o RMT：保留原始 ViT 的 [CLS] 和 patch 特征。
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
            # w/o GlobalText：不使用图像描述和面部情绪描述，保留纯图像全局表示。
            face_include_caption_avg = self.layernorm(vit_feature).mean(dim=1)
        else:
            # 全局特征（字幕+人脸）
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

            # 图像-全局文本跨模态融合
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

        for l in range(self.layers):
            # GCN_text
            if not self.without_text_gcn:
                Ax_dep = context_asp_adj_matrix.bmm(tmps)
                AxW_dep = self.W1[l](Ax_dep)
                AxW_dep = AxW_dep / denom_dep
                gAxW_dep = F.relu(AxW_dep)

            # GCN_text_img
            if not self.without_cross_gcn:
                Ax_dep_text_img = context_asp_adj_matrix_text_img.bmm(tmps_text_img)
                AxW_dep_text_img = self.W2[l](Ax_dep_text_img)
                AxW_dep_text_img = AxW_dep_text_img / denom_dep_text_img
                gAxW_dep_text_img = F.relu(AxW_dep_text_img)

        # GCN 消融时使用卷积前的特征，保留对应信息分支和张量形状。
        outputs_dep = tmps if self.without_text_gcn else gAxW_dep
        outputs_dep_text_img = tmps_text_img if self.without_cross_gcn else gAxW_dep_text_img

        # 聚合方面词特征
        asp_wn = target_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = target_mask.unsqueeze(-1).repeat(1, 1, 768)
        gcn_out_text = (outputs_dep * aspect_mask).sum(dim=1) / asp_wn
        gcn_out_text_img = (outputs_dep_text_img * aspect_mask).sum(dim=1) / asp_wn

        # 局部特征融合
        local_feat = self.linear_local(torch.cat([gcn_out_text, gcn_out_text_img], dim=1))

        # 门控融合（这是我们想要可视化的特征）
        if self.without_gated:
            # w/o Gated：使用现有的拼接+线性映射替代自适应门控。
            fused_feat = self.linear_global(torch.cat([face_include_caption_avg, local_feat], dim=1))
        else:
            fused_feat = self.gated_fusion(face_include_caption_avg, local_feat)
        fused_feat = fused_feat + local_feat  # 残差连接
        fused_feat = torch.relu(fused_feat)

        return fused_feat  # 返回门控融合后的特征


    def forward(self,inputs):
        #input_ids：文本的 token ID。vit_feature：图像特征（来自 Vision Transformer）。context_asp_adj_matrix：方面词与上下文的邻接矩阵（图结构）。其他参数包括注意力掩码、目标文本、全局文本等。
        input_ids, attention_mask,vit_feature,transformer_mask,target_input_ids,target_attention_mask,target_mask,text_length,word_length,tran_indices,context_asp_adj_matrix,globel_input_id,globel_mask,face_input_ids,face_mask= inputs


        # 分离 [CLS] 标记和图像特征
        cls_token = vit_feature[:, :1, :]  # [batch_size, 1, dim]
        image_tokens = vit_feature[:, 1:, :]  # [batch_size, 196, dim]

        # 处理图像特征：应用视觉保留机制
        batch_size, seq_len, dim = image_tokens.shape
        # 图像块为14x14（因为14*14=196）
        h = w = 14
        if self.without_rmt:
            # w/o RMT：保留原始 ViT 的 [CLS] 和 patch 特征。
            vit_feature_processed = vit_feature
        elif seq_len == h * w:
            image_tokens_reshaped = image_tokens.view(batch_size, h, w, dim)  # 重塑为 [b, h, w, c]
            # 生成相对位置编码
            rel_pos = self.retention_pos((h, w), chunkwise_recurrent=True)
            # 应用视觉保留模块
            image_tokens_processed = self.retention(image_tokens_reshaped, rel_pos, chunkwise_recurrent=True)
            # 重塑回序列格式 [b, h*w, dim]
            image_tokens_processed = image_tokens_processed.view(batch_size, h * w, dim)

            # 将 [CLS] 标记与处理后的图像特征拼接
            vit_feature_processed = torch.cat([cls_token, image_tokens_processed], dim=1)
        else:
            # 如果序列长度不是196，则跳过处理
            vit_feature_processed = vit_feature
            print('没有运行RMT')

        # 使用处理后的图像特征
        vit_feature = vit_feature_processed


        #文本模态部分 使用 BERT 编码输入文本
        outputs = self.bert(input_ids = input_ids,attention_mask = attention_mask)
        #pooler_output : 序列的第一个token 的最后一层的隐藏状态
        text_feat =  outputs.last_hidden_state
        text_feat = self.layernorm(text_feat)  #经过 LayerNorm 和 Dropout
        text_feat = self.bert_drop(text_feat)


        # 文本-图像跨模态融合（完整模型使用 MHCA，消融模型使用均值加法融合）
        text_include_img = self._fuse_text_with_image(text_feat, vit_feature)
        similarity = torch.cosine_similarity(text_include_img.unsqueeze(2), text_include_img.unsqueeze(1), dim=-1) #计算文本 token 之间的余弦相似度，用于后续图构建

        if self.without_global_text:
            # w/o GlobalText：不使用图像描述和面部情绪描述，保留纯图像全局表示。
            face_include_caption_avg = self.layernorm(vit_feature).mean(dim=1)
        else:
            ###############################################################################################################
            #  构建[CLS] + 字幕 + [SEP] + 人脸描述 + [SEP]序列
            batch_size = globel_input_id.size(0)
            device = globel_input_id.device
            cls_id = self.tokenizer.cls_token_id
            sep_id = self.tokenizer.sep_token_id
            new_input_ids = torch.cat([
                torch.full((batch_size, 1), cls_id, dtype=torch.long, device=device),  # [CLS]
                globel_input_id,
                torch.full((batch_size, 1), sep_id, dtype=torch.long, device=device),  # [SEP]
                face_input_ids,
                torch.full((batch_size, 1), sep_id, dtype=torch.long, device=device)  # [SEP]
            ], dim=1)
            # 新增：构建对应的attention_mask
            new_attention_mask = torch.ones_like(new_input_ids)

            # 新增：送入BERT编码
            combined_outputs = self.bert(input_ids=new_input_ids, attention_mask=new_attention_mask)
            combined_pooled = combined_outputs.pooler_output  # [batch_size, 768]
            combined_pooled = self.layernorm(combined_pooled)  # 层归一化
            combined_pooled = self.bert_drop(combined_pooled)  # Dropout
            #################################################################################################################
            # 图像-全局文本跨模态融合
            # 扩展 combined_pooled 为三维张量
            combined_pooled = combined_pooled.unsqueeze(1)   # (batch_size, 1, 768)
            face_include_caption = self._fuse_image_with_global_text(vit_feature, combined_pooled)
            face_include_caption_avg = face_include_caption.mean(dim=1) # 计算平均池化 [batch_size, 768]





        #GCN准备步骤，将分词聚合在一起
        gcn_text_feat = text_feat[:,1:,:]
        gcn_text_img_feat = text_include_img[:, 1:, :]    #去除 [CLS] token（索引 0），保留实际文本 token
        tmps = torch.zeros((input_ids.size(0), self.max_len, 768),dtype=torch.float32).to(input_ids.device)
        tmps_text_img = torch.zeros((input_ids.size(0), self.max_len, 768), dtype=torch.float32).to(input_ids.device) #初始化临时张量，用于存储聚合后的 GCN 特征
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
        context_asp_adj_matrix_text_img = torch.mul(context_asp_adj_matrix,similarity)  #结合余弦相似度调整原始邻接矩阵
        denom_dep = context_asp_adj_matrix.sum(2).unsqueeze(2) + 1                      #邻接矩阵的归一化因子
        denom_dep_text_img = context_asp_adj_matrix_text_img.sum(2).unsqueeze(2) + 1

        #GCN 图卷积操作；每层 GCN 对邻接矩阵和特征进行消息传递;
        for l in range(self.layers):
            # ************GCN_text*************
            if not self.without_text_gcn:
                Ax_dep = context_asp_adj_matrix.bmm(tmps)  #Ax_dep：邻接矩阵与特征的乘积
                AxW_dep = self.W1[l](Ax_dep)               #W1[l]：可学习权重矩阵
                AxW_dep = AxW_dep / denom_dep
                gAxW_dep = F.relu(AxW_dep)                 #归一化后通过 ReLU 激活

            # ************GCN_text_img*************
            if not self.without_cross_gcn:
                Ax_dep_text_img = context_asp_adj_matrix_text_img.bmm(tmps_text_img)
                AxW_dep_text_img = self.W2[l](Ax_dep_text_img)
                AxW_dep_text_img = AxW_dep_text_img / denom_dep_text_img
                gAxW_dep_text_img = F.relu(AxW_dep_text_img)

        #保留最后一层 GCN 的输出
        # GCN 消融时使用卷积前的特征，保留对应信息分支和张量形状。
        outputs_dep = tmps if self.without_text_gcn else gAxW_dep
        outputs_dep_text_img = tmps_text_img if self.without_cross_gcn else gAxW_dep_text_img

        #聚合方面词特征
        asp_wn = target_mask.sum(dim=1).unsqueeze(-1)              #标记方面词的位置
        aspect_mask = target_mask.unsqueeze(-1).repeat(1, 1, 768)
        gcn_out_text = (outputs_dep * aspect_mask).sum(dim=1) / asp_wn    #对 GCN 输出按方面词位置加权求和
        gcn_out_text_img = (outputs_dep_text_img * aspect_mask).sum(dim=1) / asp_wn



        #target cls标记 带图像信息的target  文本中gcn方面词  带图像特征中gcn方面词
        local_feat = self.linear_local(torch.cat([gcn_out_text, gcn_out_text_img], dim=1))   #拼接文本和跨模态 GCN 特征，通过全连接层

        #原始拼接代码
        #local_globa_feat = self.linear_global(torch.cat([face_include_caption_avg,local_feat],dim=1))

        # 门控融合
        if self.without_gated:
            # w/o Gated：使用现有的拼接+线性映射替代自适应门控。
            fused_feat = self.linear_global(torch.cat([face_include_caption_avg, local_feat], dim=1))
        else:
            fused_feat = self.gated_fusion(face_include_caption_avg, local_feat)  # [B,768]

        fused_feat = fused_feat + local_feat  # 局部特征的残差

        fused_feat = torch.relu(fused_feat)

        return self.outMLP(fused_feat)

        #return self.outMLP_temp(self.linear_temp(gcn_out_text))   #测试






import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score


import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax


from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax



import torch.nn as nn
from typing import Optional, Tuple, List
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.utils import softmax, scatter






class HypergraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        use_attention: bool = False,
        attention_mode: str = 'node',
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        assert attention_mode in ['node', 'edge']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            
            # 节点特征线性变换
            self.lin_node = Linear(in_channels, heads * out_channels, bias=False,
                                  weight_initializer='glorot')
            
            # 超边特征线性变换（如果提供了edge_dim）
            if edge_dim is not None:
                self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                      weight_initializer='glorot')
            else:
                self.lin_edge = None
            
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin_node = Linear(in_channels, out_channels, bias=False,
                                  weight_initializer='glorot')
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_node.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None,
                num_edges: Optional[int] = None) -> Tensor:
        


        # 1. 重新映射超边索引到连续范围
        edge_indices = hyperedge_index[1]
        unique_edge_indices, edge_map = torch.unique(edge_indices, return_inverse=True)
        mapped_hyperedge_index = torch.stack([hyperedge_index[0], edge_map], dim=0)
        
        # 计算实际的超边数量
        if num_edges is None:
            num_edges = unique_edge_indices.size(0)
        
        # 2. 处理超边权重 - 重新映射到新索引
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        else:
            # 确保权重与新索引对齐
            if hyperedge_weight.size(0) > num_edges:
                # 如果提供了更多权重，只取unique边对应的权重
                hyperedge_weight = hyperedge_weight[unique_edge_indices]
        
        # 3. 处理超边特征 - 重新映射到新索引
        if hyperedge_attr is not None:
            if hyperedge_attr.size(0) > num_edges:
                # 如果提供了更多特征，只取unique边对应的特征
                hyperedge_attr = hyperedge_attr[unique_edge_indices]
        
        num_nodes = x.size(0)
        
        # 4. 线性变换
        if self.use_attention:
            # 节点特征变换
            x = self.lin_node(x)
            # 超边特征变换
            if hyperedge_attr is not None:
                if self.lin_edge is not None:
                    hyperedge_attr = self.lin_edge(hyperedge_attr)
                else:
                    # 如果没有单独的超边变换层，使用节点特征变换层
                    hyperedge_attr = self.lin_node(hyperedge_attr)
        else:
            # 非注意力模式只变换节点特征
            x = self.lin_node(x)
        
        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            
            x_i = x[mapped_hyperedge_index[0]]
            x_j = hyperedge_attr[mapped_hyperedge_index[1]]
            
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            
            if self.attention_mode == 'node':
                alpha = softmax(alpha, mapped_hyperedge_index[1], num_nodes=num_edges)
            else:
                alpha = softmax(alpha, mapped_hyperedge_index[0], num_nodes=num_nodes)
            
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 5. 计算度矩阵 - 使用映射后的索引
        # 节点度
        D = scatter(hyperedge_weight[mapped_hyperedge_index[1]], 
                   mapped_hyperedge_index[0],
                   dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0
        
        # 超边度
        B = scatter(x.new_ones(mapped_hyperedge_index.size(1)), 
                   mapped_hyperedge_index[1],
                   dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0
        
        # 6. 传播 - 使用映射后的索引
        # 第一步：节点到超边的传播
        out = self.propagate(mapped_hyperedge_index, x=x, norm=B, alpha=alpha,
                           size=(num_nodes, num_edges))
        
        # 第二步：超边到节点的传播
        out = self.propagate(mapped_hyperedge_index.flip([0]), x=out, norm=D,
                           alpha=alpha, size=(num_edges, num_nodes))
        
        # 7. 处理多头注意力的输出
        if self.use_attention:
            if self.concat:
                out = out.view(-1, self.heads * self.out_channels)
            else:
                out = out.mean(dim=1)
        
        # 8. 添加偏置
        if self.bias is not None:
            out = out + self.bias
        
        return out, alpha

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Optional[Tensor] = None) -> Tensor:
        H = self.heads
        F = self.out_channels
        
        # 应用归一化
        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
        
        # 应用注意力权重（如果存在）
        if alpha is not None:
            out = alpha.view(-1, H, 1) * out
        
        return out




class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,batch_first=True, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.2
        )
        
        # 特征注意力层：为每个时间步的每个特征分配权重
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, input_dim)  # 输出每个特征的注意力分数
        )
        
        # 时间注意力层：为每个时间步分配权重
        self.time_attention = nn.Linear(hidden_dim, 1)
        
        # 分类层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: [batch, seq_len, hidden]
        
        # 计算时间注意力
        time_scores = self.time_attention(lstm_out)  # [batch, seq_len, 1]
        time_weights = F.softmax(time_scores.squeeze(-1), dim=-1)  # [batch, seq_len]
        
        # 计算特征注意力
        #batch_size, seq_len, hidden_dim = lstm_out.shape
        feature_scores = self.feature_attention(lstm_out)  # [batch, seq_len, input_dim]
        feature_weights = F.softmax(feature_scores, dim=-1)  # 在特征维度上归一化
        
        #使用特征注意力加权原始输入
        weighted_input = x * feature_weights  # [batch, seq_len, input_dim]
        
        #将加权后的输入再次通过LSTM
        time_context = torch.bmm(time_weights.unsqueeze(1), weighted_input).squeeze(1)
        
        
        #output = self.fc(time_context)
        output =time_context
        return output, (time_weights, feature_weights)# 时间重要性, 特征重要性 [batch, seq_len, input_dim]






class DynamicHyperGNN(nn.Module):
    """
    动态超图神经网络
    """
    def __init__(
        self,
        graph_node_features_dim: int = 25,           #节点特征维度
        graph_edge_features_dim: int = 25,           #超边特征维度
        graph_hidden_dim: int = 32,
        lstm_out_dim:int=32,
        num_timesteps: int = 2,                  #时间维度
        dropout_rate: float = 0.3,
        num_lstm_layers: int = 2,
        num_nodes:int=3000,
        batch_size:int=32,
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.graph_node_features_dim = graph_node_features_dim
        self.graph_edge_features_dim = graph_edge_features_dim
        self.graph_hidden_dim = graph_hidden_dim
        
        # 多层超图卷积 (每个时间步共享相同的层) 空间特征提取
        self.hypergraph_convs =nn.ModuleList([
            HypergraphConv(in_channels= graph_node_features_dim,
                           out_channels=2*graph_hidden_dim,
                           edge_dim=graph_edge_features_dim,  # 提供超边特征维度
                           use_attention=True,
                           attention_mode= 'edge',
                           heads= 3,
                           dropout=0.2,
                           concat=True),
            HypergraphConv(in_channels= 6*graph_hidden_dim,
                           out_channels=graph_hidden_dim,
                           edge_dim=graph_edge_features_dim,  # 提供超边特征维度
                           use_attention=True,
                           attention_mode= 'node',
                           heads= 2,
                           dropout=0.2,
                           concat=False),
        ])

        
        # LSTM层用于处理时间序列
        self.lstm = LSTMWithAttention(input_dim=graph_hidden_dim,
                                      hidden_dim=2*(graph_hidden_dim),
                                      output_dim=lstm_out_dim,
                                      batch_first=True,
                                      num_layers=num_lstm_layers
                                     )

        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(lstm_out_dim//2),
            nn.Linear(lstm_out_dim//2, 1)
        )

    def Spatial_feature_extraction(
        self, 
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        hyperedge_feature: torch.Tensor,
        hyperedge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        批量空间特征提取
        Args:
            x: [num_nodes, node_features_dim]
            hyperedge_index: [2, T]
            hyperedge_feature: [num_edges, edge_features_dim]
            hyperedge_weight: [num_edges] or None
        Returns:
            node_encoded: [num_nodes, hidden_dim]
            attention_weights: (node_attention, edge_attention)
        """
        batch_size = x.shape[0]
        node_encoded_list = []
        node_attention_list = []
        edge_attention_list = []

        num_edges = hyperedge_feature.shape[0]
        device = hyperedge_index.device

        if hyperedge_weight==None:
            hyperedge_weight=torch.ones(num_edges).to(device)
        node_encoded_i, HG_attention_1 = self.hypergraph_convs[0](
                x=x,
                hyperedge_index=hyperedge_index,
                hyperedge_weight=hyperedge_weight,
                hyperedge_attr=hyperedge_feature,)
            
            # 第二层超图卷积
        node_encoded, HG_attention_2 = self.hypergraph_convs[1](
                x=node_encoded_i,
                hyperedge_index=hyperedge_index,
                hyperedge_weight=hyperedge_weight,
                hyperedge_attr=hyperedge_feature,
        )
        
        return node_encoded, (HG_attention_1, HG_attention_2)

    
    def forward(self, node_features: torch.Tensor,
                hyperedge_indices: List[List[torch.Tensor]],
                hyperedge_feature: List[List[torch.Tensor]],
                hyperedge_weights: torch.Tensor = None,
                ):
        """
        前向传播
        Args:
            node_features: [batch_size, num_timesteps, num_nodes, num_node_features]
            hyperedge_indices: List[batch_size] List[num_timesteps] [2, T]
            hyperedge_feature: List[batch_size] List[num_timesteps] [num_edge, num_edge_features]
            hyperedge_weights: [batch_size, num_timesteps, num_edges] 或 None
        Returns:
            logits: [batch_size,num_nodes, 1]
        """
        batch_size = node_features.shape[0]
        num_nodes = node_features.shape[2]
        #device = node_features.device

        

        features_list = []
        spatial_attention_list=[]
        for b in range(batch_size):
            batch_features_list,batch_spatial_attention_list=[],[]
            for t in range(self.num_timesteps):
                x_t = node_features[b,t, :, :]  # [num_nodes, num_features]
                e_t = hyperedge_feature[b][t].to(torch.float32)  # [num_edge, num_edge_features]
                # 获取当前时间步的超图结构
                hyperedge_index_t = hyperedge_indices[b][t]  # [2, num_edges]
                # 提取空间特征
                batch_time_features, batch_time_spatial_attention=self.Spatial_feature_extraction(
                    x_t, 
                    hyperedge_index_t,
                    e_t,
                    )
                batch_features_list.append(batch_time_features)
                batch_spatial_attention_list.append(batch_time_spatial_attention)

            features_list.append(torch.stack(batch_features_list, dim=0))
            spatial_attention_list.append(batch_spatial_attention_list)
        

        features_torch= torch.stack(features_list, dim=0)  #[batch_size, num_nodes, num_timesteps, hidden_dim]

        
        batch_size, num_timesteps, num_nodes, hidden_dim = features_torch.shape
        lstm_input = features_torch.permute(0, 2, 1, 3)  # [batch_size, num_nodes, num_timesteps, hidden_dim]
        lstm_input = lstm_input.reshape(batch_size * num_nodes, num_timesteps, hidden_dim)

        lstm_out, lstm_attention = self.lstm(lstm_input)

        # 分类
        logits = self.classifier(lstm_out)
        logits=logits.reshape(batch_size, num_nodes)  
        
        return logits,[lstm_attention, spatial_attention_list], lstm_out
    

def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义批处理函数，处理动态超图数据
    """
    # 获取批量大小
    batch_size = len(batch)
    
    # 获取第一个样本的形状信息
    num_timesteps = batch[0]['node_features'].shape[0]
    num_nodes = batch[0]['node_features'].shape[1]
    node_features_dim = batch[0]['node_features'].shape[2]
    #num_edges = batch[0]['hyperedge_features'].shape[1]
    #edge_features_dim = batch[0]['hyperedge_features'][0].shape[1]
    
    # 初始化批处理张量
    batch_node_features = torch.zeros(batch_size, num_timesteps, num_nodes, node_features_dim)
    batch_hyperedge_features = []
    batch_hyperedge_indices =[]
    batch_labels = torch.zeros(batch_size, num_nodes)
    
    # 填充数据
    for i, sample in enumerate(batch):
        batch_node_features[i] = sample['node_features']
        batch_hyperedge_features.append(sample['hyperedge_features'])
        batch_hyperedge_indices.append(sample['hyperedge_indices'])
        batch_labels[i] = sample['labels']
    return {
        'node_features': batch_node_features,
        'hyperedge_features': batch_hyperedge_features,
        'hyperedge_indices': batch_hyperedge_indices,
        'labels': batch_labels
    }



# ==================== 训练器 ====================
class SimpleTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs: int = 2000,
        patience: int = 100,
        save_dir: str = './checkpoints',
        save_attention: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience  # 早停耐心值
        self.save_dir = save_dir
        self.save_attention = save_attention
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_auc_scores = []
        self.val_auc_scores = []
        
        # 早停相关变量
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_val_f1 = 0.0
        self.epochs_no_improve = 0
        self.best_epoch = 0
        self.best_model_state = None
        self.best_attention_weights = None
        
    def compute_metrics(self, y_true, y_pred, y_prob=None):
        """计算二分类评价指标"""
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 精确率、召回率、F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # AUC (如果提供了概率)
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
            
        return metrics
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        # 用于计算指标
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        for batch in self.train_loader:
            # 将数据移动到设备
            node_features = batch['node_features'].to(self.device)
            hyperedge_features = batch['hyperedge_features']
            hyperedge_features=[[tensor.to(self.device) for tensor in sublist] for sublist in hyperedge_features]
            
            hyperedge_indices = batch['hyperedge_indices']
            hyperedge_indices = [[tensor.to(self.device) for tensor in sublist] for sublist in hyperedge_indices]
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            logits, attention_weights = self.model(
                node_features=node_features,
                hyperedge_indices=hyperedge_indices,
                hyperedge_feature=hyperedge_features,
                hyperedge_weights=None
            )
            
            # 计算损失
            loss = self.criterion(logits.view(-1), labels.view(-1))
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 收集预测结果用于计算指标
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            
            all_labels.extend(labels.cpu().numpy().flatten())
            all_predictions.extend(predictions.flatten())
            all_probabilities.extend(probabilities.flatten())
        
        avg_loss = total_loss / len(self.train_loader)
        
        # 计算指标
        metrics = self.compute_metrics(
            np.array(all_labels), 
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        
        # 用于计算指标
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_attention_weights = []
        
        for batch in self.val_loader:
            # 将数据移动到设备
            node_features = batch['node_features'].to(self.device)
            hyperedge_features = batch['hyperedge_features']
            hyperedge_features =[[tensor.to(self.device) for tensor in sublist] for sublist in hyperedge_features]
            hyperedge_indices = batch['hyperedge_indices']#.to(self.device)
            hyperedge_indices = [[tensor.to(self.device) for tensor in sublist] for sublist in hyperedge_indices]
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            logits, attention_weights = self.model(
                node_features=node_features,
                hyperedge_indices=hyperedge_indices,
                hyperedge_feature=hyperedge_features,
                hyperedge_weights=None
            )
            
            # 保存注意力权重（只保存最后一个batch）
            if self.save_attention and attention_weights is not None:
                # 处理注意力权重 - 确保转换为可保存的格式
                attention_to_save = []
                for attn in attention_weights:
                    if attn is not None:
                        # 如果注意力权重是张量，移动到CPU
                        if isinstance(attn, torch.Tensor):
                            attention_to_save.append(attn.cpu())
                        # 如果是列表，将列表中的张量移动到CPU
                        elif isinstance(attn, list):
                            attention_to_save.append([a.cpu() if isinstance(a, torch.Tensor) else a for a in attn])
                        else:
                            attention_to_save.append(attn)
                    else:
                        attention_to_save.append(None)
                
                all_attention_weights.append({
                    'attention_weights': attention_to_save,
                    #'hyperedge_indices': [i.cpu() for i in hyperedge_indices],
                    'node_features': node_features.cpu()
                })
            
            # 计算损失
            loss = self.criterion(logits.view(-1), labels.view(-1))
            total_loss += loss.item()
            
            # 收集预测结果
            probabilities = torch.sigmoid(logits).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            
            all_labels.extend(labels.cpu().numpy().flatten())
            all_predictions.extend(predictions.flatten())
            all_probabilities.extend(probabilities.flatten())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算指标
        metrics = self.compute_metrics(
            np.array(all_labels), 
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        # 只保留最后一个batch的注意力权重用于保存
        attention_to_save = all_attention_weights[-1] if all_attention_weights else None
        
        return avg_loss, metrics, attention_to_save
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'train_f1_scores': self.train_f1_scores,
            'val_f1_scores': self.val_f1_scores,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch
        }
        
        # 保存当前检查点
        checkpoint_path = f'{self.save_dir}/checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，单独保存
        if is_best:
            best_model_path = f'{self.save_dir}/best_model.pt'
            torch.save(checkpoint, best_model_path)
            print(f"最佳模型已保存到: {best_model_path}")
            
            # 保存注意力权重
            if self.best_attention_weights is not None and self.save_attention:
                attention_path = f'{self.save_dir}/best_attention_weights.pt'
                torch.save(
                    self.best_attention_weights,
                    attention_path
                )
                print(f"最佳注意力权重已保存到: {attention_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复训练历史
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.train_f1_scores = checkpoint.get('train_f1_scores', [])
        self.val_f1_scores = checkpoint.get('val_f1_scores', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        print(f"从检查点加载模型: {checkpoint_path}")
        print(f"已训练epoch数: {checkpoint['epoch'] + 1}")
        print(f"最佳验证F1: {self.best_val_f1:.4f}")
    
    def early_stopping(self, val_loss, val_f1):
        """早停机制"""
        improvement = False
        
        # 使用F1分数作为主要早停指标（或使用损失）
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            self.best_epoch = len(self.train_losses) - 1  # 当前epoch索引
            improvement = True
        else:
            self.epochs_no_improve += 1
        
        # 检查是否应该早停
        if (self.best_epoch>300) and (self.epochs_no_improve >= self.patience):
            return True, improvement
        return False, improvement
    
    def train(self):
        """完整训练流程"""
        print(f"开始训练，设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"早停耐心值: {self.patience}")
        
        for epoch in range(self.num_epochs):

            # 训练阶段
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_metrics['accuracy'])
            self.train_precisions.append(train_metrics['precision'])
            self.train_recalls.append(train_metrics['recall'])
            self.train_f1_scores.append(train_metrics['f1'])
            self.train_auc_scores.append(train_metrics['auc'])
            
            # 验证阶段
            val_loss, val_metrics, attention_weights = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_metrics['accuracy'])
            self.val_precisions.append(val_metrics['precision'])
            self.val_recalls.append(val_metrics['recall'])
            self.val_f1_scores.append(val_metrics['f1'])
            self.val_auc_scores.append(val_metrics['auc'])
            
            # 打印详细指标
            print(f"Epoch {epoch+1}/{self.num_epochs}: 损失: {train_loss:.4f};准确率: { train_metrics['accuracy']:.4f};精确率: {train_metrics['precision']:.4f}; 召回率: {train_metrics['recall']:.4f};F1分数: {train_metrics['f1']:.4f}; AUC: {train_metrics['auc']:.4f}")
            print(f"验证结果:损失: {val_loss:.4f}; 准确率: {val_metrics['accuracy']:.4f}; 精确率: {val_metrics['precision']:.4f};召回率: {val_metrics['recall']:.4f}; F1分数: {val_metrics['f1']:.4f}; AUC: {val_metrics['auc']:.4f}")
            
            # 早停检查
            stop_early, improved = self.early_stopping(val_loss, val_metrics['f1'])
            
            if improved:
                self.best_model_state = self.model.state_dict().copy()
                self.best_attention_weights = attention_weights
                # 保存最佳模型
                self.save_checkpoint(epoch, is_best=True)

            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
            
            # 检查早停
            if stop_early:
                print(f"\n早停触发! 在epoch {epoch+1}停止训练")
                print(f"最佳模型在epoch {self.best_epoch+1} (F1: {self.best_val_f1:.4f})")
                break
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n加载最佳模型 (epoch {self.best_epoch+1})")
        
        # 绘制训练历史
        #self.plot_training_history()
        
        # 返回完整的训练历史
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'train_precisions': self.train_precisions,
            'train_recalls': self.train_recalls,
            'train_f1_scores': self.train_f1_scores,
            'train_auc_scores': self.train_auc_scores,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_precisions': self.val_precisions,
            'val_recalls': self.val_recalls,
            'val_f1_scores': self.val_f1_scores,
            'val_auc_scores': self.val_auc_scores,
            'best_epoch': self.best_epoch,
            'best_val_f1': self.best_val_f1,
            'best_val_loss': self.best_val_loss
        }
    
    def plot_training_history(self):
        """绘制训练历史"""
        if not self.train_losses:
            print("没有训练历史可绘制")
            return
            
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='训练准确率', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='验证准确率', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('训练和验证准确率')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1分数曲线
        axes[0, 2].plot(epochs, self.train_f1_scores, 'b-', label='训练F1分数', linewidth=2)
        axes[0, 2].plot(epochs, self.val_f1_scores, 'r-', label='验证F1分数', linewidth=2)
        if self.best_epoch < len(epochs):
            axes[0, 2].axvline(x=self.best_epoch+1, color='g', linestyle='--', 
                              label=f'最佳epoch {self.best_epoch+1}')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].set_title('训练和验证F1分数')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 精确率-召回率曲线
        axes[1, 0].plot(epochs, self.train_precisions, 'b-', label='训练精确率', linewidth=2)
        axes[1, 0].plot(epochs, self.train_recalls, 'b--', label='训练召回率', linewidth=2)
        axes[1, 0].plot(epochs, self.val_precisions, 'r-', label='验证精确率', linewidth=2)
        axes[1, 0].plot(epochs, self.val_recalls, 'r--', label='验证召回率', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('精确率和召回率')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC曲线
        axes[1, 1].plot(epochs, self.train_auc_scores, 'b-', label='训练AUC', linewidth=2)
        axes[1, 1].plot(epochs, self.val_auc_scores, 'r-', label='验证AUC', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('训练和验证AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 最佳模型标记
        if self.best_epoch < len(self.val_f1_scores):
            axes[1, 2].text(0.1, 0.5, 
                           f"最佳模型统计:\n"
                           f"Epoch: {self.best_epoch+1}\n"
                           f"验证F1: {self.best_val_f1:.4f}\n"
                           f"验证损失: {self.best_val_loss:.4f}\n"
                           f"验证准确率: {self.val_accuracies[self.best_epoch]:.4f}\n"
                           f"验证AUC: {self.val_auc_scores[self.best_epoch]:.4f}",
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        else:
            axes[1, 2].text(0.1, 0.5, 
                           f"最佳模型统计:\n"
                           f"Epoch: {self.best_epoch+1}\n"
                           f"验证F1: {self.best_val_f1:.4f}\n"
                           f"验证损失: {self.best_val_loss:.4f}",
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def evaluate_test(self, test_loader):
        """在测试集上评估模型"""
        print("\n在测试集上评估模型...")
        
        self.model.eval()
        total_loss = 0.0
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 将数据移动到设备
                node_features = batch['node_features'].to(self.device)
                hyperedge_features = batch['hyperedge_features']
                hyperedge_features = [[tensor.to(self.device) for tensor in sublist] for sublist in hyperedge_features]
                
                hyperedge_indices = batch['hyperedge_indices']#.to(self.device)
                hyperedge_indices = [[tensor.to(self.device) for tensor in sublist] for sublist in hyperedge_indices]
                
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                logits, _ = self.model(
                    node_features=node_features,
                    hyperedge_indices=hyperedge_indices,
                    hyperedge_feature=hyperedge_features,
                    hyperedge_weights=None
                )
                
                # 计算损失
                loss = self.criterion(logits.view(-1), labels.view(-1))
                total_loss += loss.item()
                
                # 收集预测结果
                probabilities = torch.sigmoid(logits).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                
                all_labels.extend(labels.cpu().numpy().flatten())
                all_predictions.extend(predictions.flatten())
                all_probabilities.extend(probabilities.flatten())
        
        avg_loss = total_loss / len(test_loader)
        
        # 计算指标
        metrics = self.compute_metrics(
            np.array(all_labels), 
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        print(f"测试结果:")
        print(f"  损失: {avg_loss:.4f}")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        
        # 混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print(f"\n混淆矩阵:")
        if cm.shape == (2, 2):
            print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}]")
            print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
        else:
            print(f"  {cm}")
        
        return metrics
    


# ==================== 主训练脚本 ====================

def main(train_dataset, val_dataset, test_dataset, model_params):
    """主训练脚本"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    # 创建模型
    print("\n创建模型...")


    model = DynamicHyperGNN(
        graph_node_features_dim=train_dataset[0]['node_features'].shape[2],
        graph_edge_features_dim=train_dataset[0]['hyperedge_features'][0].shape[1],
        num_timesteps=train_dataset[0]['node_features'].shape[0],
        num_nodes=train_dataset[0]['node_features'].shape[1],
        batch_size=batch_size,
        **model_params).to(device)
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 创建损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建训练器
    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=2000
    )
    
    # 开始训练
    results = trainer.train()
    trainer.plot_training_history()
    R=trainer.evaluate_test(test_loader=test_loader)
    print(f"\n训练完成!")
    print(f"最终训练损失: {results['train_losses'][-1]:.4f}")
    print(f"最终训练准确率: {results['train_accuracies'][-1]:.4f}")
    print(f"最终验证损失: {results['val_losses'][-1]:.4f}")
    print(f"最终验证准确率: {results['val_accuracies'][-1]:.4f}")
    
    return model, results


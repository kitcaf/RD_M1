# 任务：早期谣言信息检测

## 当前模型结构



联合训练的端到端模型，不是独立训练的。整个系统通过**共享编码器**将两个任务连接起来。

```
输入数据 → SharedEncoder → ┌→ CascadePredictor (链接预测)
                          └→ RumorDetector (谣言分类)
```

SharedEncoder：学习共享特征
- 统一特征编码: 将原始节点特征（Word2Vec + 度数特征）编码成高维表示
- 图结构学习: 通过GraphSAGE学习节点在传播图中的结构化表示
- 特征共享: 为CascadePredictor和RumorDetector提供统一的节点嵌入
- 信息聚合: 聚合邻居节点信息，捕获传播模式


CascadePredictor：时序链接预测器

- 任务：预测早期级联中"哪些边会在未来出现"
- 输入：早期级联图（前75%的传播）
- 输出：预测后续25%时间段内可能出现的链接
- 技术：使用多头注意力机制计算节点对之间的链接概率

RumorDetector：基于重构图的谣言分类器

- 任务：基于"原始早期图 + 预测链接"进行谣言分类
- 输入：重构后的完整传播图
- 输出：谣言类别（false/true/unverified/non-rumor）
- 技术：图池化 + 全局特征聚合 + 分类



## 当前模型测试结果：
跑完的结果如下：

- 平均准确率: 0.3765
- 平均F1分数: 0.3083
> 这个结果也太低了，没有下载到论文，它怎么进行对比的啊


## 当前结构问题：CascadePredictor是最有创新潜力的组件

- 简单的边预测：只是基于注意力机制预测边的存在概率
- 缺乏传播动力学：没有考虑信息传播的时序模式和用户行为
- 静态特征：只使用Word2Vec，忽略了动态传播特征

#### 改进方向：

> 能不能**生成更realistic的传播路径**

- 图扩散模型 (Graph Diffusion)
- 考虑时序依赖和传播模式

（1）基于扩散的级联预测
> 将传播建模为扩散过程
可能：Graph Diffusion Models, Score-based Generative Models

（2）基于VAE的图生成
> 学习传播图的潜在表示和生成规律
可能：Graph VAE, Conditional Graph Generation

（3）基于流模型的级联扩展
> 学习从早期到完整级联的可逆变换
可能：Graph Normalizing Flows

## 源数据集文件结构分析

根据`run.py`的完整流程分析，真正的源数据集包含以下文件：

### 📁 **核心数据文件**

#### **文件1: 推文内容文件 (`source_tweets.txt`)**
```
格式: TSV (Tab分隔)
结构: tweet_id	content
功能: 存储推文ID和对应的文本内容
示例:
12345	This is the original tweet content about some news
67890	RT @user: Another tweet content here
```

#### **文件2: 传播树目录 (`tree/`)**
```
格式: 多个.txt文件，每个文件代表一个传播级联
结构: parent_node -> child_node (使用eval解析的元组格式)
功能: 描述信息传播的树状结构和时间序列
示例: (tweet_id, user_id, timestamp) -> (child_tweet_id, child_user_id, child_timestamp)
```

#### **文件3: 标签文件 (`label.txt`)**
```
格式: 文本文件
结构: 每行包含一个传播级联的标签信息
功能: 标识每个传播级联是否为谣言
标签类型: 'false', 'true', 'unverified', 'non-rumor'
```

### 🔄 **预处理数据文件 (preprocess.py使用)**

#### **文件4-6: 分割数据集文件**
```
twitter16.train - 训练集: tweet_id	content	label
twitter16.dev   - 开发集: tweet_id	content	label  
twitter16.test  - 测试集: tweet_id	content	label
```

#### **文件7: 用户关系图 (`twitter16_graph.txt`)**
```
格式: 空格分隔
结构: user_id dst_user1:weight1 dst_user2:weight2 ...
功能: 描述用户之间的社交关系
```

### 📊 **数据处理流程对比**

| 处理方式 | 使用文件 | 特征提取 | 图构建方式 |
|---------|----------|----------|-----------|
| **run.py流程** | `source_tweets.txt` + `tree/` + `label.txt` | Word2Vec词向量 | 传播树结构 |
| **data.py流程** | `data.TD_RvNN.vol_5000.txt` + `Twitter16_label_all.txt` | 仅文本长度 | 父子推文关系 |  
| **preprocess.py流程** | `.train/.dev/.test` + `_graph.txt` | Word2Vec词向量 | 用户社交关系 |

### 💡 **核心结论**

1. **run.py是完整实现**: 包含Word2Vec文本处理 + 传播树分析
2. **data.py是简化版本**: 只使用文本长度特征，缺少语义信息  
3. **preprocess.py是传统方法**: 基于用户关系图的经典社交网络分析

**结论**: `run.py`中的数据集结构是最完整的，同时利用了**文本语义**和**传播结构**信息。
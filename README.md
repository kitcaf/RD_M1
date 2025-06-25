# 任务：早期谣言信息检测

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

### 💡 **关键发现**

1. **run.py是完整实现**: 包含Word2Vec文本处理 + 传播树分析
2. **data.py是简化版本**: 只使用文本长度特征，缺少语义信息  
3. **preprocess.py是传统方法**: 基于用户关系图的经典社交网络分析

**结论**: `run.py`中的数据集结构是最完整的，同时利用了**文本语义**和**传播结构**信息。
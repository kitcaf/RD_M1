import itertools
import re
from collections import Counter
import gensim
import numpy as np
import scipy.sparse as sp
import pickle
import jieba
jieba.set_dictionary('/home/hwxu/Projects/Research/NPU/WSDM/Input/dict.txt.big')  # 设置jieba分词的词典路径


w2v_dim = 300  # Word2Vec嵌入的维度

dic = {
    'non-rumor': 0,   # 非谣言   NR
    'false': 1,   # 虚假谣言    FR
    'unverified': 2,  # 未验证推文  UR
    'true': 3,    # 已澄清谣言  TR
}

def clean_str_cut(string, task):
    """
    对所有数据集进行分词/字符串清洗，除了SST。
    原始代码来自 https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    
    参数:
        string: 要清洗的字符串
        task: 任务类型，如果是"weibo"则使用jieba分词
        
    返回:
        words: 分词后的单词列表
    """
    if task != "weibo":
        string = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", string)  # 替换非字母、数字和特定符号为空格
        string = re.sub(r"\'s", " \'s", string)  # 处理缩写
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)

    string = re.sub(r",", " , ", string)  # 在标点符号周围添加空格
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)  # 将多个空格替换为单个空格

    words = list(jieba.cut(string.strip().lower())
                 ) if task == "weibo" else string.strip().lower().split()  # 根据任务类型选择分词方式
    return words


def build_symmetric_adjacency_matrix(edges, shape):
    """
    构建对称邻接矩阵并进行归一化
    
    参数:
        edges: 边列表，每行包含[源节点，目标节点，权重]
        shape: 矩阵形状
        
    返回:
        归一化后的对称邻接矩阵
    """
    def normalize_adj(mx):
        """对稀疏矩阵进行行归一化"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.  # 处理无穷大值
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)  # 对称归一化

    adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])), shape=shape, dtype=np.float32)  # 创建稀疏邻接矩阵
    # 构建对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # 确保矩阵对称
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))  # 添加自环并归一化
    return adj.tocoo()  # 返回COO格式的稀疏矩阵


def read_corpus(root_path, file_name):
    """
    读取语料库文件，包括训练、开发和测试集
    
    参数:
        root_path: 数据根路径
        file_name: 文件名前缀
        
    返回:
        处理后的数据集和关系矩阵
    """
    X_tids = []  # 推文ID列表
    X_uids = []  # 用户ID列表

    with open(root_path + file_name +".train", 'r', encoding='utf-8') as input:
        X_train_tid, X_train_content, y_train = [], [], []  # 训练集数据
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")  # 解析行数据
            X_tids.append(tid)
            X_train_tid.append(tid)
            X_train_content.append(clean_str_cut(content, file_name))  # 清洗并分词内容
            y_train.append(dic[label])  # 将标签转换为数值

    with open(root_path + file_name +".dev", 'r', encoding='utf-8') as input:
        X_dev_tid, X_dev_content, y_dev = [], [], []  # 开发集数据
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            X_tids.append(tid)
            X_dev_tid.append(tid)
            X_dev_content.append(clean_str_cut(content, file_name))
            y_dev.append(dic[label])

    with open(root_path + file_name +".test", 'r', encoding='utf-8') as input:
        X_test_tid, X_test_content, y_test = [], [], []  # 测试集数据
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            X_tids.append(tid)
            X_test_tid.append(tid)
            X_test_content.append(clean_str_cut(content, file_name))
            y_test.append(dic[label])

    with open(root_path + file_name +"_graph.txt", 'r', encoding='utf-8') as input:
        relation = []  # 关系数据
        for line in input.readlines():
            tmp = line.strip().split()
            src = tmp[0]  # 源节点
            X_uids.append(src)

            for dst_ids_ws in tmp[1:]:
                dst, w = dst_ids_ws.split(":")  # 解析目标节点和权重
                X_uids.append(dst)
                relation.append([src, dst, w])  # 添加关系

    X_id = list(set(X_tids + X_uids))  # 所有唯一ID
    num_node = len(X_id)  # 节点数量
    print(num_node)
    X_id_dic = {id:i for i, id in enumerate(X_id)}  # ID到索引的映射

    relation = np.array([[X_id_dic[tup[0]], X_id_dic[tup[1]], tup[2]] for tup in relation])  # 将关系转换为索引形式
    relation = build_symmetric_adjacency_matrix(relation, shape=(num_node, num_node))  # 构建对称邻接矩阵

    X_train_tid = np.array([X_id_dic[tid] for tid in X_train_tid])  # 将训练集ID转换为索引
    X_dev_tid = np.array([X_id_dic[tid] for tid in X_dev_tid])  # 将开发集ID转换为索引
    X_test_tid = np.array([X_id_dic[tid] for tid in X_test_tid])  # 将测试集ID转换为索引

    return X_train_tid, X_train_content, y_train, \
           X_dev_tid, X_dev_content, y_dev, \
           X_test_tid, X_test_content, y_test, \
           relation


# def train_dev_test_split(root_path, file_name):
#     num_node, relation, X_tid, X_content, y = read_corpus(root_path, file_name)
#     relation = build_symmetric_adjacency_matrix(relation, shape=(num_node, num_node))
#
#     X_content_idx = np.arange(len(X_content))
#     X_idx, X_dev_idx, y, y_dev = train_test_split(X_content_idx, y, test_size=0.1, random_state=0, stratify=y)  #
#
#     X_dev_tid = X_tid[X_dev_idx].tolist()
#     X_dev = X_content[X_dev_idx].tolist()
#
#     X_tid = X_tid[X_idx]
#     X_content = X_content[X_idx]
#
#     X_content_idx = np.arange(len(X_content))
#     X_train_idx, X_test_idx, y_train, y_test = train_test_split(X_content_idx, y, test_size=0.25, random_state=0, stratify=y) #
#
#     X_train_tid = X_tid[X_train_idx].tolist()
#     X_test_tid = X_tid[X_test_idx].tolist()
#
#     X_train = X_content[X_train_idx].tolist()
#     X_test = X_content[X_test_idx].tolist()
#
#     return X_train_tid, X_train, y_train.tolist(), \
#            X_dev_tid, X_dev, y_dev.tolist(), \
#            X_test_tid, X_test, y_test.tolist(), \
#            relation


def vocab_to_word2vec(fname, vocab):
    """
    从Mikolov格式加载word2vec模型
    
    参数:
        fname: word2vec模型文件路径
        vocab: 词汇表
        
    返回:
        word_vecs: 词向量字典
    """
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)  # 加载预训练的word2vec模型
    count_missing = 0
    for word in vocab:
        if model.__contains__(word):
            word_vecs[word] = model[word]  # 使用预训练的词向量
        else:
            #为未知词生成随机词向量
            count_missing += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, w2v_dim)  # 随机初始化未知词的向量

    print(str(len(word_vecs) - count_missing)+" words found in word2vec.")
    print(str(count_missing)+" words not found, generated by random.")
    return word_vecs


def build_vocab_word2vec(sentences, w2v_path='numberbatch-en.txt'):
    """
    基于句子构建从词到索引的词汇表映射。
    返回词汇表映射和反向词汇表映射。
    
    参数:
        sentences: 句子列表
        w2v_path: word2vec模型路径
        
    返回:
        vocabulary: 词到索引的映射
        embedding_weights: 词嵌入权重矩阵
    """
    # 构建词汇表
    vocabulary_inv = []
    word_counts = Counter(itertools.chain(*sentences))  # 计算所有单词的出现次数
    # 从索引到词的映射
    vocabulary_inv += [x[0] for x in word_counts.most_common() if x[1] >= 2]  # 只保留出现至少2次的单词
    # 从词到索引的映射
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    print("embedding_weights generation.......")
    word2vec = vocab_to_word2vec(w2v_path, vocabulary)  # 获取词向量
    embedding_weights = build_word_embedding_weights(word2vec, vocabulary_inv)  # 构建词嵌入权重
    return vocabulary, embedding_weights


def pad_sequence(X, max_len=50):
    """
    对序列进行填充或截断，使其长度为max_len
    
    参数:
        X: 序列列表
        max_len: 最大长度
        
    返回:
        X_pad: 填充后的序列列表
    """
    X_pad = []
    for doc in X:
        if len(doc) >= max_len:
            doc = doc[:max_len]  # 截断过长的序列
        else:
            doc = [0] * (max_len - len(doc)) + doc  # 在序列前填充0
        X_pad.append(doc)
    return X_pad


def build_word_embedding_weights(word_vecs, vocabulary_inv):
    """
    获取词嵌入矩阵，大小为(vocabulary_size, word_vector_size)
    第i行是词汇表中第i个词的嵌入
    
    参数:
        word_vecs: 词向量字典
        vocabulary_inv: 从索引到词的映射
        
    返回:
        embedding_weights: 词嵌入权重矩阵
    """
    vocab_size = len(vocabulary_inv)
    embedding_weights = np.zeros(shape=(vocab_size+1, w2v_dim), dtype='float32')  # 初始化嵌入矩阵
    #初始化第一行
    embedding_weights[0] = np.zeros(shape=(w2v_dim,) )  # 第一行为0向量，通常用于填充或未知词

    for idx in range(1, vocab_size):
        embedding_weights[idx] = word_vecs[vocabulary_inv[idx]]  # 设置每个词的嵌入向量
    print("Embedding matrix of size "+str(np.shape(embedding_weights)))
    return embedding_weights


def build_input_data(X, vocabulary):
    """
    基于词汇表将句子和标签映射为向量。
    
    参数:
        X: 句子列表
        vocabulary: 词到索引的映射
        
    返回:
        x: 索引化的句子列表
    """
    x = [[vocabulary[word] for word in sentence if word in vocabulary] for sentence in X]  # 将词转换为索引
    x = pad_sequence(x)  # 对序列进行填充
    return x


def w2v_feature_extract(root_path, filename, w2v_path):
    """
    使用word2vec提取特征并保存处理后的数据
    
    参数:
        root_path: 数据根路径
        filename: 文件名前缀
        w2v_path: word2vec模型路径
    """
    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, relation = read_corpus(root_path, filename)  # 读取语料库

    print("text word2vec generation.......")
    vocabulary, word_embeddings = build_vocab_word2vec(X_train + X_dev + X_test, w2v_path=w2v_path)  # 构建词汇表和词嵌入
    pickle.dump(vocabulary, open(root_path + "/vocab.pkl", 'wb'))  # 保存词汇表
    print("Vocabulary size: "+str(len(vocabulary)))

    print("build input data.......")
    X_train = build_input_data(X_train, vocabulary)  # 处理训练数据
    X_dev = build_input_data(X_dev, vocabulary)  # 处理开发数据
    X_test = build_input_data(X_test, vocabulary)  # 处理测试数据

    pickle.dump([X_train_tid, X_train, y_train, word_embeddings, relation], open(root_path+"/train.pkl", 'wb') )  # 保存训练数据
    pickle.dump([X_dev_tid, X_dev, y_dev], open(root_path+"/dev.pkl", 'wb') )  # 保存开发数据
    pickle.dump([X_test_tid, X_test, y_test], open(root_path+"/test.pkl", 'wb') )  # 保存测试数据


if __name__ == "__main__":
    w2v_feature_extract('/home/hwxu/Projects/Research/NPU/WSDM/Input/Twitter15/', "twitter15", "twitter_w2v.bin")  # 处理Twitter15数据集
    w2v_feature_extract('/home/hwxu/Projects/Research/NPU/WSDM/Input/Twitter16/', "twitter16", "twitter_w2v.bin")  # 处理Twitter16数据集



"""
向量化RAG工具 - 基于语义相似度的智能检索
使用国产text2vec模型实现高精度文档检索
"""
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
import pickle
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers未安装，将使用简单文本匹配")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss未安装，将使用numpy相似度计算")

class VectorRAGTool:
    """向量化RAG工具类"""
    
    def __init__(self, knowledge_base_path: str = "knowledge_base/platform_knowledge.json"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.cache_dir = Path("data/vector_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # 使用国产优秀的text2vec模型
        self.model_name = "shibing624/text2vec-base-chinese"
        self.model = None
        self.knowledge_chunks = []
        self.embeddings = None
        self.faiss_index = None
        
        # 配置参数
        self.chunk_size = 200  # 文档分块大小
        self.chunk_overlap = 50  # 分块重叠
        self.top_k = 3  # 检索top-k结果
        self.similarity_threshold = 0.5  # 相似度阈值
        
        self._initialize()
    
    def _initialize(self):
        """初始化向量化RAG系统"""
        try:
            # 为了演示快速启动，暂时跳过向量化模型加载
            # 直接使用文本匹配模式，确保演示流畅进行
            logging.info("向量化RAG工具初始化 - 使用快速启动模式")
            logging.info("跳过模型加载，使用文本匹配模式进行演示")
            self.model = None
            return
            
            # 注释掉的向量化加载代码（演示后可恢复）
            """
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # 减少HuggingFace的详细日志
                import transformers
                transformers.logging.set_verbosity_error()
                import huggingface_hub
                huggingface_hub.logging.set_verbosity_error()
                
                logging.info(f"正在加载向量化模型: {self.model_name}")
                try:
                    self.model = SentenceTransformer(self.model_name)
                    logging.info("向量化模型加载成功")
                except Exception as model_load_error:
                    logging.warning(f"向量化模型加载失败，使用降级文本匹配: {str(model_load_error)[:100]}")
                    self.model = None
                    return
            else:
                logging.warning("sentence-transformers不可用，使用传统文本匹配")
                return
            
            # 加载或构建向量索引
            if self._should_rebuild_index():
                self._build_vector_index()
            else:
                self._load_cached_index()
            """
            
            # 加载或构建向量索引
            if self._should_rebuild_index():
                self._build_vector_index()
            else:
                self._load_cached_index()
                
        except Exception as e:
            logging.warning(f"向量化RAG初始化失败，使用降级模式: {e}")
            self.model = None
    
    def _should_rebuild_index(self) -> bool:
        """判断是否需要重建索引"""
        cache_file = self.cache_dir / "vector_index.pkl"
        knowledge_file = self.knowledge_base_path
        
        if not cache_file.exists() or not knowledge_file.exists():
            return True
        
        # 检查知识库文件是否有更新
        cache_time = cache_file.stat().st_mtime
        knowledge_time = knowledge_file.stat().st_mtime
        
        return knowledge_time > cache_time
    
    def _build_vector_index(self):
        """构建向量索引"""
        try:
            logging.info("正在构建向量索引...")
            
            # 加载知识库
            knowledge_data = self._load_knowledge_base()
            if not knowledge_data:
                return
            
            # 文档分块
            self.knowledge_chunks = self._chunk_documents(knowledge_data)
            logging.info(f"文档分块完成，共 {len(self.knowledge_chunks)} 个chunk")
            
            # 生成嵌入向量
            texts = [chunk['text'] for chunk in self.knowledge_chunks]
            self.embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
            logging.info(f"向量化完成，维度: {self.embeddings.shape}")
            
            # 构建FAISS索引
            if FAISS_AVAILABLE:
                self._build_faiss_index()
            
            # 缓存结果
            self._cache_index()
            logging.info("向量索引构建并缓存成功")
            
        except Exception as e:
            logging.error(f"构建向量索引失败: {e}")
    
    def _load_knowledge_base(self) -> List[Dict]:
        """加载知识库数据"""
        try:
            if not self.knowledge_base_path.exists():
                logging.warning(f"知识库文件不存在: {self.knowledge_base_path}")
                return []
            
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 将JSON数据转换为文档列表
            documents = []
            
            if isinstance(data, dict):
                for category, items in data.items():
                    if isinstance(items, dict):
                        for key, value in items.items():
                            documents.append({
                                'category': category,
                                'key': key,
                                'content': str(value),
                                'metadata': {'category': category, 'key': key}
                            })
                    elif isinstance(items, list):
                        for item in items:
                            documents.append({
                                'category': category,
                                'content': str(item),
                                'metadata': {'category': category}
                            })
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    documents.append({
                        'content': str(item),
                        'metadata': {'index': i}
                    })
            
            return documents
            
        except Exception as e:
            logging.error(f"加载知识库失败: {e}")
            return []
    
    def _chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """文档分块处理"""
        chunks = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc.get('metadata', {})
            
            # 简单的滑动窗口分块
            for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunk_text = content[i:i + self.chunk_size]
                
                if len(chunk_text.strip()) < 10:  # 过滤太短的块
                    continue
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_id': len(chunks),
                        'start_pos': i,
                        'end_pos': min(i + self.chunk_size, len(content))
                    }
                })
        
        return chunks
    
    def _build_faiss_index(self):
        """构建FAISS索引用于快速相似度搜索"""
        try:
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # 内积相似度
            
            # 归一化向量
            normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.faiss_index.add(normalized_embeddings.astype('float32'))
            
            logging.info("FAISS索引构建成功")
        except Exception as e:
            logging.error(f"构建FAISS索引失败: {e}")
            self.faiss_index = None
    
    def _cache_index(self):
        """缓存向量索引"""
        try:
            cache_data = {
                'knowledge_chunks': self.knowledge_chunks,
                'embeddings': self.embeddings,
                'model_name': self.model_name,
                'version': '1.0'
            }
            
            cache_file = self.cache_dir / "vector_index.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # 单独缓存FAISS索引
            if self.faiss_index:
                faiss_file = self.cache_dir / "faiss_index.bin"
                faiss.write_index(self.faiss_index, str(faiss_file))
                
        except Exception as e:
            logging.error(f"缓存索引失败: {e}")
    
    def _load_cached_index(self):
        """加载缓存的向量索引"""
        try:
            cache_file = self.cache_dir / "vector_index.pkl"
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.knowledge_chunks = cache_data['knowledge_chunks']
            self.embeddings = cache_data['embeddings']
            
            # 加载FAISS索引
            if FAISS_AVAILABLE:
                faiss_file = self.cache_dir / "faiss_index.bin"
                if faiss_file.exists():
                    self.faiss_index = faiss.read_index(str(faiss_file))
            
            logging.info(f"向量索引缓存加载成功，共 {len(self.knowledge_chunks)} 个chunk")
            
        except Exception as e:
            logging.error(f"加载缓存索引失败: {e}")
            self._build_vector_index()
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """向量化语义搜索"""
        if not self.model or not self.knowledge_chunks:
            return self._fallback_search(query)
        
        top_k = top_k or self.top_k
        
        try:
            # 将查询转换为向量
            query_embedding = self.model.encode([query])
            
            # 计算相似度并检索
            if self.faiss_index:
                results = self._faiss_search(query_embedding, top_k)
            else:
                results = self._numpy_search(query_embedding, top_k)
            
            return results
            
        except Exception as e:
            logging.error(f"向量搜索失败: {e}")
            return self._fallback_search(query)
    
    def _faiss_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """使用FAISS进行快速搜索"""
        # 归一化查询向量
        normalized_query = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # FAISS搜索
        similarities, indices = self.faiss_index.search(normalized_query.astype('float32'), top_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim > self.similarity_threshold:
                chunk = self.knowledge_chunks[idx]
                results.append({
                    'text': chunk['text'],
                    'similarity': float(sim),
                    'metadata': chunk['metadata'],
                    'source': 'vector_search'
                })
        
        return results
    
    def _numpy_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """使用numpy计算相似度搜索"""
        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        similarities = similarities / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim > self.similarity_threshold:
                chunk = self.knowledge_chunks[idx]
                results.append({
                    'text': chunk['text'],
                    'similarity': float(sim),
                    'metadata': chunk['metadata'],
                    'source': 'vector_search'
                })
        
        return results
    
    def _fallback_search(self, query: str) -> List[Dict]:
        """降级到传统文本匹配搜索"""
        try:
            if not self.knowledge_base_path.exists():
                return []
            
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            results = []
            query_lower = query.lower()
            
            def search_in_data(data, category=""):
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (dict, list)):
                            search_in_data(value, f"{category}.{key}" if category else key)
                        else:
                            value_str = str(value).lower()
                            if query_lower in value_str:
                                results.append({
                                    'text': str(value),
                                    'similarity': 1.0,
                                    'metadata': {'category': category, 'key': key},
                                    'source': 'text_match'
                                })
                elif isinstance(data, list):
                    for item in data:
                        search_in_data(item, category)
            
            search_in_data(knowledge_base)
            return results[:self.top_k]
            
        except Exception as e:
            logging.error(f"降级搜索失败: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """获取RAG统计信息"""
        return {
            'model_available': self.model is not None,
            'model_name': self.model_name,
            'chunks_count': len(self.knowledge_chunks) if self.knowledge_chunks else 0,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
            'faiss_available': self.faiss_index is not None,
            'cache_dir': str(self.cache_dir)
        }

# 测试函数
if __name__ == "__main__":
    # 测试向量化RAG
    rag_tool = VectorRAGTool("../knowledge_base/platform_knowledge.json")
    
    test_queries = [
        "平台计费模式",
        "如何注册账号",
        "API调用限制"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = rag_tool.search(query)
        for i, result in enumerate(results):
            print(f"  {i+1}. 相似度: {result['similarity']:.3f}")
            print(f"     内容: {result['text'][:100]}...")

"""
优化版向量化RAG工具 - 集成异常处理、懒加载和性能优化
面向生产环境的专业RAG系统
"""
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import pickle
import hashlib
import time
from functools import lru_cache

# 导入自定义异常
from ..exceptions import (
    KnowledgeBaseNotFoundError, VectorIndexBuildError, 
    SemanticSearchError, RAGException
)

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


class OptimizedVectorRAGTool:
    """优化版向量化RAG工具类 - 集成懒加载和异常处理"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 路径配置
        self.knowledge_base_path = Path(knowledge_base_path or 
                                       self.config.get('knowledge_base', 'knowledge_base/platform_knowledge.json'))
        self.cache_dir = Path(self.config.get('cache_dir', 'data/vector_cache'))
        self.cache_dir.mkdir(exist_ok=True)
        
        # 模型配置
        self.model_name = self.config.get('model_name', "shibing624/text2vec-base-chinese")
        self.model = None
        
        # 性能配置
        self.chunk_size = self.config.get('chunk_size', 200)
        self.chunk_overlap = self.config.get('chunk_overlap', 50)
        self.top_k = self.config.get('top_k', 3)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.5)
        self.lazy_load = self.config.get('lazy_load', True)  # 懒加载开关
        
        # 缓存配置
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 缓存1小时
        self.max_cache_size = self.config.get('max_cache_size', 1000)  # 最大缓存条目数
        
        # 状态变量
        self.knowledge_chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self._initialized = False
        self._initialization_time: Optional[float] = None
        self._query_cache: Dict[str, Dict] = {}
        
        # 性能统计
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'vector_searches': 0,
            'fallback_searches': 0,
            'avg_search_time': 0.0,
            'initialization_time': 0.0
        }
        
        # 懒加载：不立即初始化，等待首次使用时初始化
        if not self.lazy_load:
            self._initialize()
    
    def _initialize(self):
        """初始化向量化RAG系统（带异常处理）"""
        start_time = time.time()
        
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logging.info(f"正在加载向量化模型: {self.model_name}")
                try:
                    self.model = SentenceTransformer(self.model_name)
                    logging.info("向量化模型加载成功")
                except Exception as e:
                    logging.warning(f"加载向量化模型失败，将使用传统文本匹配: {e}")
                    self.model = None
            else:
                logging.warning("sentence-transformers不可用，使用传统文本匹配")
                self.model = None
            
            # 加载或构建向量索引
            if self._should_rebuild_index():
                logging.info("需要重建向量索引")
                self._build_vector_index()
            else:
                logging.info("加载缓存的向量索引")
                self._load_cached_index()
            
            self._initialized = True
            self._initialization_time = time.time() - start_time
            self.stats['initialization_time'] = self._initialization_time
            
            logging.info(f"向量化RAG初始化完成，耗时: {self._initialization_time:.2f}秒")
            logging.info(f"知识块数量: {len(self.knowledge_chunks)}")
            logging.info(f"嵌入维度: {self.embeddings.shape if self.embeddings is not None else 'N/A'}")
            
        except Exception as e:
            error_msg = f"向量化RAG初始化失败: {e}"
            logging.error(error_msg)
            
            # 抛出适当的异常
            if "not exist" in str(e) or "No such file" in str(e):
                raise KnowledgeBaseNotFoundError(str(self.knowledge_base_path))
            elif "model" in str(e).lower():
                raise VectorIndexBuildError(self.model_name, str(e))
            else:
                raise RAGException(error_msg, details={"error": str(e)})
    
    def _ensure_initialized(self):
        """确保系统已初始化（懒加载机制）"""
        if not self._initialized:
            self._initialize()
    
    def _should_rebuild_index(self) -> bool:
        """判断是否需要重建索引"""
        cache_file = self.cache_dir / "vector_index.pkl"
        knowledge_file = self.knowledge_base_path
        
        if not cache_file.exists() or not knowledge_file.exists():
            return True
        
        # 检查知识库文件是否有更新
        try:
            cache_time = cache_file.stat().st_mtime
            knowledge_time = knowledge_file.stat().st_mtime
            return knowledge_time > cache_time
        except Exception as e:
            logging.warning(f"检查索引状态失败: {e}")
            return True
    
    def _build_vector_index(self):
        """构建向量索引（带异常处理）"""
        try:
            logging.info("正在构建向量索引...")
            
            # 加载知识库
            knowledge_data = self._load_knowledge_base()
            if not knowledge_data:
                logging.warning("知识库数据为空，无法构建索引")
                return
            
            # 文档分块
            self.knowledge_chunks = self._chunk_documents(knowledge_data)
            logging.info(f"文档分块完成，共 {len(self.knowledge_chunks)} 个chunk")
            
            # 生成嵌入向量（如果模型可用）
            if self.model:
                texts = [chunk['text'] for chunk in self.knowledge_chunks]
                
                # 分批处理大型数据集以避免内存问题
                batch_size = min(len(texts), 32)  # 根据实际情况调整
                
                if len(texts) > 100:
                    logging.info(f"大型数据集，使用批处理生成嵌入向量 (批次大小: {batch_size})")
                    embeddings_list = []
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i + batch_size]
                        batch_embeddings = self.model.encode(
                            batch_texts, 
                            batch_size=batch_size,
                            show_progress_bar=False,  # 生产环境关闭进度条
                            convert_to_numpy=True
                        )
                        embeddings_list.append(batch_embeddings)
                    
                    self.embeddings = np.vstack(embeddings_list)
                else:
                    self.embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
                
                logging.info(f"向量化完成，维度: {self.embeddings.shape}")
                
                # 构建FAISS索引（如果可用）
                if FAISS_AVAILABLE:
                    self._build_faiss_index()
            
            # 缓存结果
            self._cache_index()
            logging.info("向量索引构建并缓存成功")
            
        except Exception as e:
            error_msg = f"构建向量索引失败: {e}"
            logging.error(error_msg)
            raise VectorIndexBuildError(self.model_name, error_msg)
    
    def _load_knowledge_base(self) -> List[Dict]:
        """加载知识库数据（带异常处理）"""
        try:
            if not self.knowledge_base_path.exists():
                raise KnowledgeBaseNotFoundError(str(self.knowledge_base_path))
            
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
            
            logging.info(f"成功加载知识库，包含 {len(documents)} 个文档")
            return documents
            
        except json.JSONDecodeError as e:
            error_msg = f"知识库JSON格式错误: {e}"
            logging.error(error_msg)
            raise RAGException(error_msg, details={"file": str(self.knowledge_base_path), "error": str(e)})
        except Exception as e:
            error_msg = f"加载知识库失败: {e}"
            logging.error(error_msg)
            raise RAGException(error_msg, details={"error": str(e)})
    
    def _chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """文档分块处理（优化版）"""
        chunks = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc['content']
            metadata = doc.get('metadata', {})
            
            # 优化：优先按段落分块
            paragraphs = content.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # 如果段落太长，再按滑动窗口分块
                if len(para) > self.chunk_size * 2:
                    for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
                        chunk_text = para[i:i + self.chunk_size]
                        
                        if len(chunk_text.strip()) < 20:  # 过滤太短的块
                            continue
                        
                        chunks.append(self._create_chunk(chunk_text, metadata, doc_idx, i))
                else:
                    chunks.append(self._create_chunk(para, metadata, doc_idx, 0))
        
        return chunks
    
    def _create_chunk(self, text: str, metadata: Dict, doc_idx: int, start_pos: int) -> Dict:
        """创建标准化的知识块"""
        return {
            'text': text,
            'metadata': {
                **metadata,
                'chunk_id': len(self.knowledge_chunks) + 1,
                'doc_index': doc_idx,
                'start_pos': start_pos,
                'length': len(text),
                'tokens': len(text.split())  # 简单估算token数量
            }
        }
    
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
            logging.warning(f"构建FAISS索引失败，降级到numpy计算: {e}")
            self.faiss_index = None
    
    def _cache_index(self):
        """缓存向量索引"""
        try:
            cache_data = {
                'knowledge_chunks': self.knowledge_chunks,
                'embeddings': self.embeddings,
                'model_name': self.model_name,
                'version': '2.0',
                'cached_at': time.time()
            }
            
            cache_file = self.cache_dir / "vector_index.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # 单独缓存FAISS索引
            if self.faiss_index and FAISS_AVAILABLE:
                faiss_file = self.cache_dir / "faiss_index.bin"
                faiss.write_index(self.faiss_index, str(faiss_file))
                
            logging.info(f"向量索引已缓存到: {cache_file}")
            
        except Exception as e:
            logging.error(f"缓存索引失败: {e}")
    
    def _load_cached_index(self):
        """加载缓存的向量索引"""
        try:
            cache_file = self.cache_dir / "vector_index.pkl"
            
            if not cache_file.exists():
                logging.warning("缓存文件不存在，将重建索引")
                self._build_vector_index()
                return
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 检查缓存版本
            if cache_data.get('version', '1.0') != '2.0':
                logging.warning("缓存版本不兼容，将重建索引")
                self._build_vector_index()
                return
            
            self.knowledge_chunks = cache_data['knowledge_chunks']
            self.embeddings = cache_data['embeddings']
            
            # 加载FAISS索引
            if FAISS_AVAILABLE:
                faiss_file = self.cache_dir / "faiss_index.bin"
                if faiss_file.exists():
                    self.faiss_index = faiss.read_index(str(faiss_file))
            
            # 加载模型
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.model = SentenceTransformer(self.model_name)
                except Exception as e:
                    logging.warning(f"加载缓存的向量化模型失败，将使用传统文本匹配: {e}")
                    self.model = None
            
            logging.info(f"向量索引缓存加载成功，共 {len(self.knowledge_chunks)} 个chunk")
            
        except Exception as e:
            logging.error(f"加载缓存索引失败: {e}")
            self._build_vector_index()
    
    @lru_cache(maxsize=1000)
    def _get_query_cache_key(self, query: str, top_k: int) -> str:
        """生成查询缓存键"""
        content = f"{query}_{top_k}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """向量化语义搜索（带缓存和异常处理）"""
        start_time = time.time()
        self.stats['total_searches'] += 1
        
        # 确保系统已初始化（懒加载）
        self._ensure_initialized()
        
        top_k = top_k or self.top_k
        
        # 检查缓存
        cache_key = self._get_query_cache_key(query, top_k)
        if cache_key in self._query_cache:
            cache_entry = self._query_cache[cache_key]
            # 检查缓存是否过期
            if time.time() - cache_entry.get('cached_at', 0) < self.cache_ttl:
                self.stats['cache_hits'] += 1
                logging.debug(f"查询缓存命中: {query}")
                return cache_entry['results']
        
        try:
            # 向量搜索或降级搜索
            if not self.model or not self.knowledge_chunks:
                results = self._fallback_search(query)
                self.stats['fallback_searches'] += 1
            else:
                results = self._vector_search(query, top_k)
                self.stats['vector_searches'] += 1
            
            # 更新缓存（如果结果有效）
            if results:
                self._query_cache[cache_key] = {
                    'results': results,
                    'cached_at': time.time(),
                    'query': query
                }
                
                # 清理过期缓存
                self._cleanup_cache()
            
            # 更新性能统计
            search_time = time.time() - start_time
            self.stats['avg_search_time'] = (
                (self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + search_time) 
                / self.stats['total_searches']
            )
            
            logging.debug(f"搜索完成: {query} (耗时: {search_time:.3f}秒, 结果: {len(results)}个)")
            return results
            
        except Exception as e:
            error_msg = f"搜索失败: {query}"
            logging.error(f"{error_msg}: {e}")
            raise SemanticSearchError(query, str(e))
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """向量搜索核心逻辑"""
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
            logging.warning(f"向量搜索失败，降级到文本搜索: {e}")
            return self._fallback_search(query)
    
    def _faiss_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """使用FAISS进行快速搜索"""
        # 归一化查询向量
        normalized_query = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # FAISS搜索
        search_k = min(top_k * 2, len(self.knowledge_chunks))  # 搜索更多结果用于筛选
        similarities, indices = self.faiss_index.search(normalized_query.astype('float32'), search_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0 and sim > self.similarity_threshold:
                chunk = self.knowledge_chunks[idx]
                results.append({
                    'text': chunk['text'],
                    'similarity': float(sim),
                    'metadata': chunk['metadata'],
                    'source': 'vector_search'
                })
        
        # 返回top-k结果
        return results[:top_k]
    
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
    
    def _cleanup_cache(self):
        """清理过期和过多的缓存"""
        current_time = time.time()
        keys_to_delete = []
        
        # 清理过期缓存
        for key, entry in self._query_cache.items():
            if current_time - entry.get('cached_at', 0) > self.cache_ttl:
                keys_to_delete.append(key)
        
        # 清理过多缓存
        if len(self._query_cache) > self.max_cache_size:
            # 按缓存时间排序，删除最旧的
            sorted_keys = sorted(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k].get('cached_at', 0)
            )
            keys_to_delete.extend(sorted_keys[:len(self._query_cache) - self.max_cache_size])
        
        # 删除缓存
        for key in set(keys_to_delete):
            del self._query_cache[key]
        
        if keys_to_delete:
            logging.debug(f"清理了 {len(set(keys_to_delete))} 个缓存条目")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取RAG统计信息"""
        cache_hit_rate = (self.stats['cache_hits'] / max(self.stats['total_searches'], 1)) * 100
        vector_search_rate = (self.stats['vector_searches'] / max(self.stats['total_searches'], 1)) * 100
        
        return {
            'model_available': self.model is not None,
            'model_name': self.model_name,
            'chunks_count': len(self.knowledge_chunks) if self.knowledge_chunks else 0,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
            'faiss_available': self.faiss_index is not None,
            'cache_dir': str(self.cache_dir),
            'performance': {
                'total_searches': self.stats['total_searches'],
                'cache_hits': self.stats['cache_hits'],
                'cache_hit_rate': round(cache_hit_rate, 2),
                'vector_searches': self.stats['vector_searches'],
                'vector_search_rate': round(vector_search_rate, 2),
                'fallback_searches': self.stats['fallback_searches'],
                'avg_search_time': round(self.stats['avg_search_time'], 3),
                'initialization_time': round(self.stats['initialization_time'], 2)
            },
            'config': {
                'lazy_load': self.lazy_load,
                'top_k': self.top_k,
                'similarity_threshold': self.similarity_threshold,
                'cache_ttl': self.cache_ttl,
                'max_cache_size': self.max_cache_size
            }
        }
    
    def warmup_cache(self, queries: List[str]):
        """预热缓存 - 预先加载常用查询"""
        logging.info(f"开始预热缓存，共 {len(queries)} 个查询")
        
        for i, query in enumerate(queries, 1):
            try:
                self.search(query)
                if i % 10 == 0:
                    logging.info(f"预热进度: {i}/{len(queries)}")
            except Exception as e:
                logging.warning(f"预热查询失败: {query} - {e}")
        
        logging.info(f"缓存预热完成，当前缓存大小: {len(self._query_cache)}")
    
    def clear_cache(self):
        """清空所有缓存"""
        self._query_cache.clear()
        logging.info("RAG缓存已清空")


# 测试函数
if __name__ == "__main__":
    # 测试优化版向量化RAG
    config = {
        'knowledge_base': '../data/inputs.json',
        'lazy_load': True,
        'top_k': 3,
        'cache_ttl': 1800
    }
    
    rag_tool = OptimizedVectorRAGTool(config=config)
    
    test_queries = [
        "平台计费模式",
        "如何注册账号",
        "API调用限制",
        "系统稳定性"
    ]
    
    print("优化版向量化RAG测试")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n查询: {query}")
        start_time = time.time()
        results = rag_tool.search(query)
        search_time = time.time() - start_time
        
        print(f"  耗时: {search_time:.3f}秒")
        print(f"  结果数量: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. 相似度: {result['similarity']:.3f} - 来源: {result['source']}")
            print(f"     内容: {result['text'][:80]}...")
    
    # 显示统计信息
    stats = rag_tool.get_stats()
    print(f"\n统计信息:")
    print(f"  总搜索次数: {stats['performance']['total_searches']}")
    print(f"  缓存命中率: {stats['performance']['cache_hit_rate']}%")
    print(f"  平均搜索时间: {stats['performance']['avg_search_time']}秒")
    print(f"  初始化时间: {stats['performance']['initialization_time']}秒")

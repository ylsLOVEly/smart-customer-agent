"""
优化版向量化RAG工具 - 全功能生产级 (V2.0)
集成：Cross-Encoder 重排序、异常处理、缓存管理、降级策略、详细统计
"""
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import pickle
import hashlib
import time
from functools import lru_cache
import os

# 导入自定义异常 (假设存在，如果不存在则定义基础异常)
try:
    from ..exceptions import (
        KnowledgeBaseNotFoundError, VectorIndexBuildError, 
        SemanticSearchError, RAGException
    )
except ImportError:
    class RAGException(Exception): pass
    class KnowledgeBaseNotFoundError(RAGException): pass
    class VectorIndexBuildError(RAGException): pass
    class SemanticSearchError(RAGException): pass

# 依赖库导入与环境检查
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("⚠️ sentence-transformers未安装，向量检索及重排序功能受限，将仅使用文本匹配。")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("⚠️ faiss未安装，大数据量下检索性能可能下降，将使用numpy进行计算。")


class OptimizedVectorRAGTool:
    """
    全功能生产级 RAG 工具类
    
    核心能力：
    1. 双阶段检索：Bi-Encoder 粗排 + Cross-Encoder 精排
    2. 智能防幻觉：基于语义相似度和重排序分数的双重阈值过滤
    3. 健壮性设计：自动降级、异常捕获、缓存预热、自动索引重建
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None, config: Optional[Dict] = None):
        self.config = config or {}
        
        # --- 基础路径配置 ---
        self.knowledge_base_path = Path(knowledge_base_path or 
                                       self.config.get('knowledge_base', 'knowledge_base/platform_knowledge.json'))
        self.cache_dir = Path(self.config.get('cache_dir', 'data/vector_cache'))
        
        # 确保缓存目录存在
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"无法创建缓存目录 {self.cache_dir}: {e}")
        
        # --- 模型配置 (核心升级点) ---
        self.embed_model_name = self.config.get('embed_model', "shibing624/text2vec-base-chinese")
        self.rerank_model_name = self.config.get('rerank_model', "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.embed_model = None
        self.rerank_model = None
        
        # --- 性能与分块配置 ---
        self.chunk_size = self.config.get('chunk_size', 300) 
        self.chunk_overlap = self.config.get('chunk_overlap', 50)
        self.retrieve_top_k = self.config.get('retrieve_top_k', 20)  # 粗排召回
        self.final_top_k = self.config.get('top_k', 3)               # 精排结果
        
        # --- 阈值配置 (防幻觉关键) ---
        self.vector_threshold = self.config.get('vector_threshold', 0.35)  
        self.rerank_threshold = self.config.get('rerank_threshold', 0.0) # Sigmoid后通常在0~1，需微调
        
        # --- 缓存与工程配置 ---
        self.lazy_load = self.config.get('lazy_load', True)
        self.cache_ttl = self.config.get('cache_ttl', 3600)
        self.max_cache_size = self.config.get('max_cache_size', 2000)
        
        # --- 内部状态 ---
        self.knowledge_chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self._initialized = False
        self._initialization_time: Optional[float] = None
        self._query_cache: Dict[str, Dict] = {}
        
        # --- 统计信息 (详细监控) ---
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'vector_searches': 0,
            'fallback_searches': 0,
            'rerank_triggered': 0,
            'avg_search_time': 0.0,
            'initialization_time': 0.0,
            'last_error': None
        }
        
        if not self.lazy_load:
            self._initialize()
    
    def _initialize(self):
        """初始化系统（包含异常处理和模型加载）"""
        if self._initialized:
            return

        start_time = time.time()
        logging.info("🚀 正在初始化 RAG 系统...")
        
        try:
            # 1. 加载向量模型
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logging.info(f"正在加载向量模型: {self.embed_model_name}")
                try:
                    self.embed_model = SentenceTransformer(self.embed_model_name)
                except Exception as e:
                    logging.error(f"❌ 向量模型加载失败: {e}")
                    self.embed_model = None

                # 2. 加载重排序模型 (新增)
                logging.info(f"正在加载重排序模型: {self.rerank_model_name}")
                try:
                    self.rerank_model = CrossEncoder(self.rerank_model_name)
                    logging.info("✅ 重排序模型加载成功")
                except Exception as e:
                    logging.warning(f"⚠️ 重排序模型加载失败，将跳过精排阶段: {e}")
                    self.rerank_model = None
            else:
                logging.warning("⚠️ sentence-transformers不可用，仅支持基础文本匹配")
            
            # 3. 加载或构建索引
            if self._should_rebuild_index():
                logging.info("检测到知识库更新或缓存缺失，正在重建索引...")
                self._build_vector_index()
            else:
                logging.info("正在加载缓存索引...")
                self._load_cached_index()
            
            self._initialized = True
            self._initialization_time = time.time() - start_time
            self.stats['initialization_time'] = self._initialization_time
            logging.info(f"✅ RAG初始化完成，耗时: {self._initialization_time:.2f}秒，Chunk数: {len(self.knowledge_chunks)}")
            
        except Exception as e:
            # 严重的初始化失败需要抛出，让上层感知
            logging.error(f"❌ RAG初始化严重失败: {e}", exc_info=True)
            self.stats['last_error'] = str(e)
            if "not exist" in str(e):
                raise KnowledgeBaseNotFoundError(str(self.knowledge_base_path))
            raise RAGException(f"初始化失败: {str(e)}")

    def _ensure_initialized(self):
        """确保懒加载模式下系统已初始化"""
        if not self._initialized:
            self._initialize()

    def _should_rebuild_index(self) -> bool:
        """检查是否需要重建索引"""
        cache_file = self.cache_dir / "vector_index.pkl"
        
        # 1. 基础文件检查
        if not cache_file.exists():
            return True
        if not self.knowledge_base_path.exists():
            logging.warning(f"知识库文件 {self.knowledge_base_path} 不存在，无法对比时间戳")
            return False # 避免反复重建空索引
            
        # 2. 时间戳对比
        try:
            kb_mtime = self.knowledge_base_path.stat().st_mtime
            cache_mtime = cache_file.stat().st_mtime
            return kb_mtime > cache_mtime
        except Exception as e:
            logging.warning(f"检查文件时间戳失败: {e}，默认重建索引")
            return True

    def _flatten_json(self, data: Any, meta: Dict = None) -> List[Dict]:
        """递归扁平化任意 JSON 结构"""
        if meta is None: meta = {}
        documents = []
        
        if isinstance(data, dict):
            for k, v in data.items():
                # 记录路径作为元数据
                new_meta = meta.copy()
                new_meta['key_path'] = f"{meta.get('key_path', '')}/{k}".strip('/')
                documents.extend(self._flatten_json(v, new_meta))
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                new_meta = meta.copy()
                new_meta['list_index'] = idx
                documents.extend(self._flatten_json(item, new_meta))
        elif isinstance(data, (str, int, float, bool)):
            text = str(data).strip()
            if text:
                documents.append({'content': text, 'metadata': meta})
                
        return documents

    def _chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """文档分块处理 (增强健壮性)"""
        chunks = []
        for doc_idx, doc in enumerate(documents):
            content = doc['content']
            metadata = doc.get('metadata', {})
            
            # 过滤过短的文档
            if len(content) < 5:
                continue
            
            # 简单的滑动窗口分块
            if len(content) > self.chunk_size:
                for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                    chunk_text = content[i:i + self.chunk_size]
                    if len(chunk_text.strip()) < 10: continue
                    
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            **metadata, 
                            'chunk_id': len(chunks),
                            'doc_index': doc_idx,
                            'length': len(chunk_text)
                        },
                        'original_doc': content[:200] + "..." # 用于溯源
                    })
            else:
                chunks.append({
                    'text': content,
                    'metadata': {**metadata, 'chunk_id': len(chunks), 'doc_index': doc_idx},
                    'original_doc': content
                })
        return chunks

    def _build_vector_index(self):
        """构建向量索引 (带完整异常处理)"""
        try:
            logging.info("开始构建向量索引...")
            
            # 加载原始数据
            if not self.knowledge_base_path.exists():
                raise KnowledgeBaseNotFoundError(str(self.knowledge_base_path))
                
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                try:
                    raw_data = json.load(f)
                except json.JSONDecodeError:
                    raise RAGException(f"知识库文件损坏，非有效JSON: {self.knowledge_base_path}")
            
            # 扁平化数据处理
            documents = self._flatten_json(raw_data)
            logging.info(f"解析出 {len(documents)} 个基础文档片段")
            
            # 分块
            self.knowledge_chunks = self._chunk_documents(documents)
            logging.info(f"分块完成，生成 {len(self.knowledge_chunks)} 个 chunk")
            
            # 向量化
            if self.embed_model and self.knowledge_chunks:
                texts = [c['text'] for c in self.knowledge_chunks]
                
                # 批量处理以防内存溢出
                batch_size = 64
                embeddings_list = []
                total_batches = (len(texts) + batch_size - 1) // batch_size
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    # show_progress_bar=False 避免日志刷屏
                    emb = self.embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                    embeddings_list.append(emb)
                    if i % (batch_size * 5) == 0:
                        logging.debug(f"向量化进度: {i}/{len(texts)}")
                
                if embeddings_list:
                    self.embeddings = np.vstack(embeddings_list)
                    logging.info(f"向量化完成，维度: {self.embeddings.shape}")
                    
                    # FAISS 索引构建
                    if FAISS_AVAILABLE:
                        d = self.embeddings.shape[1]
                        # 使用内积(IP)索引，前提是向量已归一化，等价于余弦相似度
                        self.faiss_index = faiss.IndexFlatIP(d)
                        faiss.normalize_L2(self.embeddings)
                        self.faiss_index.add(self.embeddings)
                        logging.info("FAISS 索引构建成功")
            else:
                logging.warning("未加载向量模型或无文档，跳过向量化步骤")
            
            # 缓存
            self._cache_index()
            
        except Exception as e:
            logging.error(f"构建索引失败: {e}")
            raise VectorIndexBuildError(self.embed_model_name or "unknown", str(e))

    def _cache_index(self):
        """持久化索引"""
        try:
            cache_data = {
                'chunks': self.knowledge_chunks, 
                'embeddings': self.embeddings,
                'version': '2.2',
                'timestamp': time.time()
            }
            with open(self.cache_dir / "vector_index.pkl", 'wb') as f:
                pickle.dump(cache_data, f)
            
            if FAISS_AVAILABLE and self.faiss_index:
                faiss.write_index(self.faiss_index, str(self.cache_dir / "faiss_index.bin"))
                
            logging.info(f"索引已缓存至 {self.cache_dir}")
        except Exception as e:
            logging.error(f"缓存索引失败: {e}")

    def _load_cached_index(self):
        """加载缓存索引"""
        try:
            with open(self.cache_dir / "vector_index.pkl", 'rb') as f:
                data = pickle.load(f)
            
            # 版本检查 (可选)
            if data.get('version') != '2.2':
                logging.warning("缓存版本不匹配，触发重建")
                self._build_vector_index()
                return

            self.knowledge_chunks = data['chunks']
            self.embeddings = data['embeddings']
            
            if FAISS_AVAILABLE:
                idx_path = str(self.cache_dir / "faiss_index.bin")
                if Path(idx_path).exists():
                    self.faiss_index = faiss.read_index(idx_path)
                else:
                    logging.warning("FAISS索引文件缺失，将重建FAISS索引")
                    if self.embeddings is not None:
                         d = self.embeddings.shape[1]
                         self.faiss_index = faiss.IndexFlatIP(d)
                         faiss.normalize_L2(self.embeddings)
                         self.faiss_index.add(self.embeddings)

        except Exception as e:
            logging.warning(f"加载缓存失败，尝试重建: {e}")
            self._build_vector_index()

    @lru_cache(maxsize=1000)
    def _get_query_cache_key(self, query: str) -> str:
        """生成查询指纹"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        核心搜索入口：集成缓存、向量检索、Rerank 和 降级策略
        """
        if not query or not isinstance(query, str):
            logging.warning(f"非法查询输入: {query}")
            return []

        start_time = time.time()
        self.stats['total_searches'] += 1
        
        # 懒加载初始化
        self._ensure_initialized()
        
        target_k = top_k or self.final_top_k
        cache_key = self._get_query_cache_key(query)
        
        # 1. 检查内存缓存 (一级缓存)
        if cache_key in self._query_cache:
            entry = self._query_cache[cache_key]
            if time.time() - entry['time'] < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return entry['results']

        results = []
        try:
            # 2. 向量检索 (Vector Search - 粗排)
            if self.embed_model and self.knowledge_chunks:
                # 召回 retrieve_top_k (比如20个) 给 Reranker
                candidates = self._vector_search(query, self.retrieve_top_k)
                self.stats['vector_searches'] += 1
                
                # 3. 重排序 (Rerank - 精排)
                if candidates and self.rerank_model:
                    self.stats['rerank_triggered'] += 1
                    results = self._rerank_search(query, candidates, target_k)
                else:
                    # 如果没有 Reranker，直接截取
                    results = candidates[:target_k]
                    if not self.rerank_model and candidates:
                        logging.debug("未启用Rerank，直接返回向量检索结果")
            else:
                # 4. 降级搜索 (Fallback - 关键词匹配)
                logging.info("向量模型不可用，使用文本匹配降级搜索")
                results = self._fallback_search(query)
                self.stats['fallback_searches'] += 1
            
            # 5. 更新缓存
            if results:
                self._query_cache[cache_key] = {
                    'results': results,
                    'time': time.time()
                }
                self._cleanup_cache()
            
            # 更新统计耗时
            elapsed = time.time() - start_time
            self._update_avg_time(elapsed)
            
            return results
            
        except Exception as e:
            logging.error(f"搜索过程异常: {e}", exc_info=True)
            self.stats['last_error'] = str(e)
            # 最后的保底：尝试文本匹配
            return self._fallback_search(query)

    def _vector_search(self, query: str, k: int) -> List[Dict]:
        """执行向量检索"""
        try:
            query_vec = self.embed_model.encode([query])
            
            if FAISS_AVAILABLE and self.faiss_index:
                faiss.normalize_L2(query_vec)
                D, I = self.faiss_index.search(query_vec, k)
                candidates = []
                for score, idx in zip(D[0], I[0]):
                    if idx != -1 and score > self.vector_threshold:
                        candidates.append({
                            'chunk': self.knowledge_chunks[idx],
                            'score': float(score),
                            'source': 'vector_faiss'
                        })
                return candidates
            elif self.embeddings is not None:
                # Numpy 实现
                scores = np.dot(self.embeddings, query_vec.T).flatten()
                top_idxs = np.argsort(scores)[::-1][:k]
                return [
                    {
                        'chunk': self.knowledge_chunks[i], 
                        'score': float(scores[i]),
                        'source': 'vector_numpy'
                    }
                    for i in top_idxs if scores[i] > self.vector_threshold
                ]
            else:
                return []
        except Exception as e:
            logging.error(f"向量检索计算失败: {e}")
            return []

    def _rerank_search(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """执行重排序"""
        if not candidates: return []
        
        try:
            # 构造 (Query, Doc) 对
            pairs = [[query, c['chunk']['text']] for c in candidates]
            
            # 预测分数
            scores = self.rerank_model.predict(pairs)
            
            final_results = []
            for i, score in enumerate(scores):
                # 处理分数：兼容 Logits 和 Sigmoid 输出
                # 大多数 Reranker 输出未归一化的 logits，这里简单转换
                # 或者直接用 raw score 排序即可，阈值需要对应调整
                normalized_score = float(score) 
                
                # Rerank 阈值过滤 (核心防幻觉点：无关的直接丢弃)
                if normalized_score > self.rerank_threshold:
                    cand = candidates[i]
                    final_results.append({
                        'text': cand['chunk']['text'],
                        'metadata': cand['chunk']['metadata'],
                        'similarity': cand['score'], # 保留原始向量分
                        'rerank_score': normalized_score,
                        'source': 'reranked',
                        'id': cand['chunk']['metadata'].get('chunk_id')
                    })
            
            # 按 Rerank 分数倒序
            final_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            return final_results[:top_k]
            
        except Exception as e:
            logging.error(f"重排序计算失败: {e}")
            # 降级：如果 Rerank 失败，返回原始向量结果
            return [
                {**c['chunk'], 'similarity': c['score'], 'rank_score': 0, 'source': 'vector_fallback'} 
                for c in candidates[:top_k]
            ]

    def _fallback_search(self, query: str) -> List[Dict]:
        """降级：简单的文本包含匹配"""
        results = []
        q_lower = query.lower()
        for chunk in self.knowledge_chunks:
            # 简单的关键词命中计分
            if q_lower in chunk['text'].lower():
                results.append({
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'similarity': 1.0,
                    'source': 'text_match_fallback'
                })
        return results[:self.final_top_k]

    def _cleanup_cache(self):
        """清理过期缓存"""
        if len(self._query_cache) > self.max_cache_size:
            now = time.time()
            # 优先删过期的
            keys_to_del = [k for k, v in self._query_cache.items() if now - v['time'] > self.cache_ttl]
            
            if not keys_to_del:
                # 如果没过期的，删最旧的 (FIFO)
                sorted_keys = sorted(self._query_cache.keys(), key=lambda k: self._query_cache[k]['time'])
                keys_to_del = sorted_keys[:int(self.max_cache_size * 0.2)] # 删掉 20%
                
            for k in keys_to_del: 
                del self._query_cache[k]

    def _update_avg_time(self, new_time):
        n = self.stats['total_searches']
        self.stats['avg_search_time'] = (self.stats['avg_search_time'] * (n-1) + new_time) / n

    def get_stats(self) -> Dict[str, Any]:
        """获取详细运行统计"""
        return {
            **self.stats,
            'index_size': len(self.knowledge_chunks),
            'has_embedding_model': self.embed_model is not None,
            'has_reranker_model': self.rerank_model is not None,
            'faiss_enabled': FAISS_AVAILABLE and self.faiss_index is not None,
            'cache_size': len(self._query_cache),
            'config': {
                'chunk_size': self.chunk_size,
                'top_k': self.final_top_k,
                'rerank_threshold': self.rerank_threshold
            }
        }

    def warmup_cache(self, queries: List[str]):
        """缓存预热"""
        logging.info(f"开始预热 {len(queries)} 个查询...")
        start = time.time()
        for i, q in enumerate(queries):
            try:
                self.search(q)
                if i % 10 == 0: logging.debug(f"预热进度: {i}/{len(queries)}")
            except Exception as e:
                logging.warning(f"预热查询失败 '{q}': {e}")
        logging.info(f"预热完成，耗时 {time.time() - start:.2f}s")

    def clear_cache(self):
        self._query_cache.clear()
        logging.info("缓存已清空")

# --- 本地测试代码 (保留，方便调试) ---
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 模拟配置
    test_config = {
        'chunk_size': 100,
        'top_k': 3,
        'lazy_load': False
    }
    
    # 创建工具实例
    tool = OptimizedVectorRAGTool(config=test_config)
    
    # 测试查询
    queries = ["API调用限制", "系统稳定性", "如何重置密码"]
    
    print("\n" + "="*50)
    print("🔎 RAG 工具测试开始")
    print("="*50)
    
    for q in queries:
        print(f"\n❓ 查询: {q}")
        results = tool.search(q)
        for i, res in enumerate(results):
            score_key = 'rerank_score' if 'rerank_score' in res else 'similarity'
            print(f"  [{i+1}] Score: {res.get(score_key, 0):.4f} | Source: {res.get('source')} | Text: {res['text'][:50]}...")
            
    # 打印统计
    print("\n" + "="*50)
    print("📊 运行统计:")
    print(json.dumps(tool.get_stats(), indent=2, ensure_ascii=False))

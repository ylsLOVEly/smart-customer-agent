import logging
import asyncio
import json
import time
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import httpx
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# 尝试导入 tiktoken，如果不可用则使用回退方案
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not installed, using fallback token counting method.")

@dataclass
class CacheEntry:
    """缓存条目数据类"""
    response: str
    timestamp: float
    usage: Dict[str, Any]

class DeepSeekClient:
    """增强版DeepSeek客户端，解决并发、格式和网络依赖问题"""
    
    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.base_url = DEEPSEEK_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 并发控制
        self._semaphore = asyncio.Semaphore(3)  # 限制并发数量
        self._last_request_time = {}  # 每个模型的最后请求时间
        self._min_interval = 1.0  # 最小请求间隔(秒)
        
        # 缓存机制
        self._cache_dir = Path("data/cache")
        self._cache_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_ttl = 3600  # 缓存1小时
        
        # 输出格式规范
        self._format_validators = {
            'json': self._validate_json_format,
            'text': self._validate_text_format
        }
        
        # 网络状态监控
        self._network_status = "unknown"
        self._consecutive_failures = 0
        self._max_failures = 3
        
        # 离线知识库路径
        self._offline_responses = self._load_offline_responses()
        
    def _load_offline_responses(self) -> Dict[str, str]:
        """加载离线应急回复"""
        offline_file = Path("agent/knowledge_base/offline_responses.json")
        if offline_file.exists():
            try:
                with open(offline_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"加载离线回复失败: {e}")
        
        return {
            "default": "系统暂时无法连接到AI服务，请稍后重试。我将基于知识库为您提供帮助。",
            "system_status": "抱歉，当前无法获取实时系统状态，请联系技术支持。",
            "billing": "计费相关问题请参考知识库文档或联系客服。",
            "error": "系统遇到临时问题，正在自动修复中，请稍后重试。"
        }
    
    def _generate_cache_key(self, model: str, messages: List[Dict], temperature: float) -> str:
        """生成缓存键"""
        content = f"{model}_{messages}_{temperature}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[CacheEntry]:
        """从缓存获取回复"""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry.timestamp < self._cache_ttl:
                logging.info("使用缓存回复")
                return entry
        return None
    
    def _save_to_cache(self, cache_key: str, response: str, usage: Dict[str, Any]):
        """保存到缓存"""
        entry = CacheEntry(
            response=response,
            timestamp=time.time(),
            usage=usage
        )
        self._cache[cache_key] = entry
        
        # 持久化缓存
        cache_file = self._cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'response': response,
                    'timestamp': entry.timestamp,
                    'usage': usage
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"缓存持久化失败: {e}")
    
    async def _wait_for_rate_limit(self, model: str):
        """等待限流间隔"""
        if model in self._last_request_time:
            elapsed = time.time() - self._last_request_time[model]
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                logging.info(f"等待限流间隔: {wait_time:.2f}秒")
                await asyncio.sleep(wait_time)
        
        self._last_request_time[model] = time.time()
    
    async def _retry_with_backoff(self, func, max_retries: int = 3) -> Optional[Any]:
        """带指数退避的重试机制"""
        for attempt in range(max_retries):
            try:
                result = await func()
                if result is not None:
                    self._consecutive_failures = 0  # 重置失败计数
                    self._network_status = "healthy"
                    return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:  # 服务不可用
                    wait_time = (2 ** attempt) + (attempt * 0.5)  # 指数退避
                    logging.warning(f"503错误，第{attempt+1}次重试，等待{wait_time:.1f}秒")
                    await asyncio.sleep(wait_time)
                    continue
                elif e.response.status_code == 429:  # 限流
                    wait_time = (2 ** attempt) * 2  # 更长的等待
                    logging.warning(f"429限流，第{attempt+1}次重试，等待{wait_time:.1f}秒")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logging.error(f"HTTP错误 {e.response.status_code}: {e}")
                    break
            except Exception as e:
                wait_time = (2 ** attempt) + (attempt * 0.5)
                logging.warning(f"请求异常第{attempt+1}次重试: {e}, 等待{wait_time:.1f}秒")
                await asyncio.sleep(wait_time)
                continue
        
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_failures:
            self._network_status = "degraded"
        
        return None
    
    def _validate_json_format(self, response: str) -> Optional[str]:
        """验证并修复JSON格式"""
        try:
            # 尝试解析JSON
            json.loads(response)
            return response
        except json.JSONDecodeError:
            # 尝试修复常见的JSON格式问题
            try:
                # 移除markdown代码块标记
                cleaned = response.strip()
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                # 验证修复后的JSON
                json.loads(cleaned)
                logging.info("JSON格式已自动修复")
                return cleaned
            except:
                logging.warning("无法修复JSON格式，返回原始回复")
                return response
    
    def _validate_text_format(self, response: str) -> str:
        """验证并清理文本格式"""
        if not response or not response.strip():
            return "抱歉，回复内容为空，请重试。"
        
        # 移除多余空白字符
        cleaned = ' '.join(response.split())
        
        # 确保回复长度合理
        if len(cleaned) > 2000:
            cleaned = cleaned[:1950] + "..."
            logging.info("回复已截断到合理长度")
        
        return cleaned
    
    def _format_response(self, response: str, expected_format: str = 'text') -> str:
        """格式化回复内容"""
        if not response:
            return self._offline_responses.get("default", "回复为空")
        
        validator = self._format_validators.get(expected_format, self._validate_text_format)
        return validator(response)
    
    def _get_offline_response(self, messages: List[Dict]) -> str:
        """获取离线应急回复"""
        # 根据用户问题类型返回合适的离线回复
        if not messages:
            return self._offline_responses.get("default")
        
        user_query = messages[-1].get("content", "").lower()
        
        if any(word in user_query for word in ["系统", "稳定", "状态", "监控"]):
            return self._offline_responses.get("system_status")
        elif any(word in user_query for word in ["计费", "收费", "价格", "费用"]):
            return self._offline_responses.get("billing")
        elif any(word in user_query for word in ["错误", "异常", "问题"]):
            return self._offline_responses.get("error")
        else:
            return self._offline_responses.get("default")
    
    async def _make_api_call(self, model: str, messages: list, temperature: float) -> Optional[Dict[str, Any]]:
        """执行API调用的核心方法"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        timeout_config = httpx.Timeout(
            connect=10.0,  # 连接超时
            read=30.0,     # 读取超时
            write=10.0,    # 写入超时
            pool=60.0      # 连接池超时
        )
        
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()  # 抛出HTTP错误
            
            result = response.json()
            usage = result.get("usage", {})
            logging.info(f"模型 {model} 调用成功，Token使用: {usage}")
            return result
    
    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if TIKTOKEN_AVAILABLE:
            try:
                # 使用tiktoken编码器，deepseek模型使用cl100k_base
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception as e:
                logging.warning(f"tiktoken计数失败，使用回退方案: {e}")
        
        # 回退方案：按字符估算（通常一个token≈4个字符）
        return len(text) // 4
    
    def _truncate_messages_to_token_limit(self, messages: list, max_tokens: int = 98304) -> list:
        """根据token限制截断消息内容"""
        total_tokens = 0
        truncated_messages = []
        
        for message in reversed(messages):  # 从最新消息开始
            content = message.get('content', '')
            tokens = self._count_tokens(content)
            
            # 如果添加这条消息会超过限制，则截断内容
            if total_tokens + tokens > max_tokens:
                # 计算还允许的token数量
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens <= 0:
                    break
                
                # 简单截断内容（可按需改进为智能截断）
                # 这里采用简单字符截断，实际应基于token截断
                chars_to_keep = remaining_tokens * 4  # 粗略估计
                truncated_content = content[:chars_to_keep] + "...[内容已截断]"
                message = message.copy()
                message['content'] = truncated_content
                tokens = self._count_tokens(truncated_content)
            
            truncated_messages.insert(0, message)  # 保持原始顺序
            total_tokens += tokens
        
        logging.info(f"消息已从 {len(messages)} 条截断至 {len(truncated_messages)} 条，token使用: {total_tokens}")
        return truncated_messages
    
    def _validate_api_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        """验证API响应数据"""
        if not response_data:
            return "API返回空响应"
        
        if 'choices' not in response_data or not response_data['choices']:
            return "API响应中没有choices字段"
        
        choice = response_data['choices'][0]
        if 'message' not in choice:
            return "choices中没有message字段"
        
        message = choice['message']
        if 'content' not in message or not message['content']:
            return "message中没有content字段或内容为空"
        
        return None
    
    async def call_model(self, model: str, messages: list, temperature: float = 0.7, 
                        expected_format: str = 'text') -> Optional[str]:
        """增强版模型调用方法"""
        # 1. 检查缓存
        cache_key = self._generate_cache_key(model, messages, temperature)
        cached_entry = self._get_from_cache(cache_key)
        if cached_entry:
            return self._format_response(cached_entry.response, expected_format)
        
        # 2. 网络状态检查
        if self._network_status == "degraded":
            logging.warning("网络状况不佳，直接返回离线回复")
            return self._get_offline_response(messages)
        
        # 3. 检查token限制并截断
        truncated_messages = self._truncate_messages_to_token_limit(messages, max_tokens=98304)
        
        # 4. 并发控制
        async with self._semaphore:
            # 5. 限流等待
            await self._wait_for_rate_limit(model)
            
            # 6. DeepSeek系列模型备份策略（符合比赛单模型约束）
            model_variants = [
                model,
                "deepseek/deepseek-v3.2",
                "deepseek/deepseek-v3.2-think", 
                "deepseek/deepseek-v3.1"  # 仅使用真实存在的DeepSeek系列模型
            ]
            
            for model_to_try in model_variants:
                logging.info(f"尝试调用模型: {model_to_try}")
                
                # 7. 带重试的API调用
                async def api_call():
                    result = await self._make_api_call(model_to_try, truncated_messages, temperature)
                    # 验证API响应
                    try:
                        if result is None:
                            return None
                        validation_error = self._validate_api_response(result)
                        if validation_error:
                            logging.error(f"API响应验证失败: {validation_error}")
                            return None
                        content = result["choices"][0]["message"]["content"]
                        return content
                    except (KeyError, IndexError) as e:
                        logging.error(f"API响应格式错误: {e}, 响应: {result}")
                        return None
                
                result = await self._retry_with_backoff(api_call)
                
                if result:
                    # 8. 格式验证和修复
                    formatted_result = self._format_response(result, expected_format)
                    
                    # 9. 保存到缓存
                    self._save_to_cache(cache_key, formatted_result, {})
                    
                    return formatted_result
                
                # 如果当前模型失败，尝试下一个
                logging.warning(f"模型 {model_to_try} 调用失败，尝试下一个")
                await asyncio.sleep(0.5)  # 短暂等待避免连续失败
            
            # 10. 所有模型都失败，返回离线回复
            logging.error(f"所有模型调用失败，原始请求模型: {model}")
            return self._get_offline_response(messages)
    
    def get_network_status(self) -> Dict[str, Any]:
        """获取网络状态信息"""
        return {
            "status": self._network_status,
            "consecutive_failures": self._consecutive_failures,
            "cache_entries": len(self._cache)
        }
    
    def clear_cache(self):
        """清理缓存"""
        self._cache.clear()
        # 清理磁盘缓存
        for cache_file in self._cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logging.warning(f"清理缓存文件失败: {e}")
        logging.info("缓存已清理")

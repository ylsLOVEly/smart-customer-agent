"""
统一配置管理中心 - 支持配置文件、环境变量、动态热重载
为智能客服监控Agent提供专业的配置管理
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 导入环境变量加载器
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class ModelConfig:
    """模型配置数据类"""
    name: str = "deepseek/deepseek-v3.2-think"
    base_url: str = "https://router.shengsuanyun.com/api/v1"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_retries: int = 3
    timeout: float = 30.0
    backup_models: List[str] = field(default_factory=lambda: [
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-v3.2-think", 
        "deepseek/deepseek-v3.1"
    ])


@dataclass
class CacheConfig:
    """缓存配置数据类"""
    memory_max_size: int = 50 * 1024 * 1024  # 50MB
    disk_max_size: int = 500 * 1024 * 1024   # 500MB
    default_ttl: int = 1800  # 30分钟
    cleanup_interval: int = 300  # 5分钟
    cache_dir: str = "data/agent_cache"
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1


@dataclass
class RAGConfig:
    """RAG配置数据类"""
    knowledge_base_path: str = "knowledge_base/platform_knowledge.json"
    model_name: str = "shibing624/text2vec-base-chinese"
    chunk_size: int = 200
    chunk_overlap: int = 50
    top_k: int = 3
    similarity_threshold: float = 0.5
    lazy_load: bool = True
    cache_ttl: int = 3600
    max_cache_size: int = 1000


@dataclass
class MonitoringConfig:
    """监控配置数据类"""
    metrics_enabled: bool = True
    prometheus_port: int = 8000
    log_level: str = "INFO"
    log_file: str = "logs/agent.log"
    health_check_interval: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 5.0,         # 错误率超过5%告警
        "response_time": 5.0,      # 响应时间超过5秒告警
        "memory_usage": 80.0,      # 内存使用超过80%告警
        "cache_hit_rate": 60.0     # 缓存命中率低于60%告警
    })


@dataclass
class AlertConfig:
    """告警配置数据类"""
    feishu_webhook_url: str = ""
    feishu_enabled: bool = False
    email_enabled: bool = False
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = field(default_factory=list)
    apifox_api_url: str = ""
    apifox_token: str = ""
    apifox_enabled: bool = False


@dataclass
class AgentConfig:
    """完整的Agent配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    
    # 系统配置
    debug_mode: bool = False
    simulation_mode: bool = True
    environment: str = "development"  # development, staging, production
    version: str = "2.0.0"
    
    def __post_init__(self):
        """配置验证和后处理"""
        # 从环境变量加载敏感信息
        if not self.model.api_key:
            self.model.api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # 环境变量覆盖
        self._load_from_environment()
        
        # 验证配置
        self._validate_config()
    
    def _load_from_environment(self):
        """从环境变量加载配置"""
        # 模型配置
        if os.getenv("DEEPSEEK_BASE_URL"):
            self.model.base_url = os.getenv("DEEPSEEK_BASE_URL")
        if os.getenv("DEEPSEEK_TEMPERATURE"):
            self.model.temperature = float(os.getenv("DEEPSEEK_TEMPERATURE"))
        
        # 缓存配置
        if os.getenv("CACHE_MEMORY_SIZE"):
            self.cache.memory_max_size = int(os.getenv("CACHE_MEMORY_SIZE"))
        if os.getenv("REDIS_ENABLED"):
            self.cache.redis_enabled = os.getenv("REDIS_ENABLED").lower() == "true"
        if os.getenv("REDIS_HOST"):
            self.cache.redis_host = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            self.cache.redis_port = int(os.getenv("REDIS_PORT"))
        
        # 告警配置
        if os.getenv("FEISHU_WEBHOOK_URL"):
            self.alert.feishu_webhook_url = os.getenv("FEISHU_WEBHOOK_URL")
            self.alert.feishu_enabled = True
        if os.getenv("EMAIL_USERNAME"):
            self.alert.email_username = os.getenv("EMAIL_USERNAME")
            self.alert.email_enabled = True
        if os.getenv("EMAIL_PASSWORD"):
            self.alert.email_password = os.getenv("EMAIL_PASSWORD")
        
        # 系统配置
        if os.getenv("DEBUG_MODE"):
            self.debug_mode = os.getenv("DEBUG_MODE").lower() == "true"
        if os.getenv("SIMULATION_MODE"):
            self.simulation_mode = os.getenv("SIMULATION_MODE").lower() == "true"
        if os.getenv("ENVIRONMENT"):
            self.environment = os.getenv("ENVIRONMENT")
    
    def _validate_config(self):
        """验证配置有效性"""
        errors = []
        
        # 验证API密钥
        if not self.model.api_key:
            errors.append("DEEPSEEK_API_KEY is required")
        elif len(self.model.api_key) < 10:
            errors.append("DEEPSEEK_API_KEY appears invalid")
        
        # 验证缓存配置
        if self.cache.memory_max_size <= 0:
            errors.append("Cache memory size must be positive")
        
        # 验证RAG配置
        if self.rag.top_k <= 0:
            errors.append("RAG top_k must be positive")
        if not (0 <= self.rag.similarity_threshold <= 1):
            errors.append("RAG similarity_threshold must be between 0 and 1")
        
        # 验证监控配置
        if self.monitoring.prometheus_port <= 0 or self.monitoring.prometheus_port > 65535:
            errors.append("Invalid Prometheus port")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")


class ConfigFileHandler(FileSystemEventHandler):
    """配置文件变更监听器"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.last_modified = {}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not any(file_path.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.env']):
            return
        
        # 防止重复触发
        current_time = time.time()
        if file_path in self.last_modified:
            if current_time - self.last_modified[file_path] < 1.0:  # 1秒内忽略
                return
        
        self.last_modified[file_path] = current_time
        
        logging.info(f"配置文件变更检测: {file_path}")
        self.config_manager._reload_config()


class UnifiedConfigManager:
    """统一配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None, watch_files: bool = True):
        self.config_file = Path(config_file) if config_file else Path("config/agent_config.yaml")
        self.watch_files = watch_files
        
        # 配置对象
        self._config: Optional[AgentConfig] = None
        self._config_lock = threading.RLock()
        self._change_callbacks: List[callable] = []
        
        # 文件监控
        self._observer: Optional[Observer] = None
        
        # 加载配置
        self.reload_config()
        
        # 启动文件监控
        if self.watch_files:
            self._start_file_watcher()
    
    @property
    def config(self) -> AgentConfig:
        """获取当前配置"""
        with self._config_lock:
            if self._config is None:
                self.reload_config()
            return self._config
    
    def reload_config(self):
        """重新加载配置"""
        with self._config_lock:
            try:
                # 从文件加载配置
                if self.config_file.exists():
                    config_data = self._load_config_file()
                else:
                    logging.info(f"配置文件不存在 {self.config_file}，使用默认配置")
                    config_data = {}
                
                # 创建配置对象
                old_config = self._config
                self._config = self._create_config_from_dict(config_data)
                
                # 通知配置变更
                if old_config is not None:
                    self._notify_config_changed(old_config, self._config)
                
                logging.info("配置重新加载成功")
                
            except Exception as e:
                logging.error(f"配置重新加载失败: {e}")
                if self._config is None:
                    # 如果没有可用配置，使用默认配置
                    self._config = AgentConfig()
    
    def _reload_config(self):
        """内部重新加载方法（用于文件监控）"""
        try:
            self.reload_config()
        except Exception as e:
            logging.error(f"热重载配置失败: {e}")
    
    def _load_config_file(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.suffix.lower() == '.json':
                    return json.load(f)
                elif self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                else:
                    logging.warning(f"不支持的配置文件格式: {self.config_file.suffix}")
                    return {}
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            return {}
    
    def _create_config_from_dict(self, data: Dict[str, Any]) -> AgentConfig:
        """从字典创建配置对象"""
        try:
            # 创建各部分配置
            model_config = ModelConfig(**data.get('model', {}))
            cache_config = CacheConfig(**data.get('cache', {}))
            rag_config = RAGConfig(**data.get('rag', {}))
            monitoring_config = MonitoringConfig(**data.get('monitoring', {}))
            alert_config = AlertConfig(**data.get('alert', {}))
            
            # 创建主配置
            main_config_data = {k: v for k, v in data.items() 
                               if k not in ['model', 'cache', 'rag', 'monitoring', 'alert']}
            
            config = AgentConfig(
                model=model_config,
                cache=cache_config,
                rag=rag_config,
                monitoring=monitoring_config,
                alert=alert_config,
                **main_config_data
            )
            
            return config
            
        except Exception as e:
            logging.error(f"创建配置对象失败: {e}")
            return AgentConfig()  # 返回默认配置
    
    def _start_file_watcher(self):
        """启动文件监控"""
        try:
            self._observer = Observer()
            handler = ConfigFileHandler(self)
            
            # 监控配置文件目录
            watch_dirs = [
                self.config_file.parent,
                Path("."),  # 监控根目录的.env文件
            ]
            
            for watch_dir in watch_dirs:
                if watch_dir.exists():
                    self._observer.schedule(handler, str(watch_dir), recursive=False)
            
            self._observer.start()
            logging.info("配置文件监控已启动")
            
        except Exception as e:
            logging.error(f"启动配置文件监控失败: {e}")
    
    def _notify_config_changed(self, old_config: AgentConfig, new_config: AgentConfig):
        """通知配置变更"""
        for callback in self._change_callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logging.error(f"配置变更回调执行失败: {e}")
    
    def add_change_callback(self, callback: callable):
        """添加配置变更回调"""
        self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: callable):
        """移除配置变更回调"""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
    
    def save_config(self, config: AgentConfig = None):
        """保存配置到文件"""
        config = config or self._config
        if not config:
            return
        
        try:
            # 确保配置目录存在
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为字典
            config_dict = self._config_to_dict(config)
            
            # 保存文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                elif self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            logging.info(f"配置已保存到: {self.config_file}")
            
        except Exception as e:
            logging.error(f"保存配置失败: {e}")
    
    def _config_to_dict(self, config: AgentConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        from dataclasses import asdict
        return asdict(config)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要信息"""
        config = self.config
        
        return {
            'version': config.version,
            'environment': config.environment,
            'debug_mode': config.debug_mode,
            'simulation_mode': config.simulation_mode,
            'model': {
                'name': config.model.name,
                'base_url': config.model.base_url,
                'api_key_set': bool(config.model.api_key),
                'backup_models_count': len(config.model.backup_models)
            },
            'cache': {
                'memory_size_mb': config.cache.memory_max_size // 1024 // 1024,
                'disk_size_mb': config.cache.disk_max_size // 1024 // 1024,
                'redis_enabled': config.cache.redis_enabled
            },
            'rag': {
                'model_name': config.rag.model_name,
                'top_k': config.rag.top_k,
                'lazy_load': config.rag.lazy_load
            },
            'monitoring': {
                'enabled': config.monitoring.metrics_enabled,
                'log_level': config.monitoring.log_level,
                'prometheus_port': config.monitoring.prometheus_port
            },
            'alerts': {
                'feishu_enabled': config.alert.feishu_enabled,
                'email_enabled': config.alert.email_enabled,
                'apifox_enabled': config.alert.apifox_enabled
            },
            'loaded_at': datetime.now().isoformat(),
            'config_file': str(self.config_file)
        }
    
    def stop(self):
        """停止配置管理器"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logging.info("配置文件监控已停止")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# 全局配置管理器实例
_global_config_manager: Optional[UnifiedConfigManager] = None


def get_config_manager() -> UnifiedConfigManager:
    """获取全局配置管理器"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = UnifiedConfigManager()
    return _global_config_manager


def get_config() -> AgentConfig:
    """获取当前配置"""
    return get_config_manager().config


# 便捷函数
def reload_config():
    """重新加载配置"""
    get_config_manager().reload_config()


def save_config(config: AgentConfig = None):
    """保存配置"""
    get_config_manager().save_config(config)


def add_config_change_callback(callback: callable):
    """添加配置变更回调"""
    get_config_manager().add_change_callback(callback)


# 测试和演示函数
if __name__ == "__main__":
    # 测试配置管理器
    print("=" * 60)
    print("统一配置管理器 - 功能演示")
    print("=" * 60)
    
    # 创建配置管理器
    with UnifiedConfigManager("config/test_config.yaml") as config_manager:
        
        # 获取配置摘要
        summary = config_manager.get_config_summary()
        print("\n配置摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # 测试配置访问
        config = config_manager.config
        print(f"\n当前模型: {config.model.name}")
        print(f"缓存大小: {config.cache.memory_max_size // 1024 // 1024}MB")
        print(f"RAG模型: {config.rag.model_name}")
        
        # 添加配置变更回调
        def on_config_change(old_config, new_config):
            print("🔄 配置已变更！")
        
        config_manager.add_change_callback(on_config_change)
        
        # 保存配置示例
        config_manager.save_config()
        
        print("\n✅ 配置管理器测试完成")

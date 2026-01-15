"""
自定义异常类 - 实现精细化的异常分类和分级处理
为智能客服监控Agent提供专业的错误处理框架
"""

from typing import Optional, Dict, Any


class AgentBaseException(Exception):
    """Agent基础异常类"""
    
    def __init__(self, message: str, error_code: str = "AGENT_ERROR", 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于日志记录和API响应"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "exception_type": self.__class__.__name__
        }


# ============ 模型相关异常 ============
class ModelException(AgentBaseException):
    """模型相关异常基类"""
    
    def __init__(self, message: str, model: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if model:
            details["model"] = model
        super().__init__(message, "MODEL_ERROR", details)


class ModelConnectionError(ModelException):
    """模型连接异常"""
    
    def __init__(self, model: str, endpoint: str, reason: str, **kwargs):
        message = f"模型连接失败: {model} ({endpoint}) - {reason}"
        super().__init__(message, model, 
                         details={"endpoint": endpoint, "reason": reason, **kwargs})


class ModelRateLimitError(ModelException):
    """模型限流异常"""
    
    def __init__(self, model: str, retry_after: Optional[int] = None, **kwargs):
        message = f"模型调用被限流: {model}"
        if retry_after:
            message += f"，建议 {retry_after} 秒后重试"
        details = {"retry_after": retry_after, **kwargs}
        super().__init__(message, model, details=details)


class ModelTimeoutError(ModelException):
    """模型超时异常"""
    
    def __init__(self, model: str, timeout: float, **kwargs):
        message = f"模型调用超时: {model} (超时时间: {timeout}秒)"
        super().__init__(message, model, 
                         details={"timeout": timeout, **kwargs})


class ModelResponseError(ModelException):
    """模型响应异常"""
    
    def __init__(self, model: str, status_code: int, response: str, **kwargs):
        message = f"模型返回错误: {model} (状态码: {status_code})"
        super().__init__(message, model, 
                         details={"status_code": status_code, "response": response, **kwargs})


# ============ RAG相关异常 ============
class RAGException(AgentBaseException):
    """RAG相关异常基类"""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if query:
            details["query"] = query
        super().__init__(message, "RAG_ERROR", details)


class KnowledgeBaseNotFoundError(RAGException):
    """知识库文件不存在异常"""
    
    def __init__(self, knowledge_path: str, **kwargs):
        message = f"知识库文件不存在: {knowledge_path}"
        super().__init__(message, details={"knowledge_path": knowledge_path, **kwargs})


class VectorIndexBuildError(RAGException):
    """向量索引构建异常"""
    
    def __init__(self, model_name: str, error_details: str, **kwargs):
        message = f"向量索引构建失败: {model_name}"
        super().__init__(message, details={"model_name": model_name, "error": error_details, **kwargs})


class SemanticSearchError(RAGException):
    """语义搜索异常"""
    
    def __init__(self, query: str, error_details: str, **kwargs):
        message = f"语义搜索失败: {query}"
        super().__init__(message, query, details={"error": error_details, **kwargs})


# ============ 缓存相关异常 ============
class CacheException(AgentBaseException):
    """缓存相关异常基类"""
    
    def __init__(self, message: str, cache_key: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if cache_key:
            details["cache_key"] = cache_key
        super().__init__(message, "CACHE_ERROR", details)


class CacheConnectionError(CacheException):
    """缓存连接异常"""
    
    def __init__(self, cache_type: str, endpoint: str, error_details: str, **kwargs):
        message = f"{cache_type}缓存连接失败: {endpoint}"
        super().__init__(message, details={
            "cache_type": cache_type,
            "endpoint": endpoint,
            "error": error_details,
            **kwargs
        })


class CacheOperationError(CacheException):
    """缓存操作异常"""
    
    def __init__(self, operation: str, cache_key: str, error_details: str, **kwargs):
        message = f"缓存操作失败: {operation} (key: {cache_key})"
        super().__init__(message, cache_key, details={
            "operation": operation,
            "error": error_details,
            **kwargs
        })


# ============ 工具相关异常 ============
class ToolException(AgentBaseException):
    """工具相关异常基类"""
    
    def __init__(self, message: str, tool_name: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(message, "TOOL_ERROR", details)


class FeishuWebhookError(ToolException):
    """飞书Webhook发送异常"""
    
    def __init__(self, webhook_url: str, status_code: int, response: str, **kwargs):
        message = f"飞书Webhook发送失败 (状态码: {status_code})"
        super().__init__(message, "feishu_tool", details={
            "webhook_url": webhook_url,
            "status_code": status_code,
            "response": response,
            **kwargs
        })


class ApifoxApiError(ToolException):
    """Apifox API调用异常"""
    
    def __init__(self, api_endpoint: str, status_code: int, error_details: str, **kwargs):
        message = f"Apifox API调用失败: {api_endpoint} (状态码: {status_code})"
        super().__init__(message, "apifox_tool", details={
            "api_endpoint": api_endpoint,
            "status_code": status_code,
            "error": error_details,
            **kwargs
        })


class MonitorToolError(ToolException):
    """监控工具异常"""
    
    def __init__(self, operation: str, error_details: str, **kwargs):
        message = f"监控工具操作失败: {operation}"
        super().__init__(message, "monitor_tool", details={
            "operation": operation,
            "error": error_details,
            **kwargs
        })


# ============ 配置相关异常 ============
class ConfigurationException(AgentBaseException):
    """配置相关异常基类"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, "CONFIGURATION_ERROR", details)


class MissingConfigurationError(ConfigurationException):
    """缺少配置异常"""
    
    def __init__(self, config_key: str, **kwargs):
        message = f"缺少必要配置: {config_key}"
        super().__init__(message, config_key, details=kwargs)


class InvalidConfigurationError(ConfigurationException):
    """配置无效异常"""
    
    def __init__(self, config_key: str, config_value: Any, reason: str, **kwargs):
        message = f"配置无效: {config_key}={config_value} - {reason}"
        super().__init__(message, config_key, details={
            "config_value": config_value,
            "reason": reason,
            **kwargs
        })


# ============ 分级异常处理器 ============
class ExceptionHandler:
    """异常处理器 - 提供统一的分级异常处理"""
    
    # 异常级别映射
    SEVERITY_LEVELS = {
        "CRITICAL": 4,    # 系统不可用，需要立即人工介入
        "HIGH": 3,        # 核心功能受影响，需要优先处理
        "MEDIUM": 2,      # 功能部分受限，可以稍后处理
        "LOW": 1,         # 非核心功能问题，可以正常降级
    }
    
    @classmethod
    def get_exception_severity(cls, exception: Exception) -> str:
        """获取异常的严重级别"""
        if isinstance(exception, (
            ModelConnectionError, 
            CacheConnectionError,
            MissingConfigurationError
        )):
            return "CRITICAL"
        
        elif isinstance(exception, (
            ModelRateLimitError,
            ModelTimeoutError,
            ModelResponseError,
            FeishuWebhookError,
            ApifoxApiError
        )):
            return "HIGH"
        
        elif isinstance(exception, (
            VectorIndexBuildError,
            SemanticSearchError,
            CacheOperationError,
            MonitorToolError,
            InvalidConfigurationError
        )):
            return "MEDIUM"
        
        elif isinstance(exception, (
            KnowledgeBaseNotFoundError,
            ToolException
        )):
            return "LOW"
        
        else:
            return "MEDIUM"  # 默认级别
    
    @classmethod
    def should_retry(cls, exception: Exception) -> bool:
        """判断异常是否应该重试"""
        # 这些异常通常可以通过重试解决
        retryable_exceptions = (
            ModelRateLimitError,
            ModelTimeoutError,
            ModelConnectionError,
            CacheConnectionError,
            CacheOperationError
        )
        
        if isinstance(exception, retryable_exceptions):
            return True
        
        # 特别处理网络相关的通用异常
        if "connection" in str(exception).lower() or "timeout" in str(exception).lower():
            return True
        
        return False
    
    @classmethod
    def get_retry_delay(cls, exception: Exception) -> float:
        """获取重试延迟时间（秒）"""
        if isinstance(exception, ModelRateLimitError):
            # 限流异常，使用建议的重试时间或默认5秒
            details = getattr(exception, 'details', {})
            return float(details.get('retry_after', 5))
        
        elif isinstance(exception, ModelTimeoutError):
            # 超时异常，延迟2秒
            return 2.0
        
        elif isinstance(exception, (ModelConnectionError, CacheConnectionError)):
            # 连接异常，延迟3秒
            return 3.0
        
        else:
            # 默认延迟1秒
            return 1.0
    
    @classmethod
    def format_exception_for_logging(cls, exception: Exception) -> Dict[str, Any]:
        """格式化异常信息用于日志记录"""
        if isinstance(exception, AgentBaseException):
            return exception.to_dict()
        
        return {
            "error_code": "UNKNOWN_ERROR",
            "message": str(exception),
            "exception_type": exception.__class__.__name__,
            "details": {
                "traceback": str(exception)
            }
        }
    
    @classmethod
    def handle_exception(cls, exception: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """统一处理异常，返回处理结果"""
        severity = cls.get_exception_severity(exception)
        should_retry = cls.should_retry(exception)
        retry_delay = cls.get_retry_delay(exception) if should_retry else 0
        
        result = {
            "severity": severity,
            "severity_level": cls.SEVERITY_LEVELS[severity],
            "should_retry": should_retry,
            "retry_delay": retry_delay,
            "exception_info": cls.format_exception_for_logging(exception),
            "context": context or {},
            "handled_at": "agent.exceptions.ExceptionHandler"
        }
        
        # 根据严重级别采取不同措施
        if severity == "CRITICAL":
            result["action"] = "system_alert_and_downgrade"
            result["recovery_time"] = "immediate_intervention_required"
        
        elif severity == "HIGH":
            result["action"] = "alert_and_retry_with_backoff"
            result["recovery_time"] = "minutes_to_hours"
        
        elif severity == "MEDIUM":
            result["action"] = "log_and_continue_with_graceful_degradation"
            result["recovery_time"] = "minutes"
        
        else:  # LOW
            result["action"] = "log_only"
            result["recovery_time"] = "seconds_to_minutes"
        
        return result


# 便捷函数
def handle_exception(exception: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """便捷异常处理函数"""
    return ExceptionHandler.handle_exception(exception, context)


def should_retry(exception: Exception) -> bool:
    """便捷重试判断函数"""
    return ExceptionHandler.should_retry(exception)

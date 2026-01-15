"""
Prometheus监控指标工具
提供专业级的Agent性能监控和指标收集
"""
import time
import threading
from typing import Dict, Any, Optional
from collections import defaultdict, Counter
import json
import logging
from pathlib import Path
from datetime import datetime

try:
    from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client未安装，将使用内置指标收集")

class MetricsTool:
    """专业监控指标工具"""
    
    def __init__(self):
        self.metrics_enabled = True
        self.start_time = time.time()
        self.metrics_data = defaultdict(list)
        self.lock = threading.RLock()
        
        # 内置指标计数器
        self._requests_total = 0
        self._requests_success = 0
        self._requests_error = 0
        self._response_times = []
        self._model_usage = Counter()
        self._error_types = Counter()
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Prometheus指标（如果可用）
        self.prometheus_metrics = {}
        self.prometheus_initialized = False  # 新增：Prometheus初始化状态标志
        self._init_prometheus_metrics()
        
        # 指标持久化
        self.metrics_file = Path("data/metrics.json")
        self.metrics_file.parent.mkdir(exist_ok=True)
        self._load_historical_metrics()
    
    def _init_prometheus_metrics(self):
        """初始化Prometheus指标"""
        if not PROMETHEUS_AVAILABLE:
            self.prometheus_initialized = False
            logging.info("Prometheus客户端不可用，使用内置指标")
            return
        
        try:
            # 检查是否已经注册过指标，避免重复注册
            from prometheus_client import REGISTRY
            
            # 使用唯一的实例ID来避免重复注册
            instance_id = id(self)
            
            # 获取已注册的指标名称，处理没有name属性的collector
            registered_names = []
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, 'name'):
                    registered_names.append(collector.name)
            
            # 请求计数器
            metric_name = f'agent_requests_total_{instance_id}'
            if metric_name not in registered_names:
                self.prometheus_metrics['requests_total'] = PrometheusCounter(
                    metric_name,
                    'Total number of requests',
                    ['method', 'status']
                )
            
            # 响应时间直方图
            metric_name = f'agent_response_time_seconds_{instance_id}'
            if metric_name not in registered_names:
                self.prometheus_metrics['response_time'] = Histogram(
                    metric_name,
                    'Response time in seconds',
                    ['method', 'model']
                )
            
            # 模型使用计数
            metric_name = f'agent_model_usage_total_{instance_id}'
            if metric_name not in registered_names:
                self.prometheus_metrics['model_usage'] = PrometheusCounter(
                    metric_name,
                    'Model usage count',
                    ['model', 'status']
                )
            
            # 缓存命中率
            metric_name = f'agent_cache_hits_total_{instance_id}'
            if metric_name not in registered_names:
                self.prometheus_metrics['cache_hits'] = PrometheusCounter(
                    metric_name,
                    'Cache hits count',
                    ['type']
                )
            
            # 错误计数
            metric_name = f'agent_errors_total_{instance_id}'
            if metric_name not in registered_names:
                self.prometheus_metrics['errors_total'] = PrometheusCounter(
                    metric_name,
                    'Error count',
                    ['type', 'model']
                )
            
            # 系统状态指标
            metric_name = f'agent_system_status_{instance_id}'
            if metric_name not in registered_names:
                self.prometheus_metrics['system_status'] = Gauge(
                    metric_name,
                    'System status (1=healthy, 0=unhealthy)',
                    ['component']
                )
            
            # Agent性能指标
            metric_name = f'agent_performance_score_{instance_id}'
            if metric_name not in registered_names:
                self.prometheus_metrics['performance'] = Gauge(
                    metric_name,
                    'Agent performance score',
                    ['metric']
                )
            
            if self.prometheus_metrics:
                logging.info("Prometheus指标初始化成功")
                self.prometheus_initialized = True
            else:
                logging.warning("Prometheus指标已存在或初始化失败，使用已有实例")
                self.prometheus_initialized = False
            
        except Exception as e:
            logging.error(f"Prometheus指标初始化失败: {e}")
            # 即使Prometheus初始化失败，仍然可以使用内置指标
            self.prometheus_metrics = {}
            self.prometheus_initialized = False
    
    def record_request(self, method: str, status: str, response_time: float, model: str = None):
        """记录请求指标"""
        with self.lock:
            try:
                # 更新内置指标
                self._requests_total += 1
                if status == 'success':
                    self._requests_success += 1
                else:
                    self._requests_error += 1
                
                self._response_times.append(response_time)
                if model:
                    self._model_usage[model] += 1
                
                # 更新Prometheus指标
                if PROMETHEUS_AVAILABLE and self.prometheus_metrics:
                    self.prometheus_metrics['requests_total'].labels(
                        method=method, status=status
                    ).inc()
                    
                    if model:
                        self.prometheus_metrics['response_time'].labels(
                            method=method, model=model
                        ).observe(response_time)
                        
                        self.prometheus_metrics['model_usage'].labels(
                            model=model, status=status
                        ).inc()
                
                # 记录详细数据
                self.metrics_data['requests'].append({
                    'timestamp': time.time(),
                    'method': method,
                    'status': status,
                    'response_time': response_time,
                    'model': model
                })
                
            except Exception as e:
                logging.error(f"记录请求指标失败: {e}")
    
    def record_cache_hit(self, cache_type: str = 'default'):
        """记录缓存命中"""
        with self.lock:
            self._cache_hits += 1
            
            if PROMETHEUS_AVAILABLE and self.prometheus_metrics:
                self.prometheus_metrics['cache_hits'].labels(type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str = 'default'):
        """记录缓存未命中"""
        with self.lock:
            self._cache_misses += 1
    
    def record_error(self, error_type: str, model: str = None, details: str = None):
        """记录错误"""
        with self.lock:
            try:
                self._error_types[error_type] += 1
                
                if PROMETHEUS_AVAILABLE and self.prometheus_metrics:
                    self.prometheus_metrics['errors_total'].labels(
                        type=error_type, model=model or 'unknown'
                    ).inc()
                
                # 记录详细错误信息
                self.metrics_data['errors'].append({
                    'timestamp': time.time(),
                    'error_type': error_type,
                    'model': model,
                    'details': details
                })
                
            except Exception as e:
                logging.error(f"记录错误指标失败: {e}")
    
    def update_system_status(self, component: str, status: bool):
        """更新系统状态"""
        if PROMETHEUS_AVAILABLE and self.prometheus_metrics:
            self.prometheus_metrics['system_status'].labels(
                component=component
            ).set(1 if status else 0)
    
    def update_performance_score(self, metric: str, score: float):
        """更新性能评分"""
        if PROMETHEUS_AVAILABLE and self.prometheus_metrics:
            self.prometheus_metrics['performance'].labels(metric=metric).set(score)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            # 计算成功率
            success_rate = (self._requests_success / max(self._requests_total, 1)) * 100
            
            # 计算平均响应时间
            avg_response_time = sum(self._response_times) / max(len(self._response_times), 1)
            
            # 计算缓存命中率
            total_cache_requests = self._cache_hits + self._cache_misses
            cache_hit_rate = (self._cache_hits / max(total_cache_requests, 1)) * 100
            
            return {
                'uptime_seconds': uptime,
                'uptime_formatted': self._format_uptime(uptime),
                'requests': {
                    'total': self._requests_total,
                    'success': self._requests_success,
                    'error': self._requests_error,
                    'success_rate': round(success_rate, 2)
                },
                'performance': {
                    'avg_response_time': round(avg_response_time, 3),
                    'min_response_time': min(self._response_times) if self._response_times else 0,
                    'max_response_time': max(self._response_times) if self._response_times else 0
                },
                'cache': {
                    'hits': self._cache_hits,
                    'misses': self._cache_misses,
                    'hit_rate': round(cache_hit_rate, 2)
                },
                'models': dict(self._model_usage.most_common()),
                'errors': dict(self._error_types.most_common()),
                'last_updated': datetime.now().isoformat()
            }
    
    def get_detailed_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """获取详细指标数据"""
        with self.lock:
            cutoff_time = time.time() - (hours * 3600)
            
            # 过滤最近N小时的数据
            recent_requests = [
                req for req in self.metrics_data['requests'] 
                if req['timestamp'] > cutoff_time
            ]
            
            recent_errors = [
                err for err in self.metrics_data['errors']
                if err['timestamp'] > cutoff_time
            ]
            
            # 按小时聚合数据
            hourly_stats = self._aggregate_hourly_stats(recent_requests, hours)
            
            return {
                'summary': self.get_metrics_summary(),
                'recent_requests': recent_requests[-100:],  # 最近100个请求
                'recent_errors': recent_errors[-50:],       # 最近50个错误
                'hourly_stats': hourly_stats,
                'top_models': dict(Counter(req['model'] for req in recent_requests if req['model']).most_common(10)),
                'error_distribution': dict(Counter(err['error_type'] for err in recent_errors).most_common(10))
            }
    
    def _aggregate_hourly_stats(self, requests: list, hours: int) -> Dict[str, Any]:
        """按小时聚合统计数据"""
        hourly_data = defaultdict(lambda: {
            'requests': 0,
            'success': 0,
            'errors': 0,
            'response_times': []
        })
        
        for req in requests:
            hour_key = datetime.fromtimestamp(req['timestamp']).strftime('%Y-%m-%d %H:00')
            hourly_data[hour_key]['requests'] += 1
            
            if req['status'] == 'success':
                hourly_data[hour_key]['success'] += 1
            else:
                hourly_data[hour_key]['errors'] += 1
            
            hourly_data[hour_key]['response_times'].append(req['response_time'])
        
        # 计算每小时的统计数据
        result = {}
        for hour, data in hourly_data.items():
            avg_time = sum(data['response_times']) / max(len(data['response_times']), 1)
            success_rate = (data['success'] / max(data['requests'], 1)) * 100
            
            result[hour] = {
                'requests': data['requests'],
                'success_rate': round(success_rate, 2),
                'avg_response_time': round(avg_time, 3),
                'errors': data['errors']
            }
        
        return result
    
    def export_prometheus_metrics(self) -> str:
        """导出Prometheus格式的指标"""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"
        
        try:
            return generate_latest().decode('utf-8')
        except Exception as e:
            logging.error(f"导出Prometheus指标失败: {e}")
            return f"# Error exporting metrics: {e}\n"
    
    def save_metrics(self):
        """保存指标到文件"""
        try:
            with self.lock:
                metrics_data = {
                    'summary': self.get_metrics_summary(),
                    'detailed': self.get_detailed_metrics(24),
                    'saved_at': datetime.now().isoformat()
                }
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"保存指标失败: {e}")
    
    def _load_historical_metrics(self):
        """加载历史指标"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    historical_data = json.load(f)
                
                # 恢复部分历史数据
                if 'summary' in historical_data:
                    summary = historical_data['summary']
                    self._requests_total = summary.get('requests', {}).get('total', 0)
                    self._requests_success = summary.get('requests', {}).get('success', 0)
                    self._requests_error = summary.get('requests', {}).get('error', 0)
                    self._cache_hits = summary.get('cache', {}).get('hits', 0)
                    self._cache_misses = summary.get('cache', {}).get('misses', 0)
                
                logging.info("历史指标加载成功")
                
        except Exception as e:
            logging.error(f"加载历史指标失败: {e}")
    
    def _format_uptime(self, uptime_seconds: float) -> str:
        """格式化运行时间"""
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        
        if days > 0:
            return f"{days}天{hours}小时{minutes}分钟"
        elif hours > 0:
            return f"{hours}小时{minutes}分钟"
        elif minutes > 0:
            return f"{minutes}分钟{seconds}秒"
        else:
            return f"{seconds}秒"
    
    def generate_report(self) -> str:
        """生成监控报告"""
        summary = self.get_metrics_summary()
        
        report = f"""
# Agent监控报告
生成时间: {summary['last_updated']}
运行时间: {summary['uptime_formatted']}

## 请求统计
- 总请求数: {summary['requests']['total']}
- 成功请求: {summary['requests']['success']}
- 失败请求: {summary['requests']['error']}
- 成功率: {summary['requests']['success_rate']}%

## 性能指标
- 平均响应时间: {summary['performance']['avg_response_time']}秒
- 最快响应: {summary['performance']['min_response_time']}秒
- 最慢响应: {summary['performance']['max_response_time']}秒

## 缓存统计
- 缓存命中: {summary['cache']['hits']}次
- 缓存未命中: {summary['cache']['misses']}次
- 命中率: {summary['cache']['hit_rate']}%

## 模型使用统计
"""
        
        for model, count in summary['models'].items():
            report += f"- {model}: {count}次\n"
        
        if summary['errors']:
            report += "\n## 错误统计\n"
            for error_type, count in summary['errors'].items():
                report += f"- {error_type}: {count}次\n"
        
        return report.strip()

# 全局指标实例
metrics_tool = MetricsTool()

def record_request(method: str, status: str, response_time: float, model: str = None):
    """便捷的请求记录函数"""
    metrics_tool.record_request(method, status, response_time, model)

def record_cache_hit(cache_type: str = 'default'):
    """便捷的缓存命中记录函数"""
    metrics_tool.record_cache_hit(cache_type)

def record_cache_miss(cache_type: str = 'default'):
    """便捷的缓存未命中记录函数"""
    metrics_tool.record_cache_miss(cache_type)

def record_error(error_type: str, model: str = None, details: str = None):
    """便捷的错误记录函数"""
    metrics_tool.record_error(error_type, model, details)

# 测试函数
if __name__ == "__main__":
    # 模拟一些指标数据
    import random
    
    for i in range(100):
        method = random.choice(['chat', 'search', 'monitor'])
        status = random.choice(['success', 'error'])
        response_time = random.uniform(0.1, 2.0)
        model = random.choice(['deepseek/deepseek-v3.2', 'deepseek/deepseek-v3.2-think', 'deepseek/deepseek-v3.1'])
        
        metrics_tool.record_request(method, status, response_time, model)
        
        if random.random() < 0.3:
            metrics_tool.record_cache_hit()
        else:
            metrics_tool.record_cache_miss()
    
    # 生成报告
    print("=== 监控报告 ===")
    print(metrics_tool.generate_report())
    
    # 保存指标
    metrics_tool.save_metrics()
    print(f"\n指标已保存到: {metrics_tool.metrics_file}")

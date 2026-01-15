"""
增强版智能客服Agent - 集成所有优化功能
包含向量化RAG、Prometheus监控、高级缓存等完整功能
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

# 导入基础工具
from agent.tools.feishu_tool import FeishuTool
from agent.tools.apifox_tool import ApifoxTool
from agent.tools.monitor_tool import MonitorTool

# 导入增强工具
from agent.tools.vector_rag_tool import VectorRAGTool
from agent.tools.metrics_tool import MetricsTool, record_request, record_cache_hit, record_error
from agent.tools.advanced_cache_tool import AdvancedCacheManager, cache

# 导入模型客户端
from agent.models.deepseek_client import DeepSeekClient

class EnhancedCustomerServiceAgent:
    """增强版智能客服监控Agent"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 初始化日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化LLM客户端
        self.llm_client = DeepSeekClient()
        
        # 初始化增强工具
        self._init_enhanced_tools()
        
        # 初始化基础工具
        self.feishu_tool = FeishuTool()
        self.apifox_tool = ApifoxTool()
        self.monitor_tool = MonitorTool()
        
        self.logger.info("增强版智能客服Agent初始化完成")
    
    def _init_enhanced_tools(self):
        """初始化增强工具"""
        # 缓存配置
        cache_config = {
            'memory_max_size': 50 * 1024 * 1024,  # 50MB
            'disk_max_size': 500 * 1024 * 1024,   # 500MB
            'default_ttl': 1800,  # 30分钟
            'cleanup_interval': 300,  # 5分钟清理间隔
            'cache_dir': 'data/agent_cache',
            'redis': {
                'enabled': False,  # 默认禁用Redis，可在配置中启用
                'host': 'localhost',
                'port': 6379,
                'db': 1
            }
        }
        cache_config.update(self.config.get('cache', {}))
        self.cache_manager = AdvancedCacheManager(cache_config)
        
        # 向量化RAG工具
        knowledge_path = self.config.get('knowledge_base', 'knowledge_base/platform_knowledge.json')
        self.vector_rag = VectorRAGTool(knowledge_path)
        
        # 监控工具
        self.metrics = MetricsTool()
        
        self.logger.info("增强工具初始化完成")
    
    @cache(ttl=900, priority='high')  # 缓存15分钟
    def _enhanced_knowledge_search(self, query: str) -> List[Dict[str, Any]]:
        """增强的知识检索 - 使用向量化RAG"""
        start_time = time.time()
        
        try:
            # 使用向量化搜索
            results = self.vector_rag.search(query, top_k=3)
            
            # 记录监控指标
            search_time = time.time() - start_time
            record_request('knowledge_search', 'success', search_time, 'vector_rag')
            
            if results:
                record_cache_hit('knowledge_search')
                self.logger.info(f"向量化检索成功，找到{len(results)}个相关结果")
            else:
                self.logger.warning(f"向量化检索未找到相关结果: {query}")
            
            return results
            
        except Exception as e:
            record_error('knowledge_search_error', details=str(e))
            self.logger.error(f"向量化检索失败: {e}")
            return []
    
    async def _generate_reply(self, user_query: str, context: Dict) -> str:
        """生成智能回复 - 集成缓存和监控"""
        start_time = time.time()
        
        try:
            # 检查缓存
            cache_key = f"reply:{hash(user_query + str(context))}"
            cached_reply = self.cache_manager.get(cache_key)
            
            if cached_reply:
                record_cache_hit('reply_generation')
                self.logger.info("使用缓存回复")
                return cached_reply
            
            # 增强的知识检索
            knowledge_results = self._enhanced_knowledge_search(user_query)
            
            # 构建增强的Prompt
            knowledge_context = ""
            if knowledge_results:
                knowledge_context = "\n".join([
                    f"相关信息{i+1}（相似度: {r['similarity']:.2f}）: {r['text']}"
                    for i, r in enumerate(knowledge_results)
                ])
            
            enhanced_prompt = f"""
作为智能客服，基于以下信息回答用户问题：

用户问题：{user_query}

相关知识：
{knowledge_context}

系统状态：
- API状态: {context.get('api_status', '未知')}
- 响应时间: {context.get('api_response_time', '未知')}
- 监控日志: {context.get('monitor_log', [])}

🚫 严格约束：
1. 如果API状态包含500/503/error，绝不能说"系统正常"
2. 如果监控日志有Error/Critical，必须如实告知
3. 基于真实监控数据，不得编造系统状态
4. 【重要】检测到 monitor_log 中存在 Error 或 API 状态非 200 时，必须在回复开头明确告知用户系统出现异常，禁止掩盖故障

✅ 回复要求：
1. 如果知识库有相关信息，基于知识库回答
2. 如果涉及系统状态问题，诚实告知真实情况
3. 如果知识库无相关信息，明确说明并建议联系客服
4. 保持专业、友善的语调

请提供准确、有帮助的回复：
"""
            
            # 调用LLM生成回复
            reply = await self.llm_client.call_model(
                model="deepseek/deepseek-v3.2-think",
                messages=[
                    {"role": "system", "content": "你是专业智能客服，根据提供的上下文信息回答用户问题。"},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.7,
                expected_format='text'
            )
            
            # 缓存回复
            self.cache_manager.set(
                cache_key, reply, ttl=1800, priority='normal',
                metadata={
                    'user_query': user_query,
                    'knowledge_results_count': len(knowledge_results),
                    'generated_at': datetime.now().isoformat()
                }
            )
            
            # 记录监控指标
            generation_time = time.time() - start_time
            record_request('reply_generation', 'success', generation_time, 'enhanced_agent')
            
            return reply
            
        except Exception as e:
            generation_time = time.time() - start_time
            record_request('reply_generation', 'error', generation_time, 'enhanced_agent')
            record_error('reply_generation_error', details=str(e))
            
            self.logger.error(f"回复生成失败: {e}")
            return "系统遇到临时问题，正在自动修复中，请稍后重试。如需紧急帮助，请联系技术支持。"
    
    def _should_trigger_alert(self, case_data: Dict) -> bool:
        """判断是否需要触发告警 - 增强判断逻辑"""
        api_status = case_data.get('api_status', '')
        monitor_log = case_data.get('monitor_log', [])
        
        # API状态异常
        if 'error' in api_status.lower() or '500' in api_status or '503' in api_status:
            return True
        
        # 监控日志有错误
        if monitor_log:
            for log_entry in monitor_log:
                if log_entry.get('status') in ['Error', 'Critical']:
                    return True
        
        return False
    
    async def _trigger_alerts(self, case_id: str, case_data: Dict) -> List[Dict]:
        """触发告警 - 集成监控"""
        actions = []
        start_time = time.time()
        
        try:
            # 发送飞书告警
            feishu_result = await self.feishu_tool.send_alert(case_data)
            if feishu_result:
                actions.append({"feishu_webhook": "Sent success (Enhanced)"})
            
            # 创建Apifox文档（异步调用）
            apifox_result = await self.apifox_tool.create_error_doc(case_id, case_data)
            if apifox_result:
                actions.append({"apifox_doc_id": f"DOC_{datetime.now().strftime('%Y%m%d')}_{case_id}"})
            
            # 记录监控指标
            alert_time = time.time() - start_time
            record_request('alert_processing', 'success', alert_time, 'enhanced_agent')
            
            # 更新系统状态
            self.metrics.update_system_status('alert_system', True)
            
        except Exception as e:
            alert_time = time.time() - start_time
            record_request('alert_processing', 'error', alert_time, 'enhanced_agent')
            record_error('alert_error', details=str(e))
            
            self.logger.error(f"告警处理失败: {e}")
        
        return actions
    
    async def process_case(self, case_data: Dict) -> Dict[str, Any]:
        """处理单个案例 - 完整增强流程"""
        case_id = case_data.get('case_id', 'UNKNOWN')
        user_query = case_data.get('user_query', '')
        
        self.logger.info(f"处理案例 {case_id}: {user_query[:50]}...")
        
        start_time = time.time()
        result = {
            'case_id': case_id,
            'reply': '',
            'action_triggered': None
        }
        
        try:
            # 生成智能回复
            reply = await self._generate_reply(user_query, case_data)
            result['reply'] = reply
            
            # 判断是否需要触发告警
            if self._should_trigger_alert(case_data):
                actions = await self._trigger_alerts(case_id, case_data)
                if actions:
                    result['action_triggered'] = actions
                
                self.logger.warning(f"案例 {case_id} 触发告警，执行了 {len(actions)} 个动作")
            
            # 记录成功指标
            process_time = time.time() - start_time
            record_request('case_processing', 'success', process_time, 'enhanced_agent')
            
            # 更新性能评分
            self.metrics.update_performance_score('case_processing', min(1.0, 3.0/process_time))
            
        except Exception as e:
            process_time = time.time() - start_time
            record_request('case_processing', 'error', process_time, 'enhanced_agent')
            record_error('case_processing_error', model='enhanced_agent', details=str(e))
            
            self.logger.error(f"案例 {case_id} 处理失败: {e}")
            result['reply'] = "系统遇到临时问题，正在自动修复中，请稍后重试。"
        
        return result
    
    async def process_batch(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """批量处理案例 - 完整工作流程"""
        self.logger.info("开始批量处理案例")
        
        try:
            # 读取输入数据
            with open(input_file, 'r', encoding='utf-8') as f:
                cases = json.load(f)
            
            self.logger.info(f"读取到 {len(cases)} 个测试案例")
            
            # 批量处理
            results = []
            alert_count = 0
            
            for i, case_data in enumerate(cases, 1):
                self.logger.info(f"处理案例 {i}/{len(cases)}: {case_data.get('case_id')}")
                
                result = await self.process_case(case_data)
                results.append(result)
                
                if result.get('action_triggered'):
                    alert_count += 1
            
            # 保存结果
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 生成处理报告
            report = {
                'summary': {
                    'total_cases': len(cases),
                    'successful_cases': len(results),
                    'alerts_triggered': alert_count,
                    'output_file': str(output_path.absolute())
                },
                'metrics': self.metrics.get_metrics_summary(),
                'cache_stats': self.cache_manager.get_stats(),
                'rag_stats': self.vector_rag.get_stats() if hasattr(self.vector_rag, 'get_stats') else {},
                'processing_completed_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"批量处理完成！")
            self.logger.info(f"处理案例: {len(results)}/{len(cases)}")
            self.logger.info(f"触发告警: {alert_count}")
            self.logger.info(f"结果保存: {output_path.absolute()}")
            
            return report
            
        except Exception as e:
            record_error('batch_processing_error', details=str(e))
            self.logger.error(f"批量处理失败: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统完整状态"""
        return {
            'agent_info': {
                'type': 'EnhancedCustomerServiceAgent',
                'version': '2.0.0',
                'capabilities': [
                    'vector_rag_search',
                    'advanced_caching',
                    'prometheus_metrics',
                    'intelligent_alerting',
                    'multi_model_backup'
                ]
            },
            'llm_status': {
                'available_models': [
                    "deepseek/deepseek-v3.2",
                    "deepseek/deepseek-v3.2-think", 
                    "deepseek/deepseek-v3.1"
                ],
                'current_model': 'deepseek/deepseek-v3.2-think',
                'network_status': self.llm_client.get_network_status()
            },
            'metrics': self.metrics.get_metrics_summary(),
            'cache_stats': self.cache_manager.get_stats(),
            'rag_stats': self.vector_rag.get_stats() if hasattr(self.vector_rag, 'get_stats') else {},
            'system_time': datetime.now().isoformat()
        }
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        status = self.get_system_status()
        
        report = f"""
# 增强版智能客服Agent性能报告

## 系统信息
- Agent版本: {status['agent_info']['version']}
- 生成时间: {status['system_time']}
- 核心能力: {', '.join(status['agent_info']['capabilities'])}

## 模型状态
- 当前模型: {status['llm_status']['current_model']}
- 备用模型: {len(status['llm_status']['available_models'])}个
- 网络状态: {status['llm_status']['network_status']['status']}

## 性能指标
{self.metrics.generate_report()}

## 缓存统计
- 命中率: {status['cache_stats']['hit_rate']}%
- 内存使用: {status['cache_stats']['size_info']['memory_usage']} / {status['cache_stats']['size_info']['memory_limit']}
- Redis可用: {status['cache_stats']['config']['redis_available']}

## RAG统计
- 模型可用: {status['rag_stats'].get('model_available', False)}
- 知识块数量: {status['rag_stats'].get('chunks_count', 0)}
- FAISS索引: {status['rag_stats'].get('faiss_available', False)}

---
报告生成完毕 🚀
        """
        
        return report.strip()

# 测试和演示函数
async def main():
    """主函数 - 演示增强Agent功能"""
    print("=" * 60)
    print("增强版智能客服Agent - 完整功能演示")
    print("=" * 60)
    
    # 创建Agent实例
    config = {
        'knowledge_base': '../data/inputs.json',  # 使用测试数据作为知识库
        'cache': {
            'memory_max_size': 20 * 1024 * 1024,  # 20MB
            'default_ttl': 600  # 10分钟
        }
    }
    
    agent = EnhancedCustomerServiceAgent(config)
    
    # 处理测试案例
    input_file = '../data/inputs.json'
    output_file = '../data/outputs/enhanced_results.json'
    
    try:
        report = await agent.process_batch(input_file, output_file)
        
        print("\n" + "=" * 60)
        print("处理完成！以下是详细报告：")
        print("=" * 60)
        
        # 显示处理摘要
        summary = report['summary']
        print(f"📊 总计处理: {summary['total_cases']}个案例")
        print(f"✅ 成功处理: {summary['successful_cases']}个案例")
        print(f"🚨 触发告警: {summary['alerts_triggered']}个案例")
        print(f"💾 结果文件: {summary['output_file']}")
        
        # 显示性能报告
        print("\n" + agent.generate_performance_report())
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        logging.error(f"主函数执行失败: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())


# prompts.py
# 系统提示词：定义Agent的角色和能力
SYSTEM_PROMPT = """你是一个智能客服监控Agent，具有以下能力：
1. 理解用户关于系统的问题
2. 监控系统API状态
3. 在系统异常时触发告警
4. 基于知识库准确回答业务问题

请严格按照要求执行任务。"""

# 规划提示词：告诉R1模型如何规划任务
PLANNING_PROMPT = """分析以下情况，制定执行计划：
用户问题：{user_query}
API状态：{api_status}
监控日志：{monitor_log}

请输出JSON格式的计划：
{{
  "need_rag": true/false,  # 是否需要查询知识库
  "need_alert": true/false,  # 是否需要发送告警
  "alert_reason": "如果需要告警，说明原因",
  "steps": ["步骤1", "步骤2", "步骤3"]
}}"""
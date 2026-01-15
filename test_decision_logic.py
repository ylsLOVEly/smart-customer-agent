#!/usr/bin/env python3
"""
测试优化后的决策逻辑
验证"先查知识库，再判断状态查询"的策略
"""

import asyncio
import sys
sys.path.append('.')

from agent.agents.customer_agent import CustomerServiceAgent

async def test_decision_logic():
    """测试新的决策逻辑"""
    print("=" * 60)
    print("测试优化后的决策逻辑")
    print("=" * 60)
    
    agent = CustomerServiceAgent()
    
    # 测试案例列表
    test_cases = [
        {
            "name": "业务问题（知识库中有）",
            "query": "这个平台支持什么模型？",
            "api_status": "200 OK",
            "monitor_log": []
        },
        {
            "name": "业务问题（知识库中有）", 
            "query": "计费模式是怎样的？",
            "api_status": "200 OK",
            "monitor_log": []
        },
        {
            "name": "系统状态查询",
            "query": "今天系统稳定吗？",
            "api_status": "200 OK", 
            "monitor_log": [{"timestamp": "10:00:00", "status": "Error", "msg": "Timeout"}]
        },
        {
            "name": "混合问题（既有业务又有状态）",
            "query": "系统刚才出问题了，计费还正常吗？",
            "api_status": "200 OK",
            "monitor_log": [{"timestamp": "10:00:00", "status": "Error", "msg": "Timeout"}]
        },
        {
            "name": "知识库中没有的业务问题",
            "query": "你们有移动端APP吗？",
            "api_status": "200 OK",
            "monitor_log": []
        },
        {
            "name": "模糊的系统状态查询",
            "query": "刚才是不是有什么问题？",
            "api_status": "500 Error",
            "monitor_log": [{"timestamp": "10:00:00", "status": "Error", "msg": "Service Unavailable"}]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. 测试: {test_case['name']}")
        print(f"   问题: {test_case['query']}")
        print(f"   API状态: {test_case['api_status']}")
        
        # 创建案例数据
        case_data = {
            "case_id": f"TEST{i}",
            "user_query": test_case["query"],
            "api_status": test_case["api_status"],
            "api_response_time": "100ms",
            "monitor_log": test_case["monitor_log"]
        }
        
        # 测试决策逻辑
        monitor_result = agent.monitor_tool.check_status(
            case_data.get("api_status", "200 OK"),
            case_data.get("monitor_log", [])
        )
        
        plan = await agent._make_plan(case_data, monitor_result)
        
        print(f"   知识库是否有信息: {plan.get('has_knowledge', False)}")
        print(f"   是否为状态查询: {plan.get('is_system_status', False)}")
        print(f"   是否需要RAG: {plan.get('need_rag', False)}")
        print(f"   是否需要告警: {plan.get('need_alert', False)}")
        
        # 执行完整处理
        result = await agent.process_case(case_data)
        print(f"   最终回复长度: {len(result.get('reply', ''))} 字符")
        print(f"   触发动作: {result.get('action_triggered', '无')}")
        
        # 检查回复内容
        reply = result.get('reply', '')
        if "根据平台信息" in reply or "胜算云平台" in reply:
            print("   ✅ 使用了知识库信息")
        elif "检测到系统" in reply or "监控数据" in reply:
            print("   ✅ 使用了系统状态信息")
        else:
            print("   🔍 使用了其他回复策略")
    
    print(f"\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print("优化后的决策逻辑验证完成！")
    print("新策略：先查知识库 → 再判断状态查询")
    print("优势：避免漏检知识库中的业务问题")

if __name__ == "__main__":
    asyncio.run(test_decision_logic())

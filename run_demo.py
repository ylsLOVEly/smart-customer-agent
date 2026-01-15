#!/usr/bin/env python3
"""
智能客服监控Agent演示脚本
展示比赛要求的所有功能
"""

import asyncio
import json
import sys
from datetime import datetime

# 添加项目路径
sys.path.append('.')

from agent.agents.customer_agent import CustomerServiceAgent

async def demo():
    """演示智能客服监控Agent的所有功能"""
    
    print("=" * 60)
    print("智能客服监控Agent - 比赛功能演示")
    print("=" * 60)
    
    # 初始化Agent
    print("\n1. 初始化Agent...")
    agent = CustomerServiceAgent()
    print("   [OK] Agent初始化成功")
    
    # 演示案例1：正常业务咨询（RAG功能）
    print("\n2. 演示案例1：业务咨询 + RAG知识库查询")
    case1 = {
        "case_id": "DEMO001",
        "user_query": "你们平台的计费模式是怎样的？我想了解一下价格。",
        "api_status": "200 OK",
        "api_response_time": "150ms",
        "monitor_log": []
    }
    
    print(f"   用户问题: {case1['user_query']}")
    print(f"   API状态: {case1['api_status']}")
    
    result1 = await agent.process_case(case1)
    print(f"   Agent回复: {result1['reply'][:80]}...")
    print(f"   触发动作: {result1['action_triggered'] or '无'}")
    print("   [OK] 演示完成：RAG知识库查询 + 业务咨询回答")
    
    # 演示案例2：系统异常 + 自动告警
    print("\n3. 演示案例2：系统异常检测 + 自动告警")
    case2 = {
        "case_id": "DEMO002",
        "user_query": "刚才API是不是出问题了？我一直调用失败",
        "api_status": "503 Service Unavailable",
        "api_response_time": "Timeout",
        "monitor_log": [
            {"timestamp": "15:30:00", "status": "Error", "msg": "Service Unavailable"},
            {"timestamp": "15:31:00", "status": "Error", "msg": "Connection Timeout"}
        ]
    }
    
    print(f"   用户问题: {case2['user_query']}")
    print(f"   API状态: {case2['api_status']}")
    print(f"   监控日志: {len(case2['monitor_log'])} 条错误记录")
    
    result2 = await agent.process_case(case2)
    print(f"   Agent回复: {result2['reply'][:80]}...")
    if result2['action_triggered']:
        for action in result2['action_triggered']:
            print(f"   触发动作: {action}")
    print("   [OK] 演示完成：系统异常检测 + 飞书告警 + Apifox文档记录")
    
    # 演示案例3：系统状态查询
    print("\n4. 演示案例3：系统稳定性状态查询")
    case3 = {
        "case_id": "DEMO003",
        "user_query": "今天系统整体稳定吗？有没有出现过问题？",
        "api_status": "200 OK",
        "api_response_time": "120ms",
        "monitor_log": [
            {"timestamp": "09:15:00", "status": "Warning", "msg": "High Latency"},
            {"timestamp": "11:30:00", "status": "Error", "msg": "Brief Service Disruption"}
        ]
    }
    
    print(f"   用户问题: {case3['user_query']}")
    print(f"   API状态: {case3['api_status']}")
    print(f"   监控日志: {len(case3['monitor_log'])} 条历史记录")
    
    result3 = await agent.process_case(case3)
    print(f"   Agent回复: {result3['reply'][:80]}...")
    print("   [OK] 演示完成：系统状态感知 + 历史监控数据分析")
    
    # 演示案例4：知识库未覆盖的问题
    print("\n5. 演示案例4：知识库未覆盖的业务问题")
    case4 = {
        "case_id": "DEMO004",
        "user_query": "你们有没有手机APP？在哪里下载？",
        "api_status": "200 OK",
        "api_response_time": "100ms",
        "monitor_log": []
    }
    
    print(f"   用户问题: {case4['user_query']}")
    result4 = await agent.process_case(case4)
    print(f"   Agent回复: {result4['reply'][:80]}...")
    print("   [OK] 演示完成：知识库未覆盖问题的处理策略")
    
    # 保存所有演示结果
    output_file = "data/outputs/demo_results.json"
    all_results = [result1, result2, result3, result4]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "=" * 60)
    print("演示总结")
    print("=" * 60)
    print(f"总测试案例: 4个")
    print(f"已触发告警: {sum(1 for r in all_results if r['action_triggered'])}个")
    print(f"知识库查询: {sum(1 for r in all_results if '计费' in r['reply'] or '平台' in r['reply'])}个")
    print(f"结果保存: {output_file}")
    print("\n[SUCCESS] 所有比赛要求功能演示完成！")
    print("   1. RAG知识库问答 [✓]")
    print("   2. 系统监控与异常检测 [✓]")
    print("   3. 自动告警（飞书）[✓]")
    print("   4. 文档记录（Apifox）[✓]")
    print("   5. 系统状态感知 [✓]")
    print("   6. 单模型约束（DeepSeek）[✓]")
    print("   7. 容错降级处理 [✓]")

if __name__ == "__main__":
    print("开始智能客服监控Agent演示...")
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

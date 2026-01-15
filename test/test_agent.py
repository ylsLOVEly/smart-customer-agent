import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agents.customer_agent import CustomerServiceAgent

async def test_agent():
    """测试智能客服Agent"""
    print("=== 智能客服监控Agent测试 ===")
    
    # 1. 初始化Agent
    print("1. 初始化Agent...")
    agent = CustomerServiceAgent()
    
    # 2. 创建测试案例
    test_cases = [
        {
            "case_id": "TEST001",
            "user_query": "你们平台的计费模式是怎样的？",
            "api_status": "200 OK",
            "api_response_time": "120ms",
            "monitor_log": []
        },
        {
            "case_id": "TEST002",
            "user_query": "系统刚才是不是挂了？",
            "api_status": "500 Internal Server Error",
            "api_response_time": "Timeout",
            "monitor_log": [
                {"timestamp": "14:30:00", "status": "Error", "msg": "Service Unavailable"}
            ]
        },
        {
            "case_id": "TEST003",
            "user_query": "今天系统稳定吗？",
            "api_status": "200 OK",
            "api_response_time": "150ms",
            "monitor_log": [
                {"timestamp": "10:15:00", "status": "Error", "msg": "Connection Timeout"},
                {"timestamp": "11:30:00", "status": "Warning", "msg": "High Latency"}
            ]
        }
    ]
    
    # 3. 测试每个案例
    for case in test_cases:
        print(f"\n2. 测试案例 {case['case_id']}:")
        print(f"   问题: {case['user_query']}")
        print(f"   API状态: {case['api_status']}")
        
        try:
            result = await agent.process_case(case)
            
            print(f"   回复: {result['reply'][:100]}...")
            if result['action_triggered']:
                print(f"   触发动作: {result['action_triggered']}")
            else:
                print(f"   触发动作: 无")
                
        except Exception as e:
            print(f"   处理失败: {e}")
    
    # 4. 测试集成运行
    print("\n3. 测试完整流程...")
    try:
        # 创建输入文件
        test_input = {
            "case_id": "FULL_TEST",
            "user_query": "如何注册账号并查看计费详情？",
            "api_status": "200 OK",
            "api_response_time": "80ms",
            "monitor_log": []
        }
        
        result = await agent.process_case(test_input)
        print(f"   完整测试结果:")
        print(f"   回复长度: {len(result['reply'])} 字符")
        print(f"   是否查询知识库: {'是' if '计费' in result['reply'] or '注册' in result['reply'] else '否'}")
        print("✓ 测试完成")
        
    except Exception as e:
        print(f"✗ 完整测试失败: {e}")

async def main():
    await test_agent()

if __name__ == "__main__":
    asyncio.run(main())

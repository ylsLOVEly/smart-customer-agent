import asyncio
import json
from agent.agents.customer_agent import CustomerServiceAgent

async def main():
    print("测试智能客服Agent...")
    agent = CustomerServiceAgent()
    
    # 测试案例1：正常业务咨询
    case1 = {
        'case_id': 'TEST1',
        'user_query': '计费模式是什么？',
        'api_status': '200 OK',
        'api_response_time': '100ms',
        'monitor_log': []
    }
    
    print(f"\n测试案例1: {case1['user_query']}")
    result1 = await agent.process_case(case1)
    print(f"回复: {result1['reply'][:100]}...")
    print(f"动作: {result1['action_triggered']}")
    
    # 测试案例2：系统异常咨询
    case2 = {
        'case_id': 'TEST2',
        'user_query': '系统刚才是不是挂了？',
        'api_status': '500 Internal Server Error',
        'api_response_time': 'Timeout',
        'monitor_log': [
            {'timestamp': '14:30:00', 'status': 'Error', 'msg': 'Service Unavailable'}
        ]
    }
    
    print(f"\n测试案例2: {case2['user_query']}")
    result2 = await agent.process_case(case2)
    print(f"回复: {result2['reply'][:100]}...")
    print(f"动作: {result2['action_triggered']}")
    
    # 保存结果
    with open('data/outputs/test_results.json', 'w', encoding='utf-8') as f:
        json.dump([result1, result2], f, ensure_ascii=False, indent=2)
    print("\n测试结果已保存到 data/outputs/test_results.json")

if __name__ == '__main__':
    asyncio.run(main())

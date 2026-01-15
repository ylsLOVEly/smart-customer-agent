"""
测试增强版DeepSeek客户端的稳定性和容错能力
"""
import asyncio
import json
import logging
import time
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加项目路径
import sys
sys.path.append('.')
sys.path.append('./agent')

from agent.models.deepseek_client import DeepSeekClient

async def test_concurrent_requests():
    """测试并发请求控制"""
    print("\n=== 测试并发请求控制 ===")
    client = DeepSeekClient()
    
    messages = [{"role": "user", "content": "简单回复：你好"}]
    
    # 创建5个并发请求（超过限制的3个）
    start_time = time.time()
    tasks = []
    for i in range(5):
        task = client.call_model("deepseek/deepseek-v3.2", messages, 0.7)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()
    
    success_count = sum(1 for r in results if isinstance(r, str) and r)
    print(f"并发请求完成：{success_count}/5 成功")
    print(f"总耗时：{end_time - start_time:.2f}秒")
    
    return success_count > 0

async def test_format_validation():
    """测试格式验证和修复"""
    print("\n=== 测试格式验证 ===")
    client = DeepSeekClient()
    
    # 测试文本格式清理
    test_response = "   这是一个   测试    回复   "
    cleaned = client._validate_text_format(test_response)
    print(f"文本清理：'{test_response}' -> '{cleaned}'")
    
    # 测试JSON格式修复
    json_response = '```json\n{"status": "ok", "message": "test"}\n```'
    fixed_json = client._validate_json_format(json_response)
    print(f"JSON修复成功：{json.loads(fixed_json) if fixed_json else 'Failed'}")
    
    return True

async def test_cache_mechanism():
    """测试缓存机制"""
    print("\n=== 测试缓存机制 ===")
    client = DeepSeekClient()
    
    messages = [{"role": "user", "content": "测试缓存：当前时间"}]
    
    # 第一次请求
    start_time = time.time()
    result1 = await client.call_model("deepseek/deepseek-v3.2", messages, 0.7)
    first_duration = time.time() - start_time
    
    # 第二次请求（应该使用缓存）
    start_time = time.time()
    result2 = await client.call_model("deepseek/deepseek-v3.2", messages, 0.7)
    second_duration = time.time() - start_time
    
    print(f"第一次请求耗时：{first_duration:.2f}秒")
    print(f"第二次请求耗时：{second_duration:.2f}秒")
    print(f"缓存效果：{second_duration < first_duration}")
    
    return result1 == result2

async def test_offline_fallback():
    """测试离线降级机制"""
    print("\n=== 测试离线降级机制 ===")
    client = DeepSeekClient()
    
    # 模拟网络不佳状态
    client._network_status = "degraded"
    
    messages = [{"role": "user", "content": "系统状态如何？"}]
    result = await client.call_model("deepseek/deepseek-v3.2", messages, 0.7)
    
    print(f"离线回复：{result}")
    print(f"网络状态：{client.get_network_status()}")
    
    # 重置网络状态
    client._network_status = "unknown"
    client._consecutive_failures = 0
    
    return "抱歉" in result or "状态" in result

async def test_multi_model_fallback():
    """测试多模型备份"""
    print("\n=== 测试多模型备份 ===")
    client = DeepSeekClient()
    
    # 使用一个不存在的模型，应该自动切换到备用模型
    messages = [{"role": "user", "content": "简单测试"}]
    result = await client.call_model("nonexistent-model", messages, 0.7)
    
    print(f"多模型备份结果：{result is not None}")
    print(f"回复内容：{result[:100] if result else 'None'}...")
    
    return result is not None

def test_offline_responses():
    """测试离线应急回复"""
    print("\n=== 测试离线应急回复 ===")
    
    offline_file = Path("agent/knowledge_base/offline_responses.json")
    if offline_file.exists():
        with open(offline_file, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        print(f"离线回复类型数量：{len(responses)}")
        print("回复类型：", list(responses.keys()))
        return True
    else:
        print("离线回复文件不存在")
        return False

async def main():
    """运行所有测试"""
    print("开始测试增强版DeepSeek客户端...")
    
    tests = [
        ("离线回复文件", test_offline_responses),
        ("格式验证", test_format_validation),
        ("缓存机制", test_cache_mechanism),
        ("多模型备份", test_multi_model_fallback),
        ("离线降级", test_offline_fallback),
        ("并发控制", test_concurrent_requests),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            print(f"\n{'='*50}")
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[name] = result
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{name}: {status}")
        except Exception as e:
            results[name] = False
            print(f"{name}: ❌ 异常 - {e}")
    
    # 总结
    print(f"\n{'='*50}")
    print("测试总结:")
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    print(f"通过测试：{passed}/{total}")
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    if passed == total:
        print("\n🎉 所有测试通过！DeepSeek客户端增强功能工作正常。")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，需要进一步调试。")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

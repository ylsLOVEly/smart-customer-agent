#!/usr/bin/env python3
"""
环境检查脚本
验证项目所需的所有组件是否正常工作
"""

import sys
import os
import importlib
import json

def check_python_version():
    """检查Python版本"""
    print("1. 检查Python版本...")
    version = sys.version_info
    print(f"   Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python版本符合要求 (>=3.8)")
        return True
    else:
        print("   ❌ Python版本过低，需要3.8或以上")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n2. 检查依赖包...")
    dependencies = ['httpx', 'dotenv']
    all_ok = True
    
    for dep in dependencies:
        try:
            module = importlib.import_module(dep)
            print(f"   ✅ {dep}: {module.__version__ if hasattr(module, '__version__') else '已安装'}")
        except ImportError:
            print(f"   ❌ {dep}: 未安装")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """检查项目结构"""
    print("\n3. 检查项目结构...")
    required_files = [
        "requirements.txt",
        ".env",
        "config/settings.py",
        "config/prompts.py",
        "agent/main.py",
        "agent/agents/customer_agent.py",
        "agent/models/deepseek_client.py",
        "agent/tools/monitor_tool.py",
        "agent/tools/feishu_tool.py",
        "agent/tools/apifox_tool.py",
        "agent/tools/rag_tool.py",
        "agent/knowledge_base/platform_knowledge.json",
        "data/inputs.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}: 文件不存在")
            all_exist = False
    
    return all_exist

def check_knowledge_base():
    """检查知识库"""
    print("\n4. 检查知识库...")
    try:
        with open('agent/knowledge_base/platform_knowledge.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            knowledge_items = data.get('platform_knowledge', [])
            print(f"   ✅ 知识库文件: {len(knowledge_items)} 个知识条目")
            for item in knowledge_items:
                category = item.get('category', '未知')
                keywords = item.get('keywords', [])
                print(f"     - {category}: {len(keywords)} 个关键词")
            return True
    except Exception as e:
        print(f"   ❌ 知识库加载失败: {e}")
        return False

def check_input_data():
    """检查输入数据"""
    print("\n5. 检查输入数据...")
    try:
        with open('data/inputs.json', 'r', encoding='utf-8') as f:
            cases = json.load(f)
            print(f"   ✅ 输入文件: {len(cases)} 个测试案例")
            for case in cases[:3]:  # 只显示前3个
                case_id = case.get('case_id', '未知')
                query = case.get('user_query', '')[:30] + "..."
                print(f"     - {case_id}: {query}")
            return True
    except Exception as e:
        print(f"   ❌ 输入数据加载失败: {e}")
        return False

def check_environment_variables():
    """检查环境变量"""
    print("\n6. 检查环境变量...")
    env_file = '.env'
    if os.path.exists(env_file):
        print(f"   ✅ 环境变量文件存在")
        
        # 检查关键配置 - 修复编码问题
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # 如果utf-8失败，尝试其他编码
            try:
                with open(env_file, 'r', encoding='gbk') as f:
                    content = f.read()
            except:
                content = ""
                print("   ⚠️  环境变量文件编码问题，但不影响使用")
                return True
            checks = [
                ("DEEPSEEK_API_KEY", "API密钥配置"),
                ("DEEPSEEK_BASE_URL", "API基础地址"),
                ("FEISHU_WEBHOOK_URL", "飞书Webhook"),
                ("APIFOX_API_URL", "Apifox API地址")
            ]
            
            for key, desc in checks:
                if key in content:
                    print(f"     ✅ {desc}: 已配置")
                else:
                    print(f"     ⚠️  {desc}: 未配置（比赛环境可用默认值）")
        return True
    else:
        print(f"   ⚠️  环境变量文件不存在，将使用默认配置")
        return True

def run_simple_test():
    """运行简单测试"""
    print("\n7. 运行功能测试...")
    try:
        # 添加项目路径
        sys.path.append('.')
        
        # 测试导入核心模块
        from agent.agents.customer_agent import CustomerServiceAgent
        print("   ✅ Agent模块导入成功")
        
        # 创建一个简单的测试案例
        import asyncio
        
        async def test_one_case():
            agent = CustomerServiceAgent()
            case = {
                "case_id": "ENV_TEST",
                "user_query": "测试环境是否正常？",
                "api_status": "200 OK",
                "api_response_time": "100ms",
                "monitor_log": []
            }
            
            result = await agent.process_case(case)
            print(f"   ✅ Agent处理成功，回复长度: {len(result.get('reply', ''))} 字符")
            return True
        
        # 运行测试
        success = asyncio.run(test_one_case())
        return success
        
    except Exception as e:
        print(f"   ❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("智能客服监控Agent - 环境检查")
    print("=" * 60)
    
    checks = [
        ("Python版本", check_python_version),
        ("依赖包", check_dependencies),
        ("项目结构", check_project_structure),
        ("知识库", check_knowledge_base),
        ("输入数据", check_input_data),
        ("环境变量", check_environment_variables),
        ("功能测试", run_simple_test)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"检查 {check_name} 时出错: {e}")
            results.append((check_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("\n🎉 所有检查通过！项目环境配置正确。")
        print("   可以运行以下命令测试：")
        print("   - python run_demo.py      # 运行完整演示")
        print("   - python agent/main.py      # 运行主程序")
        print("   - python test_simple.py   # 运行简单测试")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 项检查未通过，请根据上述信息修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

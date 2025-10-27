#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 LLM 配置和连接
"""

import sys
import json
from pathlib import Path

def test_config():
    """测试配置文件"""
    config_path = Path("Data Acquisition/Extract Text/config.yaml")
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 配置文件加载成功 ({config_path.name})")
        print(f"   Provider: {config.get('llm_provider', 'unknown')}")
        return True
    except ImportError:
        print("❌ 缺少 PyYAML 库，请运行: pip install pyyaml")
        return False
    except Exception as e:
        print(f"❌ 配置文件解析失败: {e}")
        return False

def test_ollama():
    """测试 Ollama 连接"""
    try:
        import requests
        print("\n测试 Ollama 连接...")
        
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama 连接成功，可用模型数: {len(models)}")
            if models:
                print("   可用模型:")
                for model in models[:5]:  # 只显示前5个
                    print(f"   - {model.get('name', 'unknown')}")
            return True
        else:
            print(f"❌ Ollama 响应异常: {response.status_code}")
            return False
    except ImportError:
        print("❌ 缺少 requests 库，请运行: pip install requests")
        return False
    except Exception as e:
        print(f"❌ Ollama 连接失败: {e}")
        print("   提示：确保 Ollama 服务正在运行 (ollama serve)")
        return False

def test_openai():
    """测试 OpenAI 配置"""
    import os
    print("\n测试 OpenAI 配置...")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"✅ 环境变量 OPENAI_API_KEY 已设置 (长度: {len(api_key)})")
        return True
    else:
        print("⚠️  环境变量 OPENAI_API_KEY 未设置")
        print("   如需使用 OpenAI，请设置: export OPENAI_API_KEY=sk-xxx...")
        return False

def test_llm_simple():
    """简单 LLM 测试"""
    print("\n执行简单 LLM 测试...")
    
    config_path = Path("Data Acquisition/Extract Text/config.yaml")
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        provider = config.get('llm_provider')
        
        if provider == 'ollama':
            print("   使用 Ollama 进行测试...")
            import requests
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": config['ollama']['model'],
                "prompt": "Hello, respond with just 'OK'",
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            output = result.get("response", "").strip()
            print(f"✅ Ollama 测试成功，响应: {output[:50]}")
            return True
            
        elif provider == 'openai':
            print("   使用 OpenAI 进行测试...")
            import requests
            import os
            
            api_key_spec = config['openai']['api_key']
            if api_key_spec.startswith("ENV:"):
                api_key = os.environ.get(api_key_spec[4:])
            else:
                api_key = api_key_spec
            
            if not api_key:
                print("❌ API Key 未配置")
                return False
            
            url = f"{config['openai']['base_url']}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": config['openai']['model'],
                "messages": [{"role": "user", "content": "Hello, respond with just 'OK'"}],
                "max_tokens": 10
            }
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            output = result['choices'][0]['message']['content']
            print(f"✅ OpenAI 测试成功，响应: {output[:50]}")
            return True
        else:
            print(f"❌ 未知的 provider: {provider}")
            return False
            
    except Exception as e:
        print(f"❌ LLM 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("LLM 配置测试")
    print("=" * 60)
    
    results = []
    
    # 测试配置文件
    results.append(("配置文件", test_config()))
    
    # 测试 Ollama
    results.append(("Ollama 连接", test_ollama()))
    
    # 测试 OpenAI
    results.append(("OpenAI 配置", test_openai()))
    
    # 简单 LLM 测试
    results.append(("LLM 功能", test_llm_simple()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:15s} : {status}")
    
    print("\n建议:")
    provider = None
    try:
        import yaml
        config_path = Path("Data Acquisition/Extract Text/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                provider = yaml.safe_load(f).get('llm_provider')
    except:
        pass
    
    if provider == 'ollama':
        if not results[1][1]:  # Ollama 连接失败
            print("- 启动 Ollama 服务: ollama serve")
            print("- 拉取模型: ollama pull qwen3:8b")
    elif provider == 'openai':
        if not results[2][1]:  # OpenAI 配置失败
            print("- 设置 API Key: export OPENAI_API_KEY=sk-xxx...")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 所有测试通过！可以运行 extract_text_llm.py")
    else:
        print("\n⚠️  部分测试失败，请根据提示修复问题")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

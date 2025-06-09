import json
import time
from typing import List, Dict, Tuple
import requests
from datetime import datetime

# 配置
DEEPSEEK_API_KEY = ""  # 替换为你的API密钥
DEEPSEEK_CHAT_API = "https://api.deepseek.com/v1/chat/completions"
QA_FILE = "diabetes_qa_pairs.json"

class DiabetesPromptTester:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def load_qa_data(self) -> List[Dict]:
        """加载QA数据"""
        with open(QA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_prompts(self, question: str) -> Dict[str, str]:
        """生成不同风格的提示词"""
        return {
            "basic": question,
            "expert": f"""你是一名糖尿病专科医生，请用专业医学知识回答以下问题。要求：
1. 使用标准医学术语（如"多饮、多尿、多食"）
2. 包含诊断标准数值（如HbA1c≥6.5%）
3. 必要时引用权威指南

问题：{question}""",
            "patient": f"""请用通俗易懂的语言向糖尿病患者解释：
问题：{question}
要求：
1. 避免复杂医学术语
2. 给出实用建议
3. 保持温暖友好的语气"""
        }

    def query_model(self, prompt: str, model: str = "deepseek-chat") -> str:
        """查询模型"""
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(DEEPSEEK_CHAT_API, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"API调用失败: {str(e)}"

    def run_comparison(self, num_samples: int = 5) -> List[Dict]:
        """运行对比测试"""
        qa_data = self.load_qa_data()
        results = []
        
        for qa in qa_data[:num_samples]:
            prompts = self.generate_prompts(qa["question"])
            row = {"question": qa["question"], "expected": qa["answer"]}
            
            for name, prompt in prompts.items():
                row[f"{name}_prompt"] = prompt
                row[f"{name}_response"] = self.query_model(prompt)
                time.sleep(1)  # 避免速率限制
            
            results.append(row)
            print(f"已完成问题: {qa['question'][:50]}...")
        
        return results

    def save_results(self, results: List[Dict]):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prompt_comparison_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "test_time": timestamp,
                    "num_questions": len(results),
                    "model": "deepseek-chat"
                },
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到 {filename}")

        # 生成易读的报告
        report_name = f"prompt_report_{timestamp}.txt"
        with open(report_name, 'w', encoding='utf-8') as f:
            f.write("糖尿病问答提示词对比实验报告\n")
            f.write("="*80 + "\n\n")
            
            for i, res in enumerate(results, 1):
                f.write(f"{i}. 问题: {res['question']}\n")
                f.write(f"【标准答案】\n{res['expected']}\n\n")
                
                f.write("【基础提示词】\n")
                f.write(f"Prompt: {res['basic_prompt']}\n")
                f.write(f"Response: {res['basic_response']}\n\n")
                
                f.write("【专家模式提示词】\n")
                f.write(f"Prompt: {res['expert_prompt']}\n")
                f.write(f"Response: {res['expert_response']}\n\n")
                
                f.write("【患者友好模式】\n")
                f.write(f"Prompt: {res['patient_prompt']}\n")
                f.write(f"Response: {res['patient_response']}\n")
                f.write("="*80 + "\n\n")
        
        print(f"报告已生成: {report_name}")

if __name__ == "__main__":
    print("糖尿病问答提示词优化实验")
    print("="*50)
    
    tester = DiabetesPromptTester(DEEPSEEK_API_KEY)
    
    try:
        print("\n正在运行对比实验...")
        results = tester.run_comparison(num_samples=5)  # 测试5个问题
        tester.save_results(results)
        
        print("\n实验完成！生成文件：")
        print("- prompt_comparison_*.json (完整数据)")
        print("- prompt_report_*.txt (易读报告)")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    finally:
        print("\n程序结束")
"""
generate_nl.py
读取 JSONL 格式的 LTL 公式数据（final_tl 字段），使用 LLM 生成对应的英文描述
支持并行处理加速
"""

import json
import argparse
import os
import asyncio
import aiohttp
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# 尝试导入 tqdm 用于进度条
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# 尝试导入 uvloop，加速事件循环
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass  # uvloop 未安装，使用默认 asyncio 循环

# 加载 .env 文件
load_dotenv()

# 可用的LLM提供商
LLM_PROVIDERS = {
    "openai": "openai",
    "deepseek": "deepseek",
}


@dataclass
class LLMConfig:
    """LLM配置"""

    provider: str = "deepseek"  # 默认使用deepseek
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048


class GlobalRateLimiter:
    """全局速率限制器：基于 RPM 自动计算间隔"""

    def __init__(self, rpm: float):
        """
        :param rpm: 每分钟允许的最大请求数 (Requests Per Minute)
        """
        self.min_interval = 60.0 / max(1.0, float(rpm))  # 转换为秒
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def wait(self) -> None:
        if self.min_interval <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
                now = loop.time()
            self._next_time = now + self.min_interval


class AsyncLLMClient:
    """异步LLM客户端"""

    def __init__(
        self,
        config: LLMConfig,
        limiter: Optional[GlobalRateLimiter] = None,
        timeout_s: float = 60.0,
    ):
        self.config = config
        self.limiter = limiter
        self.timeout = aiohttp.ClientTimeout(total=float(timeout_s))

    async def generate(
        self, session: aiohttp.ClientSession, prompt: str, retry: int = 3
    ) -> str:
        """异步生成回复（支持动态退避）"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        if self.config.provider == "deepseek":
            url = f"{self.config.base_url or 'https://api.deepseek.com'}/v1/chat/completions"
        else:
            url = f"{self.config.base_url or 'https://api.openai.com'}/v1/chat/completions"

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        last_err: Optional[Exception] = None
        for attempt in range(int(retry)):
            try:
                if self.limiter:
                    await self.limiter.wait()

                async with session.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                ) as response:
                    status = response.status

                    # 处理 429 限流，读取 Retry-After 头部
                    if status == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = int(retry_after)
                            except ValueError:
                                wait_time = 2**attempt
                        else:
                            wait_time = 2**attempt
                        await asyncio.sleep(wait_time)
                        continue

                    text = await response.text()

                    if status in (500, 502, 503, 504):
                        raise RuntimeError(
                            f"retryable_status={status} body={text[:300]}"
                        )

                    if status != 200:
                        raise RuntimeError(f"API error: {status} - {text[:500]}")

                    try:
                        data: Dict[str, Any] = await response.json()
                    except Exception as e:
                        raise ValueError(f"Invalid JSON response: {text[:500]}") from e

                    if "choices" not in data or not data["choices"]:
                        raise ValueError(f"Invalid response structure: {data}")

                    msg = data["choices"][0].get("message") or {}
                    content = msg.get("content")
                    if not isinstance(content, str):
                        raise ValueError(f"Missing content in response: {data}")

                    return content

            except (
                asyncio.TimeoutError,
                aiohttp.ClientError,
                RuntimeError,
                ValueError,
            ) as e:
                last_err = e
                backoff = min(30.0, (2**attempt)) + (0.1 * (attempt + 1))
                await asyncio.sleep(backoff)

        raise last_err or RuntimeError("LLM request failed")


def build_translation_prompt(tl_formula: str, num_variations: int = 3) -> str:
    prompt = f"""
You are an expert in formal methods and Linear Temporal Logic (LTL) for autonomous robotic missions.

Your task is to translate an LTL formula into precise English mission descriptions.

### SEMANTIC REQUIREMENTS
The natural language must be a strict semantic interpretation of the LTL formula.

Do not simplify, reinterpret, or omit any logical operator.

### OPERATOR SEMANTICS
Interpret each operator strictly according to LTL semantics.

G φ  : φ must hold at all future time steps (always)

F φ  : φ must occur at least once in the future (eventually)

X φ  : φ must hold at the next time step

φ U ψ : φ must remain true until ψ occurs, and ψ must eventually occur

G F φ : φ must occur infinitely often

F G φ : eventually φ becomes permanently true

### ATOMIC PROPOSITIONS
Keep atomic propositions exactly as written, such as prop_1, prop_2.

Treat them as system states, mission events, or environmental conditions.

Do not rename or reinterpret them.

### COORDINATION LANGUAGE
If the formula contains multiple simultaneous conditions (for example conjunctions),
describe them using coordination phrases such as:

"while simultaneously ensuring that"

"subject to the additional requirement that"

"in parallel with the constraint that"

### VARIATION REQUIREMENTS
Generate {num_variations} linguistically distinct sentences.

Each sentence must describe the same LTL semantics but use different sentence structures.

Possible styles include:

• mission requirement description  
• operational constraint description  
• command-style instruction  
• temporal narrative description  

Avoid repeating the same sentence structure.

### EXAMPLE

LTL:
( not prop_2 until prop_1 ) and globally ( ( not prop_1 or not prop_2 ) and finally prop_1 )

Possible translations:

The system must avoid prop_2 until prop_1 occurs, which must eventually happen; afterwards prop_1 and prop_2 must never occur simultaneously while prop_1 continues to recur indefinitely.

Prop_1 must occur before prop_2 becomes permissible; throughout operation prop_1 and prop_2 must remain mutually exclusive, and prop_1 must repeatedly occur over time.

### OUTPUT FORMAT
Return exactly {num_variations} complete sentences.

Each line must contain exactly one sentence.

IMPORTANT: Each sentence MUST end with a period (.), exclamation mark (!), or question mark (?).

Incomplete sentences are not acceptable.

Do not include:

numbering  
bullet points  
explanations  
extra text  

### TASK

Translate the following LTL formula into {num_variations} COMPLETE sentences:

{tl_formula}

Remember: Each sentence must be grammatically complete and end with proper punctuation.
"""
    return prompt


def parse_llm_response(response: str, expected_count: int = 3) -> List[str]:
    """解析LLM响应，提取生成的句子"""
    sentences = []
    lines = response.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 去除可能的编号前缀，例如 "1."、"2)" 等
        # 更保守的去除方式：只去除行首的数字和标点
        cleaned = re.sub(r'^[\d\.\)\-–—]+\s*', '', line)
        
        # 检查句子完整性：应该以句号、感叹号或问号结尾
        if cleaned and len(cleaned) > 10:
            # 如果句子不以标点符号结尾，可能是截断的
            if not cleaned.endswith(('.', '!', '?')):
                # 尝试找到最后一个完整的句子
                last_punct = max(
                    cleaned.rfind('.'),
                    cleaned.rfind('!'),
                    cleaned.rfind('?')
                )
                if last_punct > 10:  # 确保还有足够的内容
                    cleaned = cleaned[:last_punct + 1]
                else:
                    # 如果没有找到标点，跳过这个不完整的句子
                    continue
            
            sentences.append(cleaned)

    return sentences[:expected_count]


async def translate_single_formula_async(
    client: AsyncLLMClient,
    session: aiohttp.ClientSession,
    tl: str,
    num_variations: int = 3,
    retry: int = 3,
) -> List[str]:
    """异步翻译单个LTL公式"""

    for attempt in range(retry):
        try:
            prompt = build_translation_prompt(tl, num_variations)
            response = await client.generate(session, prompt, retry=3)
            sentences = parse_llm_response(response, num_variations)

            # 检查是否获取到足够数量的完整句子
            if len(sentences) >= num_variations:
                return sentences
            elif len(sentences) > 0:
                # 部分成功，返回有效的句子
                print(f"警告: 仅获取 {len(sentences)}/{num_variations} 个完整句子")
                return sentences

        except Exception as e:
            # 减少错误日志输出，避免影响性能
            if attempt == retry - 1:
                print(f"  最终失败: {e}")
            await asyncio.sleep(2**attempt)

    return []


async def process_single_entry(
    client: AsyncLLMClient,
    session: aiohttp.ClientSession,
    entry: Dict,
    entry_idx: int,
    num_variations: int,
    semaphore: asyncio.Semaphore,
    rate_limiter: GlobalRateLimiter,
) -> List[Dict]:
    """处理单条记录"""

    async with semaphore:
        # 速率限制由 GlobalRateLimiter 统一处理

        # 从 final_tl 字段获取 LTL 公式
        tl = entry.get("final_tl", "")

        if not tl:
            return []

        sentences = await translate_single_formula_async(
            client, session, tl, num_variations
        )

        if not sentences:
            return []

        results = []
        for sent_idx, sentence in enumerate(sentences):
            results.append(
                {
                    "id": entry.get("id", entry_idx),
                    "original_id": entry_idx,
                    "tl": tl,
                    "nl": sentence,
                    "sentence_idx": sent_idx,
                }
            )
        return results


async def process_dataset_async(
    input_path: str,
    output_path: str,
    llm_config: LLMConfig,
    num_variations: int = 3,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_concurrency: int = 10,
    rpm: float = 1000,
) -> List[Dict]:
    """异步并行处理整个数据集"""

    # 读取输入文件
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))

    # 限制处理范围
    if end_idx is None:
        end_idx = len(entries)
    entries = entries[start_idx:end_idx]

    print(f"将处理 {len(entries)} 条记录 (从 {start_idx} 到 {end_idx - 1})")
    print(f"并行数: {max_concurrency}, 目标 RPM: {rpm}")

    # 创建全局限速器
    limiter = GlobalRateLimiter(rpm)

    # 创建LLM客户端和信号量
    # timeout_s 设为 600 秒以应对网络延迟
    client = AsyncLLMClient(llm_config, limiter=limiter, timeout_s=600.0)
    semaphore = asyncio.Semaphore(max_concurrency)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 500

    total_entries = len(entries)
    completed = 0
    produced = 0

    print("开始处理...")
    
    # 创建进度条
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_entries, desc="处理进度", unit="条")
    
    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        with open(output_file, "w", encoding="utf-8") as out_f:
            for batch_start in range(0, total_entries, batch_size):
                batch = entries[batch_start : batch_start + batch_size]

                tasks = []
                for i, entry in enumerate(batch):
                    idx = start_idx + batch_start + i
                    tasks.append(
                        process_single_entry(
                            client,
                            session,
                            entry,
                            idx,
                            num_variations,
                            semaphore,
                            limiter,
                        )
                    )

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(batch_results):
                    completed += 1
                    if isinstance(result, Exception):
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    if not result:
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    if isinstance(result, list):
                        for row in result:
                            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            produced += 1
                    
                    if pbar is not None:
                        pbar.update(1)

                if completed % 100 == 0 or completed == total_entries:
                    print(
                        f"进度: {completed}/{total_entries} (已写入 {produced} 条结果)"
                    )
                    out_f.flush()
    
    # 关闭进度条
    if pbar is not None:
        pbar.close()

    print(f"\n完成! 结果已保存到: {output_path}")
    print(f"共生成 {produced} 条 NL-LTL 对")
    return []


def main():
    parser = argparse.ArgumentParser(
        description="LTL公式(final_tl字段)转英文自然语言句子"
    )

    # 输入输出
    parser.add_argument(
        "-i",
        "--input",
        default="/data/ljr/llm_experiments/ljr_ltl_datasetV1/rawltl_improved.jsonl",
        help="输入JSONL文件路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/data/ljr/llm_experiments/ljr_ltl_datasetV1/new_nl2ltl.jsonl",
        help="输出JSONL文件路径",
    )

    # LLM配置
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek"],
        default="deepseek",
        help="LLM提供商",
    )
    parser.add_argument("--model", default="deepseek-chat", help="模型名称")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API密钥 (也可通过 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量设置)",
    )
    parser.add_argument("--base-url", default="", help="API base URL")

    # 处理选项
    parser.add_argument(
        "-n", "--num-variations", type=int, default=4, help="每个公式生成的句子数量"
    )
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--end", type=int, default=None, help="结束索引")
    parser.add_argument(
        "-c", "--concurrency", type=int, default=20, help="并行数量 (默认50)"
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=500,
        help="每分钟最大请求数 (Requests Per Minute, 默认500)",
    )

    args = parser.parse_args()

    # 获取API密钥 - 优先使用参数，其次使用环境变量
    api_key = args.api_key
    if not api_key:
        if args.provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        api_key = os.environ.get("API_KEY", "")

    if not api_key:
        print("错误: 请通过以下方式提供API密钥:")
        print("  1. --api-key 参数")
        print("  2. 环境变量 DEEPSEEK_API_KEY / OPENAI_API_KEY")
        print("  3. .env 文件中的 API_KEY")
        return

    # 构建配置
    llm_config = LLMConfig(
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
    )

    # 运行异步处理
    asyncio.run(
        process_dataset_async(
            input_path=args.input,
            output_path=args.output,
            llm_config=llm_config,
            num_variations=args.num_variations,
            start_idx=args.start,
            end_idx=args.end,
            max_concurrency=args.concurrency,
            rpm=args.rpm,
        )
    )


if __name__ == "__main__":
    main()

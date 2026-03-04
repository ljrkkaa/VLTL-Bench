"""
LTL公式转英文自然语言句子生成器
读取JSONL格式的LTL公式数据，使用LLM生成对应的英文描述
支持并行处理加速（优化版）
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

# MODIFIED: 尝试导入 uvloop，加速事件循环
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
    max_tokens: int = 1024


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
    """异步LLM客户端（优化版）"""

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

                    # MODIFIED: 处理 429 限流，读取 Retry-After 头部
                    if status == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = int(retry_after)
                            except ValueError:
                                wait_time = 2 ** attempt
                        else:
                            wait_time = 2 ** attempt
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
                backoff = min(30.0, (2 ** attempt)) + (0.1 * (attempt + 1))
                await asyncio.sleep(backoff)

        raise last_err or RuntimeError("LLM request failed")


def build_translation_prompt(masked_tl: str, num_variations: int = 3) -> str:
    """构建更严格的 masked LTL → NL 翻译提示词（保持不变）"""
    prompt = f"""
You are an expert in Linear Temporal Logic (LTL) and formal semantics.
Your task is to translate an LTL formula into natural English sentences
that are semantically equivalent. The translation must preserve the
exact temporal and logical structure.

CRITICAL REQUIREMENT:
The generated sentences must be logically equivalent to the LTL formula.
Do NOT weaken or strengthen the meaning.
Do NOT remove temporal scope.
Do NOT change quantification structure.

------------------------------------------------------------------
LTL Operators and Their Exact Meanings

globally φ
    φ holds at every time step.

finally φ
    φ will hold at some future time step.

globally finally φ
    φ holds infinitely often.

next φ
    φ holds at the next time step (immediate next).

φ until ψ
    φ must continuously hold until ψ becomes true.

φ implies ψ
    Whenever φ holds, ψ must also hold.

------------------------------------------------------------------
IMPORTANT SEMANTIC DISTINCTIONS

1. "whenever φ" corresponds to globally (φ → ...)

2. "if φ ever happens" usually corresponds to
   finally φ → ...
   This is NOT equivalent to globally (φ → ...)

3. "immediately" or "in the next step" must be used
   when translating the next operator.

4. "keep doing X until Y" corresponds to X until Y.

5. "infinitely often" corresponds to globally finally.

Avoid vague words such as:
- ever
- once
- sometime
- maybe
unless they precisely match the operator semantics.

------------------------------------------------------------------
Correct Example

LTL:
globally ( secure(blanket,entrance) implies finally collect(tissue) )

Correct translation:
"Whenever the blanket is secured at the entrance,
the robot must eventually collect the tissue."

Incorrect translation:
"If the blanket is ever secured at the entrance,
then the tissue must eventually be collected."
(This weakens the universal quantification.)

------------------------------------------------------------------
Correct Example with Until and Next

LTL:
globally ( p implies next ( p until q ) )

Correct translation:
"Whenever p occurs, then starting from the next step,
p must continue to hold until q occurs."

------------------------------------------------------------------

Now translate the following *masked* LTL formula into {num_variations}
different natural English sentences.

All sentences must be logically identical paraphrases.
They must describe the exact same behavior.

Each sentence must:
- Preserve exact semantics
- Respect temporal operators precisely
- Sound natural
- Be mission-oriented (robot task style)

Important:
- Atomic propositions are IDs like prop_1, prop_2, ...
- Keep these IDs as-is in the English sentences (do NOT rename them).
- Refer to them as "prop_1", "prop_2" etc.

Masked LTL (with proposition IDs):
{masked_tl}

Output:
Generate {num_variations} sentences.
One sentence per line.
Do not add explanations.
"""

    return prompt


# MODIFIED: 简化解析逻辑，使用 lstrip 去除数字前缀
def parse_llm_response(response: str, expected_count: int = 3) -> List[str]:
    """解析LLM响应，提取生成的句子（优化版）"""
    sentences = []
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        # 去除可能的编号前缀，例如 "1."、"2)" 等
        line = line.lstrip("0123456789. )-–—")
        if line and len(line) > 10:  # 过滤太短的行
            sentences.append(line)

    return sentences[:expected_count]


async def translate_single_formula_async(
    client: AsyncLLMClient,
    session: aiohttp.ClientSession,
    tl: str,
    masked_tl: str,
    num_variations: int = 3,
    retry: int = 3,
) -> List[str]:
    """异步翻译单个LTL公式"""

    for attempt in range(retry):
        try:
            prompt = build_translation_prompt(masked_tl, num_variations)
            response = await client.generate(session, prompt, retry=3)
            sentences = parse_llm_response(response, num_variations)

            if sentences:
                return sentences

        except Exception as e:
            # 减少错误日志输出，避免影响性能
            if attempt == retry - 1:
                print(f"  最终失败: {e}")
            await asyncio.sleep(2 ** attempt)

    return []


async def process_single_entry(
    client: AsyncLLMClient,
    session: aiohttp.ClientSession,
    entry: Dict,
    entry_idx: int,
    num_variations: int,
    semaphore: asyncio.Semaphore,
    rate_limiter: GlobalRateLimiter,  # MODIFIED: 传入 limiter 而非 rate_limit
) -> List[Dict]:
    """处理单条记录"""

    async with semaphore:
        # 速率限制由 GlobalRateLimiter 统一处理，无需额外 sleep

        tl = entry.get("tl", "")
        masked_tl = entry.get("masked_tl", "")

        if not tl or not masked_tl:
            return []

        sentences = await translate_single_formula_async(
            client, session, tl, masked_tl, num_variations
        )

        if not sentences:
            return []

        results = []
        for sent_idx, sentence in enumerate(sentences):
            results.append(
                {
                    "id": f"{entry_idx}_{sent_idx}",
                    "original_id": entry_idx,
                    "tl": tl,
                    "masked_tl": masked_tl,
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
    rpm: float = 120,  # MODIFIED: 使用 rpm 代替 rate_limit
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
    client = AsyncLLMClient(llm_config, limiter=limiter, timeout_s=60.0)
    semaphore = asyncio.Semaphore(max_concurrency)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    failed_path = output_file.with_suffix(output_file.suffix + ".failed.jsonl")

    batch_size = 500

    total_entries = len(entries)
    completed = 0
    produced = 0

    print("开始处理...")
    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        with (
            open(output_file, "w", encoding="utf-8") as out_f,
            open(failed_path, "w", encoding="utf-8") as fail_f,
        ):
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
                            limiter,  # 传递 limiter
                        )
                    )

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(batch_results):
                    completed += 1
                    if isinstance(result, Exception):
                        fail_f.write(
                            json.dumps(
                                {
                                    "original_id": start_idx + batch_start + i,
                                    "error": str(result),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue

                    if not result:
                        fail_f.write(
                            json.dumps(
                                {
                                    "original_id": start_idx + batch_start + i,
                                    "error": "empty_result",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue

                    if isinstance(result, list):
                        for row in result:
                            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            produced += 1
                    else:
                        fail_f.write(
                            json.dumps(
                                {
                                    "original_id": start_idx + batch_start + i,
                                    "error": f"unexpected_result_type={type(result).__name__}",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                # MODIFIED: 降低日志频率
                if completed % 100 == 0 or completed == total_entries:
                    print(
                        f"进度: {completed}/{total_entries} (已写入 {produced} 条结果)"
                    )
                    out_f.flush()
                    fail_f.flush()

    print(f"\n完成! 结果已保存到: {output_path}")
    print(f"失败记录已保存到: {failed_path.as_posix()}")
    print(f"共生成 {produced} 条 NL-LTL 对")
    return []


def main():
    parser = argparse.ArgumentParser(description="LTL公式转英文自然语言句子")

    # 输入输出
    parser.add_argument(
        "-i",
        "--input",
        default="new_generated_datasets/warehouse.jsonl",
        help="输入JSONL文件路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="new_generated_datasets/warehouse_nl.jsonl",
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
        "-n", "--num-variations", type=int, default=3, help="每个公式生成的句子数量"
    )
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--end", type=int, default=None, help="结束索引")
    parser.add_argument(
        "-c", "--concurrency", type=int, default=50, help="并行数量 (默认10)"
    )
    # MODIFIED: 改用 rpm 参数
    parser.add_argument(
        "--rpm", type=float, default=120, help="每分钟最大请求数 (Requests Per Minute, 默认120)"
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
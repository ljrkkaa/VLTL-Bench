"""
generate_cot.py
读取 JSONL 格式的 NL-LTL 对数据 (nl 和 final_tl/tl 字段)
使用 LLM 生成 CoT (Chain of Thought) 推理，并输出 Alpaca 格式的数据
包含 NL-LTL 匹配度验证，仅生成匹配的样本
支持并行处理加速
"""

import json
import argparse
import os
import asyncio
import aiohttp
import re
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

try:
    from dataset_generators.ltl_verifier import are_formulas_equivalent, SPOT_AVAILABLE
except ImportError:
    from ltl_verifier import are_formulas_equivalent, SPOT_AVAILABLE

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


@dataclass
class LLMConfig:
    """LLM配置"""

    provider: str = "deepseek"  # 默认使用deepseek
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.3
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
            url = (
                f"{self.config.base_url or 'https://www.dmxapi.cn'}/v1/chat/completions"
            )

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


def build_ltl_generation_prompt(nl_requirement: str) -> str:
    """构建 NL -> LTL 提示词。"""
    prompt = f"""Translate the following natural language requirement into one LTL formula.

Natural language requirement:
"{nl_requirement}"

Rules:
- Output exactly one LTL formula.
- Do not output explanations.
- Do not output markdown or code fences.
- Use the dataset's textual LTL style when possible.
- Prefer these operator words: globally, finally, next, until, not, and, or, implies.
- Preserve proposition names exactly as they appear in the requirement.

Example:
Natural language requirement:
"Whenever the robot enters region A, it must eventually reach region B."

Output:
globally ( region_A implies finally region_B )

Now output only the formula.
"""
    return prompt


def parse_generated_ltl(response: str) -> Optional[str]:
    """解析模型生成的 LTL 公式。"""
    cleaned = response.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return None

    for line in lines:
        candidate = re.sub(r"^LTL\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        if candidate:
            return candidate

    return None


async def generate_ltl_from_nl_async(
    client: AsyncLLMClient,
    session: aiohttp.ClientSession,
    nl_requirement: str,
    retry: int = 3,
) -> Optional[str]:
    """异步根据 NL 生成候选 LTL。"""
    for attempt in range(retry):
        try:
            prompt = build_ltl_generation_prompt(nl_requirement)
            response = await client.generate(session, prompt, retry=3)
            generated_ltl = parse_generated_ltl(response)

            if generated_ltl and len(generated_ltl) > 3:
                return generated_ltl

        except Exception as e:
            if attempt == retry - 1:
                print(f"  生成候选LTL失败: {e}")
            await asyncio.sleep(2**attempt)

    return None


def build_cot_prompt(nl_requirement: str, ltl_formula: str) -> str:
    """构建 CoT (Chain of Thought) 提示词"""
    prompt = f"""Generate a short reasoning for translating Natural Language (NL) into Linear Temporal Logic (LTL).

NL: "{nl_requirement}"
LTL: "{ltl_formula}"

Rules:
- Keep reasoning concise.
- Use at most 4 short lines.
- Each line describes one step.
- Do not restate the final formula.
- Avoid explanations or background text.

Output format:

<think>
Propositions: ...
Temporal meaning: ...
Operator mapping: ...
Structure: ...
</think>

Example

NL: Always ensure that if prop_1 is true then prop_2 holds in the next step.
LTL: G(prop_1 -> X prop_2)

<think>
Propositions: prop_1, prop_2
Temporal meaning: rule must hold at all times
Operator mapping: always→G, next step→X
Structure: implication between prop_1 and next prop_2
</think>

Now generate reasoning.

NL: "{nl_requirement}"
LTL: "{ltl_formula}"
"""
    return prompt


def parse_cot_response(response: str) -> Optional[str]:
    """解析 LLM 响应，提取完整 <think>...</think> 块"""
    # 直接提取完整的 <think>...</think> 块
    match = re.search(r"<think>.*?</think>", response, re.DOTALL)

    if match:
        think_block = match.group(0).strip()
        return think_block

    # 如果找不到完整的 think 块，返回 None
    return None


async def generate_cot_async(
    client: AsyncLLMClient,
    session: aiohttp.ClientSession,
    nl_requirement: str,
    ltl_formula: str,
    retry: int = 3,
) -> Optional[str]:
    """异步生成 CoT 推理"""

    for attempt in range(retry):
        try:
            prompt = build_cot_prompt(nl_requirement, ltl_formula)
            response = await client.generate(session, prompt, retry=3)

            # 前置过滤：确保响应包含完整的 <think> 标签
            if "<think>" not in response or "</think>" not in response:
                continue

            cot = parse_cot_response(response)

            if cot and len(cot) > 20:
                return cot

        except Exception as e:
            if attempt == retry - 1:
                print(f"  最终失败: {e}")
            await asyncio.sleep(2**attempt)

    return None


async def process_single_entry(
    client: AsyncLLMClient,
    session: aiohttp.ClientSession,
    entry: Dict,
    entry_idx: int,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict]:
    """处理单条记录"""

    async with semaphore:
        # 从条目获取 NL 和 LTL
        nl = entry.get("nl", "")
        tl = entry.get("tl") or entry.get("final_tl", "")

        if not nl or not tl:
            return None

        # 先由 NL 生成候选 LTL，再用 SPOT 做公式等价验证
        generated_tl = await generate_ltl_from_nl_async(client, session, nl)
        if not generated_tl:
            return None

        is_equivalent, _, _, _ = are_formulas_equivalent(generated_tl, tl)
        if not is_equivalent:
            return None

        # 生成 CoT
        cot = await generate_cot_async(client, session, nl, tl)

        if not cot:
            return None

        # 构建纯 Alpaca 格式（cot 已经包含完整的 <think>...</think> 块）
        alpaca_entry = {
            "instruction": "Translate the natural language requirement into LTL.",
            "input": nl,
            "output": f"{cot}\n{tl}",
        }

        return alpaca_entry


async def process_dataset_async(
    input_path: str,
    output_path: str,
    llm_config: LLMConfig,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_concurrency: int = 10,
    rpm: float = 1000,
) -> None:
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
    client = AsyncLLMClient(llm_config, limiter=limiter, timeout_s=600.0)
    semaphore = asyncio.Semaphore(max_concurrency)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 500

    total_entries = len(entries)
    completed = 0
    produced = 0
    skipped = 0
    failed = 0

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
                            semaphore,
                        )
                    )

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    completed += 1
                    if isinstance(result, Exception):
                        failed += 1
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    if result is None:
                        skipped += 1
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    # 写入 Alpaca 格式的数据
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    produced += 1

                    if pbar is not None:
                        pbar.update(1)

                if completed % 100 == 0 or completed == total_entries:
                    print(
                        f"进度: {completed}/{total_entries} (已写入 {produced} 条, 跳过 {skipped} 条, 失败 {failed} 条)"
                    )
                    out_f.flush()

    # 关闭进度条
    if pbar is not None:
        pbar.close()

    print(f"\n完成! 结果已保存到: {output_path}")
    print(f"共生成 {produced} 条 Alpaca+CoT 数据 (经过公式等价验证)")
    print(f"跳过 {skipped} 条 (候选公式不一致或生成失败)")
    print(f"失败 {failed} 条 (异常错误)")


def main():
    parser = argparse.ArgumentParser(
        description="NL-LTL 对生成 Alpaca + CoT 格式的数据 (包含匹配度验证)"
    )

    # 输入输出
    parser.add_argument(
        "-i",
        "--input",
        default="/data/ljr/llm_experiments/ljr_ltl_datasetV1/new_nl2ltl.jsonl",
        help="输入JSONL文件路径 (包含 nl 和 tl/final_tl 字段)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/data/ljr/llm_experiments/ljr_ltl_datasetV2/nl2ltl_alpaca_cot.jsonl",
        help="输出JSONL文件路径 (Alpaca + CoT 格式)",
    )

    # LLM配置
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek"],
        default="openai",
        help="LLM提供商",
    )
    parser.add_argument("--model", default="gpt-4.1-mini", help="模型名称")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API密钥 (也可通过 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量设置)",
    )
    parser.add_argument("--base-url", default="", help="API base URL")

    # 处理选项
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument(
        "--end", type=int, default=10, help="结束索引 (None 表示处理全量)"
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=20, help="并行数量 (默认20)"
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=500,
        help="每分钟最大请求数 (Requests Per Minute, 默认500)",
    )

    args = parser.parse_args()

    if not SPOT_AVAILABLE:
        print("错误: 当前流程依赖 SPOT 做公式等价验证，请先安装并配置 SPOT。")
        return

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
            start_idx=args.start,
            end_idx=args.end,
            max_concurrency=args.concurrency,
            rpm=args.rpm,
        )
    )


if __name__ == "__main__":
    main()

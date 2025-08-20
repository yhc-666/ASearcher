import queue
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

@dataclass
class Record:
    """
    搜索轨迹中单个记录的数据结构
    
    包含四种记录类型：
    - prompt: 初始问题
    - llm_gen: LLM 生成的内容（包含工具调用）
    - search_results: 搜索引擎返回的结果
    - webpage: 访问网页获取的内容
    
    同时存储 RL 训练所需的 tokens 和 logprobs 数据
    """
    type: str # prompt/llm_gen/search_results/webpage
    text: str
    # for webpage and search results
    short_text: str = ""
    # RL data
    input_len: Optional[int] = None
    input_tokens: Optional[List[int]] = None
    output_len: Optional[int] = None
    output_tokens: Optional[List[int]] = None
    output_logprobs: Optional[List[float]] = None
    output_versions: Optional[List[int]] = None

    def to_dict(self):
        return asdict(self)

class AgentMemory:
    """
    Agent 对话记忆管理器
    
    核心功能：
    1. 存储完整的多轮对话历史(prompt + LLM生成 + 工具结果）
    2. 将记忆序列转换为下一轮 LLM 输入 prompt
    3. 统计轨迹执行情况(tokens数、工具调用次数等)
    """
    def __init__(self, prompt):
        self.memory = [Record(type="prompt", text=prompt)]
    
    def llm_gen_count(self):
        return sum([r.type == "llm_gen" for r in self.memory])
    
    def filter_records(self, record_type):
        return [r for r in self.memory if r.type == record_type]
    
    def prepare_prompt(self):
        """
        将记忆中的所有记录拼接为完整的 LLM 输入 prompt
        
        拼接逻辑：
        prompt + search_results + <think> + llm_gen + webpage + <think> + llm_gen + ...
        
        关键点：工具结果后会添加 <think> 标签，引导 LLM 思考
        Why think:
            1. 认知流程模拟: <think> 标签模拟了人类接收新信息后的思考过程，强制模型在行动前进行反思和规划。
            2. 分段式推理: 通过 stop=["</think>"] 实现了"信息输入→思考→行动"的三阶段循环，避免了模型匆忙做出决策。
        """
        prompt = ""
        for r in self.memory:
            if r.type == "prompt":
                prompt = r.text
            elif r.type in ["search_results", "webpage"]:
                prompt = prompt + "\n\n" + r.short_text + "\n<think>\n"
            elif r.type == "llm_gen":
                prompt = prompt + r.text
            else:
                raise RuntimeError(f"Unknown record type: {r.type}")
        return prompt
    
    def add_record(self, r: Record):
        self.memory.append(r)
    
    def logging_stats(self) -> Dict:
        llm_gens = self.filter_records(record_type="llm_gen")
        search_results = self.filter_records(record_type="search_results")
        webpages = self.filter_records(record_type="webpage")
        ret = dict(
            num_llm_gens=len(llm_gens),
            num_input_tokens=sum([len(r.input_tokens) for r in llm_gens]),
            num_output_tokens=sum([len(r.output_tokens) for r in llm_gens]),
            num_search_queries=len(search_results),
            num_success_search_queries=len([r for r in search_results if "No search results are found" not in r.text]),
            num_failed_search_queries=len([r for r in search_results if "No search results are found" in r.text]),
            num_pages=len(webpages),
            num_success_url_accesses=len([r for r in webpages if ">>>> Page 1 >>>>" in r.text]),
            num_failed_url_accesses=len([r for r in webpages if ">>>> Page 1 >>>>" in r.text]),
        )
        return ret
    
    def to_dict(self):
        return [r.to_dict() for r in self.memory]

class SearchAgent:
    """
    单个搜索任务的 Agent 实例
    
    核心职责：
    1. 管理单个问题的完整搜索轨迹
    2. 处理工具调用结果的分页队列（大网页内容分块处理）
    3. 检测任务完成状态（是否输出了 <answer> 标签）
    4. 为 RL 训练提供完整的轨迹数据
    
    注意：每个 SearchAgent 实例对应一条完整的搜索轨迹，训练完成后会被销毁
    """
    def __init__(self, prompt):
        self.prompt = prompt
        self.memory = AgentMemory(prompt=prompt)
        self.summary_job_queue = queue.Queue(128)  # 工具结果分页队列
    
    @property
    def num_turns(self):
        return self.memory.llm_gen_count()
    
    @property
    def is_finished(self):
        """
        检测 Agent 是否已完成任务
        
        判断依据：LLM 生成内容中是否包含 <answer>...</answer> 标签
        这是 ASearcher 框架中表示任务完成的标准格式
        """
        pattern = r'<answer>(.*?)</answer>'
        return any([len(re.findall(pattern, r.text, re.DOTALL)) > 0 for r in self.memory.filter_records("llm_gen")])
    
    def add_summary_jobs(self, summary_jobs):
        if not isinstance(summary_jobs, list):
            summary_jobs = [summary_jobs]
        for summary_job in summary_jobs:
            assert (summary_job.get("type", "unkown") in ["search_results", "webpage"]), ("Unknown summary_job type: " + summary_job.get("type", "unknown"))
            self.summary_job_queue.put_nowait(summary_job)
    
    def prepare_llm_query(self):
        """
        准备下一轮 LLM 生成请求
        
        核心逻辑：
        1. 基础 prompt = 历史对话记录
        2. 如果有待处理的工具结果(job_queue),则添加到 prompt 末尾, 并在最后加入<think>
           - 强制性思考机制: <think> 不是可选的，而是必须的。每当工具结果加入prompt时，系统强制要求模型先思考再行动。
           - Stop Token 控制: 通过 stop=["</think>"] 实现了精确控制，确保模型不能跳过思考阶段直接进入下一个动作。
           - 🔄 完整的执行循环
                1. LLM生成: "我需要搜索..."
                    stop tokens: ["</search>", "</access>", "</answer>"]
                2. 工具执行: 搜索并返回结果
                3. 强制思考阶段:
                    prompt += 搜索结果 + "<think>\n"
                    stop tokens: ["</think>"]  # 只能停在思考结束
                4. LLM思考: "从结果看...下一步应该..."
                    必须输出 </think> 才能停止
                5. 继续生成:
                    stop tokens: ["</search>", "</access>", "</answer>"]
                    可以输出下一个动作

        3. 设置合适的 stop tokens:
           - 正常情况：停在工具调用标签 </search>, </access>, </answer>
           - 处理工具结果时：停在 </think>（让 LLM 先思考再继续）
        
        这是 ASearcher 异步执行的关键：工具结果通过队列异步添加到对话中
        """
        prompt = self.memory.prepare_prompt()
        sampling_params = dict(stop=["</search>", "</access>", "</answer>"])
        if not self.summary_job_queue.empty():
            summary_job = self.summary_job_queue.get_nowait()
            if summary_job["type"] in ["search_results", "webpage"]:
                prompt = prompt + "\n\n" + summary_job["text"] + "\n<think>\n"
                new_record = Record(
                    type=summary_job["type"], 
                    text=summary_job["text"], 
                    short_text=summary_job.get("short_text", summary_job["text"]),
                )
                self.memory.add_record(new_record)
                sampling_params["stop"] = ["</think>"]
        return prompt, sampling_params
    
    def consume_llm_response(self, resp, completion_text):
        """
        处理 LLM 生成结果
        
        核心步骤：
        1. 将 LLM 生成内容存储到记忆中，包含 RL 训练所需的所有数据
        2. 解析工具调用：从生成文本中提取 <search>, <access>, <answer> 标签
        3. 返回工具调用列表供外部执行
        
        XML 标签设计：
        - <search>查询内容</search>: 搜索引擎查询
        - <access>URL</access>: 访问网页内容
        - <answer>最终答案</answer>: 任务完成标志
        """
        new_record = Record(
            type="llm_gen",
            text=completion_text,
            input_len=resp.input_len,
            input_tokens=resp.input_tokens,
            output_len=resp.output_len,
            output_tokens=resp.output_tokens,
            output_logprobs=resp.output_logprobs,
            output_versions=resp.output_versions            
        )
        self.memory.add_record(new_record)

        tool_calls = []
        for pattern in [r'<search>(.*?)</search>', r'<access>(.*?)</access>', r'<answer>(.*?)</answer>']:
            matches = re.findall(pattern, completion_text, re.DOTALL)
            if matches:
                match = matches[-1]
                tool_calls.append(str(pattern.replace('(.*?)', match)))

        return tool_calls

    def consume_tool_response(self, res, topk=5):
        """
        处理工具调用结果，实现异步分页处理机制
        
        搜索结果处理：
        - 取 top-k 个搜索结果
        - 格式化为 <information> 标签包围的文本
        - 每个文档限制在 5000 字符内
        
        网页内容处理（关键的分页机制）：
        - 大网页内容（最大 250K 字符）分割为多个 25K 的块
        - 每个块作为独立的 job 加入队列
        - LLM 会逐个处理这些块，实现渐进式内容理解
        - short_text 存储前 100 字符用于记忆管理
        
        这是 ASearcher 处理超长网页的核心机制
        """
        # process the search results
        if res["type"] == "search":
            summary_job = dict(type="search_results")

            documents = res["documents"][:topk]
            urls = res["urls"][:topk]

            if len(documents) > 0:
                doc_id_template = "[Doc {doc_id}]({url}):\n"
                text = "<information>\n" + "\n\n".join([doc_id_template.format(doc_id=str(k+1), url=url) + doc[:5000] for k, (doc, url) in enumerate(zip(documents, urls))]) + "\n</information>"
            else:
                text = "<information>\nNo search results are found.\n</information>"

            summary_job["text"] = text 
            self.add_summary_jobs(summary_job)
        
        # process the webpage
        elif res["type"] == "access":
            summary_jobs = []          
            page = res["page"]
            if page is not None and page.strip() != "":
                page = page[:250000]  # 限制最大网页长度
                while len(page) > 0 and len(summary_jobs) < 10:  # 最多分割为 10 个块
                    _len = min(25000, len(page))  # 每块最大 25K 字符
                    summary_jobs.append(dict(
                        type="webpage",
                        text=f"<information>\n>>>> Page {len(summary_jobs) + 1} >>>>\n\n" + page[:_len] + "\n</information>",
                        short_text=f"<information>\n>>>> Page {len(summary_jobs) + 1} >>>>\n\n" + page[:100] + "\n</information>",
                    ))
                    page = page[_len:]
            else:
                summary_jobs.append(dict(
                    type="webpage",
                    text="<information>\nNo More Information is Found for this URL.\n</information>",
                ))
            self.add_summary_jobs(summary_jobs)

    def get_answer(self):
        text, _ = self.prepare_llm_query()
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None
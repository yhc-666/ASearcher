import asyncio
import os
import sys
import uuid
import json
import gc
import torch
import torch.distributed as dist
import numpy as np
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast

from dataclasses import dataclass, field

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
)
from areal.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.api.cli_args import GRPOConfig
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, seeding, stats_tracker

from ASearcher.train.prompts import SEARCH_ACCESS_PROMPT_TEMPLATE, SEARCH_ONLY_PROMPT_TEMPLATE, INVALID_PROMPT, VALID_PROMPT
from ASearcher.train.search_agent import SearchAgent
from ASearcher.utils.search_tool import SearchToolBox
from ASearcher.utils.rewards import correct_format_fn

worker_id = uuid.uuid4().hex[:4]

logger = logging.getLogger(f"ASearcher @ {worker_id}")



class ASearcherWorkflow(RolloutWorkflow):
    """
    ASearcher 异步 RL 训练工作流
    
    核心功能：
    1. 异步收集多条搜索轨迹 (collect_agent_trajectory)
    2. 将多轮 LLM 生成拼接为完整训练序列 (arun_episode)
    3. 支持极长搜索轨迹 (最多 128 轮对话)
    """
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dataset_path: str,
        dump_dir: str | None = None,
        max_turns: int = 128,           # 最大对话轮数，支持极长搜索
        n_trajs: int = 1,               # 每个问题收集的轨迹数
        search_client_type: str = "async-online-search-access",  # 搜索客户端类型
        reward_type: str = "F1",        # 奖励类型 (F1 或 EM)
        topk: int = 5,                  # 搜索返回前 k 个结果
        valid_inst_ratio: float = 1.0,  # 有效实例比例 (用于课程学习)
        max_tokens: int = 32000,        # 最大 token 数限制
        search_only: bool = True,       # 是否仅使用搜索 (不使用访问网页)
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.search_only = search_only
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.max_turns = max_turns
        self.n_trajs = n_trajs
        self.reward_type = reward_type
        self.topk = topk
        self.valid_inst_ratio = valid_inst_ratio
        self.search_client_type = search_client_type

        self.toolbox = SearchToolBox(dataset_path=dataset_path, reward_type=self.reward_type, topk=self.topk, search_client_type=self.search_client_type)
    
    async def collect_agent_trajectory(self, valid_inst, qid, prompt, engine):
        """
        异步收集单条 Agent 轨迹 - 异步 RL 的核心
        
        关键机制：
        1. 每个轨迹创建独立 SearchAgent 实例
        2. 多轮迭代：LLM 生成 -> 工具调用 -> 更新状态
        3. 工具调用会中断 LLM 生成，处理后继续
        """
        agent = SearchAgent(prompt)  # 每个轨迹独立的 Agent 实例
        score = 0
        ground_truth = None
        # RID 确保同一轨迹的所有请求路由到相同 SGLang 服务器 (粘性路由)
        traj_rid = uuid.uuid4().hex
        while agent.num_turns < self.max_turns and not agent.is_finished:
            # 1. Agent 构建 prompt (包含历史 + 可能的工具响应)
            query_prompt, sampling_params = agent.prepare_llm_query()

            # 2. 异步调用 LLM 生成 (会在工具标签处停止)
            input_ids = self.tokenizer.encode(query_prompt, add_special_tokens=False)
            req = LLMRequest(
                rid=traj_rid,  # 粘性路由确保一致性
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
            )
            if "stop" in sampling_params:
                req.gconfig.stop = sampling_params["stop"]  # 关键：在 </search>, </access>, </answer> 处停止
            if len(input_ids) + self.gconfig.max_new_tokens >= self.max_tokens:
                break
            resp = await engine.agenerate(req)
            completion_str = self.tokenizer.decode(resp.output_tokens)

            # 3. 解析工具调用 (提取 <search>, <access>, <answer> 标签)
            tool_calls = agent.consume_llm_response(resp, completion_str)

            # 4. 执行工具并计算奖励
            if tool_calls is not None and len(tool_calls) > 0:
                tool_call = tool_calls[0]  # 只处理第一个工具调用
                res = (await self.toolbox.step((qid, [tool_call])))[0]
                
                # 将工具结果加入 Agent 的 job_queue
                agent.consume_tool_response(res, topk=self.topk)

                # 更新奖励和真实答案
                if "score" in res:
                    score = res["score"]
                if "ground_truth" in res:
                    ground_truth = res["ground_truth"]

            # 检查是否遇到结束标记
            if resp.output_tokens[-1] in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                break

        # 5. 计算最终奖励 (结合内容准确性和格式正确性)
        llm_gen_records = agent.memory.filter_records("llm_gen")
        format_reward = float(all([correct_format_fn(i, r.text) for i, r in enumerate(llm_gen_records)]))

        # 综合奖励 = 内容分数 × 格式分数
        score = (score or 0) * format_reward
        pred_answer = agent.get_answer()
        judge_q_invalid = False
        if pred_answer is not None:
            # 惩罚包含问题判断词汇的答案
            judge_q_invalid = any([_c in pred_answer for _c in ["question", "invalid", "appropriate", "valid"]])
        if valid_inst and judge_q_invalid:
            score = -0.5  # 问题判断错误的惩罚
        
        # 收集轨迹统计信息
        stats = agent.memory.logging_stats()
        stats.update(dict(
            score=score,
            judge_q_invalid = judge_q_invalid,
            format_reward=format_reward,
        ))

        return ground_truth, score, agent.memory, stats  # 返回轨迹的完整信息       
    
    async def arun_episode(self, engine, data):
        """
        运行一个完整的训练回合 - 将多条轨迹转换为训练数据
        
        关键步骤：
        1. 并发收集 n_trajs 条轨迹 (异步优势)
        2. 归一化奖励 (减去均值)
        3. 将多轮 LLM 生成拼接为完整序列
        """
        # 获取问题的唯一标识符
        qid = None
        for key in ["query_id", "id", "qid"]:
            qid = data.get(key, None)
            if qid is not None:
                break
        qid = str(qid) or uuid.uuid4().hex

        # 检查是否已经训练过 (支持断点恢复)
        if self.dump_dir is not None:
            import glob
            _pattern = os.path.join(self.dump_dir, "*", f"{qid}.jsonl")
            if len(glob.glob(_pattern)) > 0:
                logger.info(f"{qid} is already trained on")
                return None

        # 准备提示词模板 (根据配置选择搜索模式)
        version = engine.get_version()
        prompt_template = SEARCH_ONLY_PROMPT_TEMPLATE if self.search_only else SEARCH_ACCESS_PROMPT_TEMPLATE
        prompt = prompt_template.format(question=data["question"])
        # 课程学习：随机选择是否要求有效答案
        valid_inst: bool = np.random.uniform(0, 1) <= self.valid_inst_ratio
        if valid_inst:
            prompt = prompt.replace(INVALID_PROMPT, VALID_PROMPT)

        # 异步并发收集多条轨迹 - 异步 RL 的核心优势
        trajs = await asyncio.gather(*[self.collect_agent_trajectory(valid_inst, qid, prompt, engine) for _ in range(self.n_trajs)])

        # 处理收集到的轨迹数据
        ground_truth, scores, results, stats = None, [], [], []
        for gt, score, traj, traj_stats in trajs:
            if gt is not None:
                ground_truth = gt
            scores.append(score)
            stats.append(traj_stats)
        
        # 奖励归一化 (减去均值) - PPO 训练的标准做法
        raw_scores = scores
        score_mean = np.asarray(scores).mean()
        scores = [s-score_mean for s in scores]  # 归一化后的奖励
        if all([s==0 for s in scores]):
            return None  # 如果所有奖励都为 0，跳过此样本

        # 转换为 PPO 训练数据：将多轮 LLM 生成拼接为完整序列
        trajs = [traj for _, _, traj, _ in trajs]
        for i, traj_memory in enumerate(trajs):
            traj_stats = stats.pop(0)
            first_llm_gen = True
            # 处理轨迹中的每个 LLM 生成记录
            for j, record in enumerate(traj_memory.memory):
                if record.type != "llm_gen":  # 只处理 LLM 生成的部分
                    continue
                # 拼接输入和输出 tokens
                seq = record.input_tokens + record.output_tokens
                logprobs = [0.0] * record.input_len + record.output_logprobs  # 输入部分 logprob 为 0
                loss_mask = [0] * record.input_len + [1] * record.output_len  # 只在输出部分计算损失
                versions = [-1] * record.input_len + record.output_versions

                # 构建训练数据张量
                res = dict(
                    input_ids=torch.tensor(seq).unsqueeze(0),          # 完整序列
                    loss_mask=torch.tensor(loss_mask).unsqueeze(0),     # 损失掩码
                    logprobs=torch.tensor(logprobs).unsqueeze(0),       # 对数概率
                    versions=torch.tensor(versions).unsqueeze(0),       # 模型版本
                    attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                    rewards=torch.tensor([float(scores[i])]),           # 归一化奖励 (同一个traj内，每个round的reward一样)
                )
                # 标记轨迹开始 (用于 PPO 的 advantage 计算)
                if first_llm_gen:
                    res.update(dict(begin_of_trajectory=torch.tensor([1.0]),))
                    res.update({k: torch.tensor([v]) for k, v in traj_stats.items()})
                    first_llm_gen = False
                else:
                    res.update(dict(begin_of_trajectory=torch.tensor([0.0]),))
                    res.update({k: torch.tensor([v]) for k, v in traj_stats.items()})
                results.append(TensorDict(res, batch_size=[1]))

        # 保存轨迹数据 (用于调试和分析)
        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            # 将轨迹保存为 JSONL 格式
            with open(
                os.path.join(self.dump_dir, str(version), f"{qid}.jsonl"), "w"
            ) as f:
                for i, (traj_memory, raw_score) in enumerate(zip(trajs, raw_scores)):
                    f.write(json.dumps(dict(memory=traj_memory.to_dict(), reward=raw_score, ground_truth=ground_truth, traj_idx=i)) + "\n")

        # 将所有训练数据拼接成批次
        results = concat_padded_tensors(results)
        return results  # 返回 PPO 训练可用的批次数据

@dataclass
class AgentRLConfig(GRPOConfig):
    max_turns: int = field(
        default=128,
        metadata={
            "help": "maximum number of turns for search agent"
        }
    )
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        }
    )
    search_client_type: str = field(
        default="async-online-search-access",
        metadata={
            "help": "Type of tool (async-online-search-access/async-search-access). By default we use 'async-online-search-access'"
        }
    )
    reward_type: str = field(
        default="F1",
        metadata={
            "help": "The type of reward function"
        }
    )
    topk: int = field(
        default=5,
        metadata={
            "help": "search returns the top-k results. Default top_k=5"
        }
    )
    valid_inst_ratio: float = field(
        default=1.0,
        metadata={
            "help": "We randomly force a ratio of queries to produce valid anwers. By default valid_inst_ratio=1.0"
        }
    )
    recover_start_step: int = field(
        default=0,
        metadata={
            "help": "The step to recover from"
        }
    )


def get_search_dataset(dataset_path, tokenizer, rank, world_size):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    # dataset = dataset.filter(lambda x: len(tokenizer.encode(x["question"])) <= 1024)
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def main(args):
    """
    ASearcher 异步 RL 训练主函数
    
    训练流程：
    1. 初始化分布式环境和数据
    2. 创建推理引擎 (SGLang) 和训练引擎 (PPO)
    3. 异步收集轨迹并进行 PPO 训练
    4. 定期更新模型权重和保存检查点
    """
    config, _ = load_expr_config(args, AgentRLConfig)
    config: AgentRLConfig

    # 分布式训练环境设置
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    # 创建分布式数据加载器
    worker_batch_size = config.train_dataset.batch_size // world_size
    train_dataloader = StatefulDataLoader(
        get_search_dataset(config.train_dataset.path, tokenizer, rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,  # 不做额外处理
        drop_last=config.train_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # 初始化推理引擎 (用于轨迹收集)
    rollout = RemoteSGLangEngine(config.rollout, worker_id=worker_id)
    rollout.initialize(None, ft_spec)

    # 初始化训练引擎 (PPO Actor)
    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None  # 参考模型 (可选)

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    weight_update_meta = [
        WeightUpdateMeta.from_disk(config.saver)
        # WeightUpdateMeta.from_fsdp_nccl(
        #     AllocationMode.from_str(config.allocation_mode), actor
        # )
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = ASearcherWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        dataset_path=config.train_dataset.path,
        max_turns=config.max_turns,
        n_trajs=config.n_trajs,
        search_client_type=config.search_client_type,
        reward_type=config.reward_type,
        topk=config.topk,
        valid_inst_ratio=config.valid_inst_ratio,
        max_tokens=config.actor.mb_spec.max_tokens_per_mb,
    )

    # Run training.
    saver = Saver(config.saver, ft_spec, for_recover=False)
    stat_logger = StatsLogger(config.stats_logger, ft_spec)

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    # logger.commit(f"total_epochs={total_epochs} step_per_epoch={steps_per_epoch}")
    data_generator = iter(train_dataloader)
    start_step = config.recover_start_step or 0
    # 主训练循环 - 异步 RL 训练
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch

        print(f"Epoch {epoch}. Step: {step}/{steps_per_epoch}")

        # 1. 轨迹收集阶段 - 异步执行搜索任务
        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                # 异步模式：自动管理批次和并发
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow, expected_batch_size=worker_batch_size)
            else:
                # 同步模式：逐个处理数据
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(train_dataloader)
                    data = next(data_generator)
                batch = rollout.rollout_batch(data, workflow=workflow)

        # 2. PPO 训练阶段
        batch = batch.to(actor.device)
        # 同步所有进程，确保轨迹收集完成
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        # 重新计算对数概率 (如果需要)
        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        # 计算参考模型概率 (如果有参考模型)
        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        # 计算 GAE 优势函数
        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")
        
        # 清理 GPU 内存
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        # PPO 更新步骤
        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)  # 执行 PPO 损失计算和反向传播
            actor.step_lr_scheduler()        # 更新学习率
            log_gpu_stats("ppo update")

        # 3. 模型权重同步 - 关键的异步训练协调步骤
        with stats_tracker.record_timing("update_weights"):
            rollout.pause()  # 暂停推理引擎
            if dist.get_rank() == 0:
                # 主进程启动权重更新
                future = rollout.update_weights(weight_update_meta)
            # 所有进程上传权重到推理服务器
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()  # 等待权重更新完成
            # 同步所有进程
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()
            rollout.resume()  # 恢复推理引擎
            # 更新版本号 (用于轨迹标记)
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

        # 4. 保存检查点
        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step)

        # 5. 记录训练统计
        stat_logger.commit(epoch, step, global_step, stats)

    stat_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])

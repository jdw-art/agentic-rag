# Ragas Evaluation

本目录包含基于 **Ragas** 的项目级 RAG 评测脚本，直接复用本项目现有的 `chat_with_agent()` 与 `rag_trace["retrieved_chunks"]`，而不是单独再实现一套检索链。

## 文件说明

- `rag_eval_example.py`
  示例脚本，仅作参考。
- `run_ragas_eval.py`
  正式评测脚本。读取 `data/test/labor_law_ragas_dataset.csv` 的 10 条样本，调用本项目 Agent 生成 `response` 和 `retrieved_contexts`，并评测：
  - `faithfulness`
  - `context_recall`
  - `context_precision`

## 评测数据格式

输入数据集位于：

- [labor_law_ragas_dataset.csv](/Users/jacob/GitProject/SuperMew/data/test/labor_law_ragas_dataset.csv)

输入 CSV 只需要两列：

```csv
user_input,reference
```

执行评测时，脚本会自动补齐：

- `response`
- `retrieved_contexts`

其中 `retrieved_contexts` 来自每轮问答返回的 `rag_trace["retrieved_chunks"]` 中的 `text` 字段。

## 运行前准备

1. 确保 `.env` 已配置模型访问参数。

脚本会按以下优先级读取评测用模型配置：

- `OPENAI_API_KEY`，否则回退到 `ARK_API_KEY`
- `OPENAI_API_BASE_URL` / `OPENAI_BASE_URL`，否则回退到 `BASE_URL`
- `GRADE_MODEL`，否则回退到 `MODEL`

2. 确保项目依赖服务可用：

- PostgreSQL
- Redis
- Milvus
- 已完成劳动法文档入库，否则知识库检索拿不到有效上下文

3. 安装评测依赖。

项目默认依赖里没有内置 `ragas` 与 `datasets`，需要额外安装：

```bash
uv add ragas datasets pandas
```

或：

```bash
pip install ragas datasets pandas
```

## 运行方式

在项目根目录执行：

```bash
python evaluate/run_ragas_eval.py
```

如果你使用 `uv`：

```bash
uv run python evaluate/run_ragas_eval.py
```

## 输出结果

脚本会在 `evaluate/output/` 下生成带时间戳的结果文件：

- `ragas_eval_YYYYMMDD_HHMMSS_samples.jsonl`
  每条样本的完整问答结果，包含 `response`、`retrieved_contexts` 和 `rag_trace`
- `ragas_eval_YYYYMMDD_HHMMSS_samples.csv`
  便于查看的扁平化样本结果
- `ragas_eval_YYYYMMDD_HHMMSS_metrics.json`
  聚合指标结果
- `ragas_eval_YYYYMMDD_HHMMSS_metrics_detail.csv`
  如果当前 Ragas 版本支持 `to_pandas()`，会额外输出逐样本指标明细

## 指标说明

- `faithfulness`
  评估 `response` 是否被 `retrieved_contexts` 支撑。
- `context_recall`
  评估 `retrieved_contexts` 是否覆盖 `reference` 中的关键信息。
- `context_precision`
  评估 `retrieved_contexts` 的排序与相关性质量。

## 说明

- 脚本会为每条样本创建独立 `session_id`，避免多轮上下文串扰。
- 评测直接调用本项目 Agent，因此结果会真实反映当前系统的检索、改写、生成链路表现。
- 若 `retrieved_contexts` 为空，通常说明文档未入库、向量库不可用，或 Agent 没有触发知识库工具。

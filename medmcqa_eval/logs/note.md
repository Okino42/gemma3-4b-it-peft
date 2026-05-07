  eval_medmcqa.py
  这是核心评测脚本，用来跑某一个模型在 MedMCQA validation 上的推理。

  输入：

  - --model-name：必须传，值来自 configs/models.yaml
  - configs/models.yaml：告诉它这个模型是 base 还是 adapter，路径在哪里
  - configs/eval.yaml：告诉它数据文件、字段名、prompt、batch size、生成参数、输出目录
  - data/validation-00000-of-00001.parquet：MedMCQA validation 数据

  典型命令：

  python eval_medmcqa.py --model-name gemma-3-4b-med-lora-s1

  它做的事：

  1. 从 models.yaml 找到 gemma-3-4b-med-lora-s1 的配置。
  2. 读取 base 模型路径 /projects/checkpoint/gemma-3-4b-it。
  3. 如果有 adapter_path，就用 PeftModel.from_pretrained() 加载 LoRA/DoRA adapter。
  4. 读取 parquet 数据。
  5. 对每条题目调用 prompt_templates.py 生成 chat messages。
  6. 用 tokenizer 的 apply_chat_template(..., add_generation_prompt=True) 转成 Gemma prompt。
  7. 批量 model.generate()。
  8. 从模型输出里解析 A/B/C/D。
  9. 写出预测文件和指标文件。

  输出：

  outputs/<model_name>/predictions.jsonl
  outputs/<model_name>/metrics.json

  例如：

  outputs/gemma-3-4b-med-lora-s1/predictions.jsonl
  outputs/gemma-3-4b-med-lora-s1/metrics.json

  predictions.jsonl 每行大概是：

  {
    "id": "...",
    "model_name": "gemma-3-4b-med-lora-s1",
    "question": "...",
    "options": {
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    },
    "gold_index": 2,
    "gold_letter": "C",
    "pred_letter": "C",
    "prediction_text": "C",
    "correct": true,
    "choice_type": "single",
    "subject_name": "Medicine",
    "topic_name": null
  }

  metrics.json 包括：

  - 总题数
  - 解析出答案的数量
  - 未能解析 A/B/C/D 的数量
  - 正确数
  - accuracy
  - answered-only accuracy
  - 按 subject_name 分组的准确率
  - 按 choice_type 分组的准确率

  额外参数：

  python eval_medmcqa.py --model-name gemma-3-4b-it --limit 20

  只跑前 20 条，适合在计算节点上 smoke test。

  python eval_medmcqa.py --model-name gemma-3-4b-it --resume

  如果 predictions.jsonl 已经有部分结果，会跳过已有 id，继续补跑。

  python eval_medmcqa.py --model-name gemma-3-4b-it --overwrite

  删除旧预测，重新跑。

  prompt_templates.py
  这个文件不直接运行，它是 eval_medmcqa.py 调用的工具文件。

  输入：

  - 一条 MedMCQA 样本，比如包含：
      - question
      - opa
      - opb
      - opc
      - opd
  - configs/eval.yaml 里的字段映射和 prompt 配置

  它做的事：

  把一条 parquet 数据转成 Gemma chat messages：

  [
      {
          "role": "system",
          "content": "You are a careful medical exam assistant. Choose the single best answer from the given options."
      },
      {
          "role": "user",
          "content": "Question:\n...\n\nOptions:\nA. ...\nB. ...\nC. ...\nD. ...\n\nAnswer with only one letter: A, B, C, or D."
      }
  ]

  然后 eval_medmcqa.py 会继续把这个 messages 交给：

  tokenizer.apply_chat_template(...)

  这样好处是 prompt 逻辑集中在一个地方，以后如果你要改成中文 prompt、zero-shot prompt、few-shot prompt，不需要改模型加载和打分逻辑。

  输出：

  - Python 内部的 list[dict]，也就是 chat messages
  - 不写文件

  score_medmcqa.py
  这个脚本负责对某一个模型已经生成好的预测文件重新打分。

  输入：

  - --model-name
  - 对应模型的：

    outputs/<model_name>/predictions.jsonl

  典型命令：

  python score_medmcqa.py --model-name gemma-3-4b-med-lora-s1

  它做的事：

  1. 读取 predictions.jsonl。
  2. 对比每条的 pred_letter 和 gold_letter。
  3. 重新计算：
      - accuracy
      - correct / total
      - unparsed
      - confusion matrix
      - 按 subject 分组准确率
      - 按 choice_type 分组准确率
  4. 写回 metrics.json。

  输出：

  outputs/<model_name>/metrics.json

  例如：

  outputs/gemma-3-4b-med-lora-s1/metrics.json

  为什么要有它？

  因为有时候你可能改了答案解析规则，但不想重新跑 5 个大模型。只要 predictions.jsonl 还在，就可以轻量重新打分。

  summarize_results.py
  这个脚本负责汇总所有模型的结果，不跑模型，也不重新逐题打分。

  输入：

  - 自动扫描：

    outputs/*/metrics.json

  典型命令：

  python summarize_results.py

  它做的事：

  1. 找到每个模型目录下的 metrics.json。
  2. 抽取关键指标：
      - model_name
      - accuracy
      - correct
      - total
      - answered
      - unparsed
      - accuracy_answered_only
  3. 合并成一个总表。

  输出：

  results/summary.csv
  results/summary.json

  summary.csv 适合直接看或导入 Excel，格式大概是：

  model_name,accuracy,correct,total,answered,unparsed,accuracy_answered_only,metrics_path
  gemma-3-4b-it,0.42,1757,4183,4170,13,0.4213,outputs/gemma-3-4b-it/metrics.json
  gemma-3-4b-med-lora-s1,0.45,1882,4183,4180,3,0.4502,outputs/gemma-3-4b-med-lora-s1/metrics.json

  整体流程
  你后面实际用的时候是这样：

  1. 提交一个模型：

  sbatch scripts/run_medmcqa_one.sh gemma-3-4b-med-lora-s1

  这个 .sh 内部会依次跑：

  python eval_medmcqa.py --model-name gemma-3-4b-med-lora-s1
  python score_medmcqa.py --model-name gemma-3-4b-med-lora-s1

  2. 五个模型都跑完后，登录节点汇总：

  PYTHONNOUSERSITE=1 python summarize_results.py

  最终你主要看：

  results/summary.csv

  更细的逐题错误分析看：

  outputs/<model_name>/predictions.jsonl
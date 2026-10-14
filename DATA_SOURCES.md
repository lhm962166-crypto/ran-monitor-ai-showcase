# 数据与功能参考

## TelecomTS 数据

训练数据来源：Ali Maatouk et al., **TelecomTS: A Multi-Modal Telecom Dataset for Time Series and Language Analysis**，见 [Hugging Face 数据集页](https://huggingface.co/datasets/AliMaatouk/TelecomTS) 和 [论文页面](https://proceedings.mlr.press/v306/feng26s.html)。数据卡元数据将许可标记为 MIT；上游文件列表没有单独的 LICENSE 文件。仓库不分发该数据或其派生样本，使用者应在获取数据时核对上游当时的许可、条款和文件说明。

此前的本地开发使用了 690 个窗口的派生子集。当前 Git 仓库不分发该子集、固定评估切分或基于它训练的模型。根目录中的 `sample_window.json` 以及 `ran_monitor_ai/demo.py` 运行时生成的窗口由本项目代码生成，不含 TelecomTS 样本值；它们用于推理流程和软件演示，不是经过真实网络验证的数据集。

要训练应用，请自行从上游数据页取得数据，并准备满足 [数据格式说明](README.md#json-窗口格式) 的 JSONL 文件。本仓库根目录的 MIT 许可证适用于项目代码，不取代上游数据许可，也不覆盖依赖。`RAN_MONITOR_DATA_PATH` 可指定本机训练文件路径。项目生成器只使用代码中写明的基线与扰动规则，不会读取本机 TelecomTS 文件。

## 功能参考

功能行为参考：[TAB-0705/5G_RAN_Anomaly_Detection](https://github.com/TAB-0705/5G_RAN_Anomaly_Detection)。本项目是独立实现，没有复制该项目的源文件或模型文件。参考项目与数据集都在 README 和本文件中保留出处。

## 评测边界

历史开发子集有 690 个窗口、420 个重叠时间组，但没有明确的基站、数据流或故障事件 ID；故障标签集中在较早时间块，最后两个时间块没有故障。其分层组留出仅用于旧开发阶段的对照，不能证明独立事件或未来站点泛化。由于数据和切分未随仓库发布，这些历史指标无法通过干净克隆复现，不能当作本仓库当前可复现的基准。

模型评估只读取记录中明确提供的数据来源标签，例如 `anomaly_origin` 或 `labels.origin`，不根据故障类别或 KPI 猜测数据来源。实时分数漂移提示只表示模型输出分布变化，不代表准确率下降；没有人工标签时不会自动调整模型。

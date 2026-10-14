# GitHub 发布检查清单

公开版使用一个新的、干净历史的仓库；旧的 `ran-monitor-ai` 仓库继续保持 Private。此清单只授权发布本目录中通过检查的项目文件，不把旧仓库提交记录、Actions 历史或本机实验文件迁过去。

## 署名、许可和来源

- Git 提交身份使用 GitHub 用户名和 GitHub noreply 邮箱；项目对外介绍仍按个人真实贡献署名。
- 根目录 [LICENSE](../LICENSE) 只适用于项目代码和生成器代码，不覆盖第三方数据与依赖。
- README 与 [DATA_SOURCES.md](../DATA_SOURCES.md) 保留功能参考项目、TelecomTS 论文和数据集链接；合成演示明确标注为项目生成。
- 当前公开候选不包含 TelecomTS 数据、旧固定切分或基于它训练的模型。历史开发结果保留在模型卡中，并明确标为不可复现的历史记录。

## 干净克隆验收

在新的 Windows 环境中按 README 创建虚拟环境并安装依赖后，先运行自动测试和编译，再启动实际交互演示：

```powershell
python -m compileall -q app.py ran_monitor_ai tests scripts
python -m unittest discover -s tests -v
python -m ran_monitor_ai demo
```

测试使用项目生成的固定种子合成 fixture，无需 TelecomTS 或 SMTP/Twilio 凭据；发布前以完整测试运行的实际结果为准，并确认无意外跳过。检查浏览器中的检测、告警保存/处置、审计、回放与报告导出。截图必须来自真实运行的页面，并在说明中标记合成数据。CI 使用 Ubuntu/Windows × pandas 2/3 四种组合；新候选的 Actions 运行必须全部通过后再公开。

`evaluate`、`compare-detectors`、`verify_reproducibility.py` 与 `promotion-check` 面向额外提供的训练数据，不是干净演示的前置条件。演示数据不能通过生产晋级门禁。

## 候选仓库卫生

- 仅从核验后的提交文件创建候选，不递归复制整个工作目录。不要包含 `.git`、`.env`、`data/*.jsonl`、`artifacts/`、SQLite 文件、缓存、虚拟环境或用户运行日志。
- 发布前检查完整文件清单、暂存区、忽略规则，并运行可用的密钥扫描器。确认 `.env.example` 只含空占位符。
- 新仓库先创建为 Private，再推送单一干净根提交；核对远端历史中没有旧提交、数据对象或本机凭据。
- 从新仓库地址再次克隆，重新执行安装、测试和演示。检查 README、链接、截图、LICENSE 与实际候选一致。
- 确认新仓库 Actions 全部通过，并验证匿名访问时 README、源代码、LICENSE 和截图可见，再执行公开发布步骤。保留旧私有仓库及其 URL 不变。

## 面试演示

```powershell
python -m ran_monitor_ai demo
```

展示一条窗口如何进入检测、查看指标与原因候选、保存并处置告警、检查审计，再导出报告。介绍任何数字时同时说明数据规模、切分方式和测量环境；合成演示的分数不能说成真实网络准确率或生产效果。

GitHub 仓库简介可以使用：

> 中文 5G RAN KPI 异常监测演示：两阶段故障研判、可审计告警、事件感知时间组评测与可复现合成演示。

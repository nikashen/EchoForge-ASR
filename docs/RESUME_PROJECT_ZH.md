# EchoForge-ASR：中文流式 ASR 鲁棒性实验室

## 一句话定位

面向语音/音频算法岗位的中文流式 ASR 实验平台：用真实 PCM WebSocket 传输、确定性 VAD、双阶段转写修订和可审计 VoiceLab 界面，把“实时识别链路”和“实验结论证据”放在同一个可复现系统中。

## 核心难点与方案

- **实时音频链路的正确性与稳定性**：定义 `EFA1` 二进制协议，固定 16 kHz、单声道、PCM16LE，校验 magic、长度、采样数、序号和 generation；重复帧幂等、序号跳跃拒绝，并用有界队列、单消费者、心跳和明确关闭码处理背压与断连。
- **流式结果会持续变化**：将结果建模为单调修订链 `partial -> stream_final -> dual_pass_final`，维护稳定前缀并输出字符级 repair diff；二阶段 verifier 失败时降级为 `stream_only / VERIFIER_FAILED`，不伪造修订成功。
- **VAD 不能依赖 chunk 边界**：实现双阈值滞回、speech start/end debounce、跨 chunk 状态、flush/reset/snapshot，使语音起止事件在不同分块方式下保持确定性。
- **模型运行时与演示环境隔离**：以协议注入 `sherpa-onnx` 流式适配器和 `faster-whisper` 端点验证适配器，模型懒加载且只接受操作者提供的本地目录；fake backend 用于 CI 和 Pages，避免下载权重并保持演示可复现。
- **真实模型接入先做前置校验**：提供 `echoforge preflight`，在不加载权重的前提下 fail-closed 检查模型目录、关键 ONNX/Whisper 文件和运行时依赖，避免把服务生命周期 ready 误写成模型可推理或质量结论。
- **实验结果需要可追溯**：提供固定种子的噪声/信噪比和电话信道扰动、中文归一化、编辑计数及冻结 manifest 校验；证据不完整时返回 `not_yet_evaluated`，而不是输出看似完整的结论。
- **从算法状态到可观察产品界面**：浏览器通过 AudioWorklet 采集并重采样到 16 kHz，支持 Mic/File/确定性 Demo 三种来源；VoiceLab 展示波形、VAD、修订时间线、热词能力状态和 JSON/SRT/VTT 导出。

## 可验证结果

- 本地测试套件：**119 tests passed**。
- CI 覆盖 Python 3.11/3.12，执行 pytest、ruff、mypy、compileall、wheel smoke 和 Pages 构建。
- GitHub Pages 已提供无需模型和 API key 的确定性演示，可复现协议状态、修订链、repair diff、时间线和证据边界。

## 简历 Bullet（可直接使用）

- 设计并实现中文流式 ASR 实验平台 EchoForge-ASR，构建 `EFA1` PCM16LE WebSocket 协议，覆盖序号/generation 校验、重复帧幂等、乱序拒绝、背压、心跳和断连清理。
- 实现跨 chunk 确定性双阈值 VAD 与 debounce 状态机，串联流式 decoder 和端点 verifier，建立 `partial -> stream_final -> dual_pass_final` 单调修订链及字符级纠错 diff。
- 以依赖注入方式接入 sherpa-onnx 与 faster-whisper，模型懒加载、权重不入库；提供 deterministic fake backend，使 CI 与 GitHub Pages 在无模型环境下仍可复现完整交互链路。
- 开发 FastAPI/WebSocket 服务和 VoiceLab 前端，支持真实麦克风/文件流、VAD telemetry、热词能力协商、Session Timeline 以及 JSON/SRT/VTT 导出。
- 建立固定种子扰动、中文文本归一化、编辑计数和冻结 manifest 的 fail-closed 评估流程；以 119 项测试、双版本 Python CI、Pages 构建和真实模型接入的静态 preflight 机制验证工程可交付性。

## 面试深挖点

1. **为什么使用二进制帧而不是 JSON 音频？** 说明 PCM 的带宽和编码开销、固定帧头如何支持快速拒绝，及序号/采样数双重记账如何发现丢帧或截断。
2. **如何保证 VAD 不受分块方式影响？** 说明跨 chunk 保留能量状态、双阈值滞回和起止 debounce；对 flush、reset、snapshot 的边界行为给出测试思路。
3. **为什么不直接覆盖 partial？** 说明稳定前缀、单调 revision id、事件溯源和字符级 diff 如何让前端增量渲染、回放和问题定位都可解释。
4. **流式 decoder 或 verifier 出错时怎么办？** 流式状态不可安全回滚，因此 fail-closed；verifier 失败保留已确认的 stream final，并显式发出降级原因。
5. **Fake backend 的价值和局限是什么？** 它验证传输、状态机、UI 和导出，不代表模型识别能力；真实模型必须通过本地路径和 readiness 检查接入。
6. **怎样避免演示数据被误解为模型指标？** Pages 使用独立 `DemoTransport`，Evidence 页面标明运行时和声明边界；只有冻结样本、模型哈希、归一化规则和可复算报告齐备后才发布评估结论。

## 证据边界

- Pages 是确定性协议/界面回放，不是托管推理服务；其固定文本和耗时不能作为模型或生产系统结论。
- 当前仓库不携带模型权重、原始录音或未授权评估数据，也没有冻结的公开评估报告；因此不对识别质量、实时性、生产 SLA、方言/远场泛化或业务收益作量化承诺。
- 真实评估必须记录数据与模型哈希、样本划分、文本归一化版本和完整报告，并由独立流程复算；证据缺失或校验失败时保持 `not_yet_evaluated`。

## 相关入口

- 项目总览：[README.md](../README.md)
- 系统边界：[ARCHITECTURE.md](ARCHITECTURE.md)
- 二进制协议：[WEBSOCKET_PROTOCOL.md](WEBSOCKET_PROTOCOL.md)
- 评估规则：[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)
- 发布边界：[PUBLICATION_BOUNDARY.md](PUBLICATION_BOUNDARY.md)

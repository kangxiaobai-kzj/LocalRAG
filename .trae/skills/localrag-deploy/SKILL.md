---
name: "localrag-deploy"
description: "一键部署 LocalRAG 项目（源码方式）：前置体检、venv/依赖/模型/配置/启动全流程引导与逐步校验。当用户要求部署、安装、启动本项目，或部署/启动报错时触发，自动诊断并修复常见问题。"
---

# LocalRAG 一键部署（源码方式）

本 Skill 让智能体以"分步执行 + 逐步校验 + 出错自愈"的方式，帮用户完成 LocalRAG 的源码部署与启动，并在部署过程中遇到问题时自主定位、修复、验证，减少用户手动折腾。

## 何时使用

- 用户说"部署 / 安装 / 搭建 / 启动 / 运行 LocalRAG"、"帮我跑起来"
- 部署或启动过程报错（依赖、模型、端口、编码、权限等）
- 首次运行引导（建环境、装依赖、下模型、填配置）

## 项目关键事实

- 仓库：`https://github.com/kangxiaobai-kzj/LocalRAG`
- 要求：Python 3.10+；Windows 10/11 优先（DPAPI 加密、内嵌 OCR 二进制基于 Windows）
- 入口：`start_agent.bat`（一键）/ `streamlit run streamlit_app.py`（命令行）；`mcp_server.py` 为 MCP 服务端
- 依赖：`requirements.txt`（版本锁定，含 Windows 专有包如 pywin32）
- 模型：Embedding（FastEmbed / bge-small-zh-v1.5，约 90MB）+ 重排（bge-reranker-v2-m3，约 2.2GB）；重排缺失时检索自动降级
- 配置：`config.json` 首次启动本地生成（gitignore）；API Key 在 Web「设置」页填写；密钥 Windows 用 DPAPI 加密，非 Windows 退化为 base64 混淆
- 服务：Streamlit，端口 8501，仅绑定 `127.0.0.1`（`.streamlit/config.toml`）
- 模型缓存可配置：`FASTEMBED_CACHE_PATH` / `HF_HOME`（默认用户目录）

## 部署流程（每步执行后校验，通过再进下一步）

1. **前置体检**
   - `python --version` 需 ≥3.10；不在 PATH 则引导安装并勾选 "Add python.exe to PATH"
   - 确认磁盘剩余空间 ≥ 5GB（模型 + 依赖约 3GB）
   - 确认网络可达 HuggingFace（默认镜像 `hf-mirror.com`，见 `HF_ENDPOINT`）
2. **克隆仓库**：`git clone https://github.com/kangxiaobai-kzj/LocalRAG.git` 到目标目录
3. **创建虚拟环境**
   - `python -m venv venv`，激活（Windows：`venv\Scripts\activate`）
   - `python -m pip install --upgrade pip`
4. **安装依赖**：`pip install -r requirements.txt`
   - 失败时先定位根因（网络/版本/平台包），可切换国内镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`
5. **校验/下载模型**：`python scripts\check_models.py`
   - 完整则继续；缺失时运行 `python scripts\download_models.py`（或 `install_models.bat`）联网下载
   - 下载超时可检查 `HF_ENDPOINT` 镜像、磁盘空间、断点重试
6. **配置引导**：启动后提示用户到 Web「设置」页选择服务商、填 API Key、点「测试连接」
7. **启动并验证**
   - `start_agent.bat`（或 `streamlit run streamlit_app.py`）
   - 验证 `http://127.0.0.1:8501` 可访问；走通一次问答（含知识库上传与检索）确认端到端可用

## 常见问题诊断表

| 现象 | 处置 |
| :--- | :--- |
| 依赖安装失败 | 看 pip 报错：网络超时换镜像源；版本冲突看是否需先升级 pip；`pywin32` 等 Windows 包在非 Windows 平台需跳过并提示降级 |
| 模型下载失败 / 超时 | 检查 `HF_ENDPOINT`（默认 hf-mirror.com）、磁盘空间；重试；Embedding 缺失时提示检索会降级 |
| 端口 8501 被占用 | 检查是否已有实例；或改 `.streamlit/config.toml` 端口 |
| `start_agent.bat` 中文乱码 / 命令碎片 | cmd 以 GBK 解析 bat，确认文件编码与 `chcp` 一致；或改用命令行逐步执行定位 |
| 中文用户名 / 路径导致链接失败 | 构建/临时目录改用 ASCII 路径（如设置 `TEMP`/`TMP` 到纯英文路径） |
| DPAPI 解密失败（换机器/换用户） | 提示删除本地 `chat_sessions/`、`config.json` 重新生成配置（旧会话不可恢复） |
| 扫描件 PDF 无法 OCR | 检查 `bin/` 下 Poppler/Tesseract 及 `chi_sim.traineddata` 语言包 |
| 找不到 Python 或版本过低 | 安装 Python 3.10+ 并勾选 "Add python.exe to PATH" |
| Web 打不开 | 确认进程存活、端口监听在 127.0.0.1、防火墙未拦截 |

## 智能体介入原则

- 出错时**先读错误输出/日志再动手**，不盲目重装或全删
- 每次只做一步，执行后校验结果，再继续下一步
- 优先定位根因：网络 / 权限 / 依赖 / 编码 / 路径，而非粗暴重试
- 修改或删除用户文件前先备份，涉及 `config.json`、`chat_sessions/`、`knowledge_base/`、`chroma_db/` 等本地数据时**只提示、不擅自删除**
- 涉及敏感信息（API Key、密钥、内部文档名）不外泄、不写入日志或文件
- 完成后给出验证结论：页面可访问 + 一次问答/检索可用

## 验证清单

- [ ] `python --version` ≥ 3.10
- [ ] venv 创建并激活成功
- [ ] 依赖全部安装（`pip list` 关键包存在）
- [ ] 模型缓存校验通过（或用户确认跳过、接受降级）
- [ ] 配置页 API Key 测试连接通过
- [ ] `http://127.0.0.1:8501` 可访问
- [ ] 上传一个文档并完成一次带溯源的回答

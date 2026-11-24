# anon-proxy

一个放在 Codex CLI / Claude Code 和官方付费 API 之间的脱敏网关。下游用户的请求先进网关：身份标识被删掉或换成不可逆伪名，正文里的敏感字符串换成替身，然后用**你自己的商业 API key** 转发给 OpenAI / Anthropic。回复回来时再把替身换回原文（流式输出和工具调用参数里也会换），客户端照常工作。

纯 Python 标准库实现，Python 3.11+，不需要安装任何依赖。

## 上游还能看到什么

下面这些是用本机 codex 0.170.0 和 Claude Code 2.1.312 实际抓包对出来的。

| 客户端原本会发 | 经过网关后 |
|---|---|
| codex：请求头 `session-id`、`thread-id`、`x-client-request-id`、`x-codex-window-id`、`x-codex-turn-metadata`（含 `installation_id`）、`originator` | 不转发。请求头从零重建，只有统一 UA、`Content-Type`、你的 key |
| codex：请求体 `client_metadata`（常驻的 `installation_id`、会话和轮次 ID、沙箱模式、时间戳） | 删除 |
| codex：`prompt_cache_key`（等于会话 ID） | 换成按用户加盐的 HMAC 伪名。同一会话内保持不变，所以提示缓存照样命中 |
| codex：客户端自己生成的消息 ID（UUIDv7，带时间戳） | 换成确定性的伪 UUID；服务端生成的 ID 不动 |
| codex：`<timezone>Europe/Berlin</timezone>` | 改成配置里的值（默认 UTC） |
| Responses API 默认 `store=true` | 强制 `store=false` |
| CC：`metadata.user_id`（含 `device_id`、会话 ID） | 删除（可改成每个用户一个伪名，见 `end_user_id`） |
| CC：system 第一段 `x-anthropic-billing-header: cc_version=…; cc_entrypoint=…` | 默认删除（`strip_billing_header`；只有订阅附带额度时的影响见下文 “Credit balance too low” 一节） |
| CC：`X-Claude-Code-Session-Id`、`X-Stainless-OS/Arch/Runtime-Version`、`x-app` | 不转发（只保留功能必需的 `anthropic-version`、`anthropic-beta`） |
| 正文：`C:\Users\<名字>`、`/Users/<名字>`、`/home/<名字>`，以及学到的用户名在全文其他位置的出现 | 换成 `user1a2b3c4d` |
| 正文：API key、GitHub/AWS/Slack/HF token、JWT、私钥块、`PASSWORD=…` 这类赋值 | 换成 `SECRET_…` |
| 正文：邮箱、内外网 IP、`DESKTOP-XXXX` 主机名、`github.com/<组织>/<仓库>`、`literals` 里配置的字面量 | 换成对应替身 |

替身对每个下游用户是确定的（HMAC），同一个值在同一个用户那里永远是同一个替身；不同用户之间互不相同，上游没法跨用户关联。

## 快速开始

```powershell
copy config.example.toml config.toml
python -m anon_proxy keygen alice          # 输出一行 "gk-..." = "alice"，贴进 config.toml 的 [users]
$env:OPENAI_API_KEY = "你的 OpenAI API key"
$env:ANTHROPIC_API_KEY = "你的 Anthropic API key"
python -m anon_proxy serve -c config.toml
```

不想用环境变量，也可以把 key 放进一个文件，在配置里写 `api_key_file = "key.txt"`（相对配置文件所在目录）。网关每次请求都现读这个文件，换 key 不用重启；环境变量设了的话优先用环境变量。`key.txt`、`config.toml`、`data/`、`audit/` 都已写进 `.gitignore`。

要让别人连进来，把 `listen` 改成 `0.0.0.0:8787`，并在前面加一层 HTTPS 反向代理（Caddy / nginx）。网关 key 是明文放在请求头里的，直接走公网 HTTP 等于把 key 送人。

### 下游 Codex

`~/.codex/config.toml`：

```toml
model_provider = "anon"

[model_providers.anon]
name = "anon proxy"
base_url = "https://你的网关/v1"
env_key = "ANON_PROXY_KEY"
wire_api = "responses"
```

然后设置 `ANON_PROXY_KEY` 为分给他的 `gk-...`。自定义 provider 走 HTTP SSE，网关支持 `/v1/responses`（含 `/responses/compact`）、`/v1/chat/completions`。

### 下游 Claude Code

```powershell
$env:ANTHROPIC_BASE_URL = "https://你的网关"
$env:ANTHROPIC_API_KEY = "gk-..."
```

也可以写进 `~/.claude/settings.json` 的 `env`。网关支持 `/v1/messages` 和 `/v1/messages/count_tokens`。

## 和 Claude thinking 历史校验的兼容

Claude Opus 5.5、Fable 5.x 会校验回传的 thinking 块：它之前的 system、工具和消息必须和生成时逐字节一致，否则直接返回 400（2026-08-31 以后注册的账号默认强制，Fable 5.2 对所有账号强制）。网关为此做了两件事：

- **脱敏结果冻结**：一段文字第一次脱敏后，结果按原文缓存，之后每轮都原样复用。即使后面学到了新用户名，也不会回头改动早先的轮次。早先那段文字当时已经发出去了，再改它并不能补救隐私，只会让历史对不上。
- **`thinking_mismatch = "drop_block"`**（默认）：对 `thinking.type = "adaptive"` 的请求加上 `block_binding.prefix_mismatch_behavior` 和 `thinking-binding-controls-2026-08-01` beta。万一历史还是变了（比如网关重启后缓存清空、改了 `literals`），API 只丢弃受影响的 thinking 块继续回答，不会整个请求报错。丢了几块会出现在网关日志的 `thinking_dropped=` 里。设成 `"error"` 可以在测试时让问题直接暴露；设成 `"off"` 则不改动请求。

已在真实 API 上验证（Opus 5.5，Claude Code 经网关做一轮带 thinking 的工具调用）：把第二轮请求的 `prefix_mismatch_behavior` 改成 `"error"` 原样重放，返回 200，`input_transformations` 为空，提示缓存也照常命中。

HMAC 密钥（`data/secret.key`）要保持不变：换了它，所有替身都会变，进行中的会话历史也就全对不上了。

## Claude Code 报 “Credit balance too low”，但 key 明明有钱

Anthropic 会从请求本身认出 Claude Code：`claude-cli/…` 的 UA、`claude-code-20250219` beta、Claude Code 的系统提示词，三样里有任何一样就够（实测）。如果组织里只有 Max / Team / Ultra 订阅附带的每月 API 额度、没有自己买的余额，这类请求会被拒，普通 API 调用却照常能用。按 Anthropic 的说明，这笔额度不能用于交互式 Claude Code；`claude -p` 和 Agent SDK 只有在你自己用这个组织的 key 运行时才算。

- Anthropic 靠 Claude Code 写在 system 第一段的 `x-anthropic-billing-header` 认出 `claude -p`（里面写着 `cc_entrypoint=sdk-cli`）。网关默认删掉这一段，`claude -p` 也就被当成交互式 Claude Code 拒掉。设 `strip_billing_header = false` 会原样转发这一段，代价是上游能看到 Claude Code 版本号和启动方式。
- 交互式 Claude Code、以及给别人用的流量，请在 Console 的 Billing 页面另外购买额度。网关不会把 Claude Code 请求伪装成普通 API 调用来花这笔额度，也请不要这样配置它。
- `user_agent` 默认是 `claude-cli/…`，不是 Claude Code 的客户端经过网关也会被当成 Claude Code。只接这类客户端时，可以改成别的固定字符串。

## 自己核查

- 在 `[privacy]` 里设 `audit_dir = "audit"`，每个发往上游的请求（脱敏后、key 已隐去）都会存一份 JSON，上游看到的就是这些。
- `python -m anon_proxy mask -c config.toml "C:\Users\me\x 我的邮箱 a@corp.cn"` 可以看一段文字会被怎么替换。
- `python -m unittest discover -s tests` 跑全部测试。`tests/fake_upstream.py` 是一个假上游，会把收到的每个请求记进日志，可以把网关的上游地址指向它做实测。

## 做不到的事（请务必看）

- **模型必须看到内容才能干活。** 代码、文件名、项目路径、报错信息里没被规则命中的部分都会原样上去。真名、公司名、项目代号、内部域名请加进 `literals`。
- 规则是正则，不是 NER：人名不会自动识别；不像密钥的密码（比如纯字母）会漏掉。
- Anthropic 的 thinking 块带签名，必须原样回传，网关不改它。thinking 是模型基于已脱敏的输入生成的，本来就只含替身；但如果一段对话的前半截没经过网关，里面的原文会原样上去。
- 图片、PDF 等 base64 内容不处理。
- 操作系统和 shell 类型（`Windows`、`powershell`）留在提示词和工具描述里，模型要靠它生成正确的命令；客户端大版本也能从 `anthropic-beta` 列表和系统提示词的措辞看出来。
- 不处理时序和流量特征；上游看到的是网关的 IP。
- 客户端自己的遥测和统计上报不经过网关。Claude Code 要设 `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`；codex 的请求元数据里写着 `analytics_enabled: true`，需要在 codex 那边关掉它的统计上报（具体配置项以 codex 文档为准）。
- 上游照样会按自己的政策留存 API 数据。比如 OpenAI 默认保留用于滥用监控的日志，零数据保留（ZDR）要单独申请；组织设置里的数据共享选项也要自己确认是关闭的。以官网当前条款为准。
- `end_user_id = "none"` 时，任何一个下游用户违规，都会算到你的 key/组织头上。
- 模型偶尔会改写替身（大小写、拆开），这种情况还原不了。
- `force_store_false` 和依赖 `previous_response_id` 的客户端不兼容（Codex 本身不依赖）。

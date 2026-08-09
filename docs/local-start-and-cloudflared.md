# 本地启动与 Cloudflare Tunnel

本文档说明如何在本地启动 EvoAgent，并通过 `cloudflared` 将本地的
`127.0.0.1:8080` 暴露为临时 HTTPS 地址。

## 前置条件

- Python 3.11
- 已安装 `cloudflared`，并可在 PowerShell 中执行 `cloudflared --version`
- 可选：Docker Desktop（使用 Compose 启动时需要）

## 方式一：直接运行项目

在项目根目录打开 PowerShell，首次运行时安装依赖：

```powershell
python -m pip install -r requirements.txt
```

若项目根目录还没有 `.env`，复制示例配置并按需修改：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

启动服务：

```powershell
python -m evoagent
```

服务默认监听 `http://127.0.0.1:8080`。另开一个 PowerShell 窗口验证：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

## 方式二：使用 Docker Compose

在项目根目录执行：

```powershell
docker compose up --build
```

该方式会启动 EvoAgent、PostgreSQL 和 RocketMQ，EvoAgent 仍通过本机的
`http://127.0.0.1:8080` 访问。后台运行可使用：

```powershell
docker compose up --build -d
```

查看日志或停止服务：

```powershell
docker compose logs -f evoagent
docker compose down
```

## 启动 Cloudflare Tunnel

确认 EvoAgent 已在本机 `127.0.0.1:8080` 运行后，另开一个 PowerShell
窗口执行：

```powershell
cloudflared tunnel --url http://127.0.0.1:8080
```

命令输出会包含类似下面的公网 HTTPS 地址：

```text
https://example.trycloudflare.com
```

保持 EvoAgent 和 `cloudflared` 两个终端窗口均处于运行状态。每次重启
`cloudflared` 后，临时 `trycloudflare.com` 地址可能改变；如果用于 GitHub
Webhook，请将 Payload URL 更新为：

```text
https://example.trycloudflare.com/webhooks/github
```

## 安全注意事项

Tunnel 会将管理台和 API 一同暴露到公网。暴露前应在 `.env` 中启用认证，并
设置足够强的随机密钥和管理员密码：

```env
EVOAGENT_AUTH_REQUIRED=true
EVOAGENT_AUTH_SECRET=<至少 32 字节的随机密钥>
EVOAGENT_BOOTSTRAP_ADMIN_USERNAME=admin
EVOAGENT_BOOTSTRAP_ADMIN_PASSWORD=<强密码>
```

不要将 `.env`、管理员密码、API Key 或 Webhook 密钥提交到仓库。长期部署时，
建议通过反向代理只对公网开放 `/webhooks/github`（以及按需开放 `/health`），
不要直接暴露整个管理台。

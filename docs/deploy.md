# 部署指南

本文档说明量化潜伏系统的多种部署方式，按推荐程度排序。

## 部署方式对比

| 方式 | 成本 | 稳定性 | 运维 | 适用场景 |
|------|------|--------|------|---------|
| GitHub Actions | 免费 | 中（受平台调度影响） | 零运维 | 个人使用、MVP |
| Docker + VPS | 低（~$5/月） | 高 | 中 | 订阅服务、长期运行 |
| 腾讯云函数 | 按量计费 | 高 | 低 | 微信回调、推送网关 |
| docker-compose 全栈 | 低 | 高 | 中 | 一体化部署 |

---

## 方式一：GitHub Actions（零成本）

详见 [README.md](README.md#快速开始)。适合个人使用，无需服务器。

**限制**：
- 60 天不活动会自动禁用（已通过 `keepalive.yml` 解决）
- 定时触发可能延迟 5-30 分钟（已通过错峰触发缓解）
- 公开仓库免费 2000 分钟/月，私有仓库有限额

---

## 方式二：Docker + VPS（推荐订阅服务）

### 1. 准备 VPS

推荐配置：
- 2 核 CPU / 2GB 内存 / 20GB 磁盘
- Ubuntu 22.04 LTS
- 国内云厂商（腾讯云/阿里云）网络更稳定

### 2. 安装 Docker

```bash
# Ubuntu
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# 重新登录使 docker 组生效
```

### 3. 部署系统

```bash
# 克隆仓库
git clone https://github.com/Lishoulan/lianghua.git
cd lianghua

# 配置环境变量
cp .env.example .env
vim .env  # 填入 API Keys

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f scanner
docker-compose logs -f cloud-func
```

### 4. 配置定时任务

`docker-compose.yml` 已内置 cron 调度：
- 盘中扫描：工作日 13:00 北京时间
- 盘后扫描：工作日 21:00 北京时间
- keepalive：每周一 03:00

如需调整，编辑 `docker-compose.yml` 中 `scanner` 服务的 cron 表达式。

### 5. 配置 Nginx 反向代理（可选）

```nginx
server {
    listen 80;
    server_name push.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

配置 SSL：
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d push.yourdomain.com
```

### 6. 配置微信公众号回调

在微信公众号后台 → 开发 → 基本配置：
- 服务器URL：`https://push.yourdomain.com/wechat`
- Token：与 `.env` 中 `WECHAT_TOKEN` 一致
- EncodingAESKey：与 `.env` 中 `WECHAT_ENCODING_AES_KEY` 一致

---

## 方式三：腾讯云函数（推送网关）

适合只需微信回调 + 推送网关的场景，无需 7×24 运行服务器。

### 1. 部署云函数

```bash
# 安装 Serverless CLI
npm install -g serverless

# 在项目根目录创建 serverless.yml（参考腾讯云文档）
# 上传 wechat_push/ 目录
```

### 2. 配置 API 网关

在腾讯云控制台 → API 网关：
- 创建 API 并绑定云函数
- 路由配置：
  - `GET  /wechat` → `wechat_handler`
  - `POST /wechat` → `wechat_handler`
  - `POST /push`   → `push_handler`
  - `GET  /health` → `health_handler`
  - `GET  /stats`  → `stats_handler`

### 3. 配置环境变量

在云函数环境变量中设置：
```
WECHAT_APP_ID=...
WECHAT_APP_SECRET=...
WECHAT_TOKEN=...
WECHAT_ENCODING_AES_KEY=...
SUPABASE_URL=...
SUPABASE_KEY=...
PUSH_API_KEY=...
```

---

## 方式四：Supabase 数据库配置

### 1. 创建 Supabase 项目

访问 [supabase.com](https://supabase.com) 创建项目。

### 2. 执行 Schema

进入 SQL Editor，粘贴并执行 [docs/schema.sql](schema.sql)。

### 3. 获取 API Keys

在 Settings → API：
- `Project URL` → 填入 `.env` 的 `SUPABASE_URL`
- `service_role` secret → 填入 `.env` 的 `SUPABASE_KEY`（**不要用 anon key**）

### 4. 配置 Row Level Security

`schema.sql` 已配置 RLS 策略：
- `service_role`：完全访问（云函数使用）
- `anon`：无权限（不直接暴露给前端）

---

## 监控与告警

### 健康检查

部署后，配置外部监控服务定期检查：

```
GET https://push.yourdomain.com/health
```

返回 `200` 表示健康，`503` 表示降级，`500` 表示故障。

推荐监控服务：
- [UptimeRobot](https://uptimerobot.com)（免费 50 个监控）
- 腾讯云监控 / 阿里云云监控

### 运营统计

```
GET https://push.yourdomain.com/stats?days=7
Authorization: Bearer <PUSH_API_KEY>
```

返回推送次数、成功率、信号数量、订阅用户分布。

### GitHub Actions 告警

`daily_push.yml` 已配置自动告警：
- 推送失败 → 自动创建 Issue（标签 `push-failure`）
- 推送恢复 → 自动关闭告警 Issue

---

## 数据备份

### DuckDB 缓存备份

```bash
# 定时备份（crontab）
0 3 * * * docker cp tradingagents-scanner:/app/results/stock_cache.duckdb /backup/stock_cache_$(date +\%Y\%m\%d).duckdb
```

### Supabase 备份

Supabase 免费版提供每日自动备份（保留 7 天）。Pro 版可配置更长的保留期。

---

## 升级

```bash
cd lianghua
git pull origin main
docker-compose build
docker-compose up -d
```

## 故障排查

| 症状 | 排查 |
|------|------|
| 推送未到达 | 检查 `docker logs`、Server酱 Key、微信公众号 access_token |
| 数据拉取失败 | 检查 TUSHARE_TOKEN、网络连通性、降级到 mootdx |
| 云函数 502 | 检查 API 网关配置、云函数日志、超时设置 |
| Supabase 写入失败 | 检查 service_role key、RLS 策略、表结构 |
| DuckDB 锁定 | 检查是否有并发写入，重启 scanner 容器 |

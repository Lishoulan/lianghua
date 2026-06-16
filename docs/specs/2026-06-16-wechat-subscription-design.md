# 微信公众号订阅分销方案设计

## 概述

将量化潜伏信号通过微信公众号（订阅号）群发推文推送给订阅用户，实现零成本订阅分销。

## 架构

```
GitHub Actions (扫描计算层)
  12:55 盘中扫描 → Server酱推送管理员组
  20:55 盘后扫描 → Server酱推送 + POST信号到腾讯云函数 → 公众号群发
       │
       ▼
腾讯云函数 (推送网关)
  wechat_handler: 微信事件回调(关注/取关/消息)
  push_handler:   接收信号 → 生成图文HTML → 上传素材 → 群发推文
       │
  Supabase: 用户表(open_id, 关注状态, 订阅状态)
       │
  ├── 微信公众号 (图文群发给所有关注者)
  └── Server酱 (管理员组完整技术版)
```

## 推送策略

| 时段 | 管理员组(Server酱) | 公众号订阅者 |
|------|-------------------|-------------|
| 14:00盘中 | ✅ 实时推送 | ❌ 无 |
| 22:00盘后 | ✅ 完整技术版 | ✅ 精简图文版 |

## 微信API流程

1. 获取access_token (GET https://api.weixin.qq.com/cgi-bin/token)
2. 上传图文素材 (POST https://api.weixin.qq.com/cgi-bin/material/add_news)
3. 群发消息 (POST https://api.weixin.qq.com/cgi-bin/message/mass/sendall)

## 数据模型 (Supabase)

```sql
users (
  open_id TEXT PRIMARY KEY,
  nickname TEXT,
  subscribe_at TIMESTAMPTZ,
  is_subscribed BOOLEAN DEFAULT true,
  plan TEXT DEFAULT 'trial',  -- trial/free/monthly/quarterly/yearly
  plan_expire TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
)

push_logs (
  id SERIAL PRIMARY KEY,
  push_time TIMESTAMPTZ,
  mode TEXT,
  signal_count INT,
  signals_json JSONB,
  wechat_mass_id TEXT,
  wechat_success INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
)
```

## 变现路径

- 阶段1: 群发完整内容，积累用户
- 阶段2: 群发摘要+详情页订阅墙
- 阶段3: 升级认证服务号+微信支付

## 环境变量

- WECHAT_APP_ID: 公众号AppID
- WECHAT_APP_SECRET: 公众号AppSecret
- WECHAT_TOKEN: 服务器配置Token(自定义)
- WECHAT_ENCODING_AES_KEY: 消息加解密Key(自定义)
- SUPABASE_URL: Supabase项目URL
- SUPABASE_KEY: Supabase API Key
- PUSH_API_KEY: GitHub Actions调用推送接口的认证Key(自定义)

## 公众号配置

- AppID: wxf7a248020d89a85e
- 服务器URL: 腾讯云函数API网关地址
- Token: 自定义，用于验证微信签名
- EncodingAESKey: 自定义，用于消息加解密

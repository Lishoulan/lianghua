# 安全策略

## 报告安全漏洞

如果你发现安全漏洞，请**不要**通过公开 Issue 报告。

请通过以下方式私密报告：

1. 发送邮件至项目维护者（通过 GitHub 个人主页邮箱）
2. 或使用 GitHub Security Advisory：**Security** 标签页 → **Report a vulnerability**

请在报告中包含：
- 漏洞类型（如 SQL 注入、敏感信息泄露、认证绕过）
- 受影响的文件/模块
- 复现步骤
- 影响范围
- 建议的修复方案（如有）

## 响应时间

| 阶段 | 预期时间 |
|------|---------|
| 确认收到报告 | 48 小时内 |
| 初步评估 | 5 个工作日内 |
| 修复发布 | 视严重程度 7-30 天 |

## 支持的版本

仅对最新发布版本（main 分支）提供安全更新。

## 安全注意事项

本项目涉及以下敏感信息，使用时请注意：

| 敏感信息 | 存储位置 | 保护措施 |
|---------|---------|---------|
| `TUSHARE_TOKEN` | GitHub Secrets / `.env` | 仅通过环境变量注入，不写入代码 |
| `SERVERCHAN_KEY` | GitHub Secrets / `.env` | 同上 |
| `WECHAT_APP_SECRET` | GitHub Secrets / `.env` | 同上 |
| `SUPABASE_KEY` | GitHub Secrets / `.env` | 使用 Service Role Key，不暴露到前端 |
| `PUSH_API_KEY` | GitHub Secrets / `.env` | 推送接口 Bearer 认证 |
| `GITHUB_TOKEN` | `.env`（仅本地） | 仅 `workflow` 权限，不提交到仓库 |

### 最佳实践

1. **永远不要**将 `.env` 文件提交到 Git
2. **永远不要**在代码中硬编码 API Key
3. GitHub Actions 中使用 `secrets.*` 引用敏感信息
4. 腾讯云函数部署时使用环境变量配置
5. Supabase 使用 Row Level Security (RLS) 限制数据访问
6. 定期轮换 API Key

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。量化交易存在风险，使用者需自行承担一切后果。

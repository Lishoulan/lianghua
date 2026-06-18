# 量化潜伏系统 — 贡献指南

感谢你对量化潜伏系统的关注！本文档说明如何参与本项目开发。

## 行为准则

参与本项目即代表你同意遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。请在所有交流中保持尊重和包容。

## 如何贡献

### 报告 Bug

1. 在 [Issues](../../issues) 页面搜索是否已有相同问题
2. 若无，使用 [Bug 报告模板](.github/ISSUE_TEMPLATE/bug_report.md) 创建新 Issue
3. 请包含：复现步骤、预期行为、实际行为、环境信息、错误日志

### 提交功能建议

1. 使用 [功能建议模板](.github/ISSUE_TEMPLATE/feature_request.md) 创建 Issue
2. 说明使用场景、期望行为、替代方案

### 提交代码

#### 开发流程

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/<你的用户名>/lianghua.git
cd lianghua

# 2. 创建功能分支
git checkout -b feat/your-feature-name

# 3. 安装开发依赖
pip install -r requirements.txt
pip install pytest

# 4. 编写代码 + 测试
#    策略代码放 classic_ta/，公共逻辑放 common/
#    新增测试到 tests/

# 5. 运行测试
python -m pytest tests/ -v

# 6. 提交（遵循约定式提交）
git add <files>
git commit -m "feat: 简要描述"

# 7. 推送并发起 PR
git push origin feat/your-feature-name
```

#### 提交信息规范

遵循 [约定式提交](https://www.conventionalcommits.org/zh-hans/)：

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feat:` | 新功能 | `feat: 新增 V6.5 行业轮动加速因子` |
| `fix:` | Bug 修复 | `fix: 修复 DuckDB 并发写入死锁` |
| `refactor:` | 重构 | `refactor: 合并 v63/v64 推送入口` |
| `perf:` | 性能优化 | `perf: 预筛选批量并发优化` |
| `docs:` | 文档 | `docs: 更新部署指南` |
| `test:` | 测试 | `test: 新增 OAMV 择时测试` |
| `style:` | 格式 | `style: 统一 import 排序` |
| `chore:` | 杂项 | `chore: 升级依赖版本` |

#### 代码风格

- Python 3.10+
- 缩进 4 空格，文件 UTF-8 无 BOM
- 模块级常量全大写下划线（`BEST_PARAMS`）
- 函数/方法小写下划线（`compute_industry_momentum`）
- 类名驼峰（`SyncScanner`）
- 每个模块顶部需有中文 docstring 说明用途
- 数据源调用需有降级机制（akshare → tushare → mootdx）
- 新增配置项需同步更新 `.env.example`

#### PR 检查清单

提交 PR 前请确认：

- [ ] `python -m pytest tests/ -v` 全部通过
- [ ] 未提交 `.env`、API Keys 等敏感信息
- [ ] 如有新依赖，已更新 `requirements.txt`
- [ ] 如有配置变更，已更新 `.env.example`
- [ ] 代码风格与项目一致
- [ ] PR 描述清晰，关联相关 Issue

## 策略版本规范

策略版本号遵循 `V<major>.<minor>`：

- **major**: 架构级变更（如 V6 → V7）
- **minor**: 维度叠加（如 V6.3 → V6.4 新增评分因子）

每个新版本必须：
1. 继承上一版本全部能力
2. 在 `classic_ta/v<version>_ambush_model.py` 实现
3. 通过回测验证（5年全市场）
4. 更新 `CHANGELOG.md`

## 项目结构约定

```
classic_ta/          # 策略实现（每个版本一个文件）
  common/            # 跨版本共享逻辑
  v6X_ambush_model.py
ml_strategy/         # ML 择时模块
wechat_push/         # 微信公众号推送
tests/               # 单元测试
docs/specs/          # 设计文档（日期前缀）
.github/workflows/   # CI/CD
```

## 许可证

提交的代码将遵循 [MIT License](LICENSE)。

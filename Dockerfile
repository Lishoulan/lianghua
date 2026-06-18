# ══════════════════════════════════════════════════════════
# 量化潜伏系统 — Dockerfile
# 支持本地运行、VPS 部署、腾讯云容器服务
# ══════════════════════════════════════════════════════════

FROM python:3.11-slim AS base

# 设置时区为北京时间（影响交易时间判断）
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（akshare/mootdx 需要的编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── 依赖层（利用 Docker 层缓存）──
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── 应用层 ──
COPY . .

# 创建数据目录
RUN mkdir -p /app/results/daily /app/results/oamv_cache

# 健康检查（如果部署了云函数端点，可改为 HTTP 检查）
HEALTHCHECK --interval=5m --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from classic_ta.stock_data_duckdb import get_cache_stats; s=get_cache_stats(); print(f'OK: {s.get(\"count\",0)} stocks cached')" || exit 1

# 默认入口：运行每日推送
CMD ["python", "-u", "classic_ta/daily_push.py"]

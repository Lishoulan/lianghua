-- ══════════════════════════════════════════════════════════
-- 量化潜伏系统 — Supabase 数据库 Schema
-- ══════════════════════════════════════════════════════════
-- 用法：在 Supabase SQL Editor 中执行本文件
-- ══════════════════════════════════════════════════════════

-- ── 扩展 ──
create extension if not exists "uuid-ossp";

-- ══════════════════════════════════════════════════════════
-- 1. 用户表
-- ══════════════════════════════════════════════════════════
create table if not exists public.users (
    open_id          text primary key,
    nickname         text default '',
    avatar_url       text default '',
    is_subscribed    boolean default true,
    plan             text default 'trial' check (plan in ('trial','free','monthly','quarterly','yearly','expired')),
    trial_start      timestamptz,
    trial_end        timestamptz,
    plan_start       timestamptz,
    plan_expire      timestamptz,
    subscribe_at     timestamptz default now(),
    unsubscribe_at   timestamptz,
    last_active_at   timestamptz default now(),
    created_at       timestamptz default now(),
    updated_at       timestamptz default now()
);

comment on table public.users is '微信公众号订阅用户';
comment on column public.users.plan is 'trial=试用 free=免费 monthly=月度 quarterly=季度 yearly=年度 expired=已过期';

-- ══════════════════════════════════════════════════════════
-- 2. 推送日志表
-- ══════════════════════════════════════════════════════════
create table if not exists public.push_logs (
    id               serial primary key,
    push_time        timestamptz default now(),
    mode             text check (mode in ('intraday','after_hours')),
    oamv_status      jsonb,
    signal_count     integer default 0,
    signals_json     jsonb,
    industry_stats   jsonb,
    wechat_mass_id   text,
    wechat_success   integer default 0,
    wechat_error     text,
    duration_seconds real,
    created_at       timestamptz default now()
);

create index if not exists idx_push_logs_time on public.push_logs(push_time desc);
create index if not exists idx_push_logs_mode on public.push_logs(mode);

-- ══════════════════════════════════════════════════════════
-- 3. 订阅事件表（审计日志）
-- ══════════════════════════════════════════════════════════
create table if not exists public.subscription_events (
    id               serial primary key,
    open_id          text not null references public.users(open_id) on delete cascade,
    event_type       text not null check (event_type in ('subscribe','unsubscribe','trial_start','trial_end','plan_upgrade','plan_expire','plan_renew')),
    from_plan        text,
    to_plan          text,
    amount           numeric(10,2),
    note             text,
    created_at       timestamptz default now()
);

create index if not exists idx_sub_events_openid on public.subscription_events(open_id);
create index if not exists idx_sub_events_type on public.subscription_events(event_type);

-- ══════════════════════════════════════════════════════════
-- 4. 系统监控指标表
-- ══════════════════════════════════════════════════════════
create table if not exists public.metrics (
    id               serial primary key,
    metric_name      text not null,
    metric_value     numeric not null,
    metric_tags      jsonb default '{}'::jsonb,
    recorded_at      timestamptz default now()
);

create index if not exists idx_metrics_name_time on public.metrics(metric_name, recorded_at desc);

-- ══════════════════════════════════════════════════════════
-- 5. 更新时间触发器
-- ══════════════════════════════════════════════════════════
create or replace function public.handle_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_users_updated on public.users;
create trigger trg_users_updated
    before update on public.users
    for each row execute function public.handle_updated_at();

-- ══════════════════════════════════════════════════════════
-- 6. 自动过期订阅函数
-- ══════════════════════════════════════════════════════════
create or replace function public.expire_subscriptions()
returns integer
language plpgsql
as $$
declare
    expired_count integer;
begin
    update public.users
    set plan = 'expired'
    where plan in ('trial','monthly','quarterly','yearly')
      and plan_expire is not null
      and plan_expire < now()
      and plan != 'expired';

    get diagnostics expired_count = row_count;

    -- 记录过期事件
    insert into public.subscription_events (open_id, event_type, from_plan, to_plan, note)
    select open_id, 'plan_expire', plan, 'expired', '自动过期'
    from public.users
    where plan = 'expired'
      and plan_expire < now()
      and updated_at > now() - interval '1 minute';

    return expired_count;
end;
$$;

-- ══════════════════════════════════════════════════════════
-- 7. 获取有效订阅用户数函数
-- ══════════════════════════════════════════════════════════
create or replace function public.get_active_subscriber_count()
returns integer
language sql
as $$
    select count(*)::integer
    from public.users
    where is_subscribed = true
      and plan in ('trial','monthly','quarterly','yearly')
      and (plan_expire is null or plan_expire > now());
$$;

-- ══════════════════════════════════════════════════════════
-- 8. Row Level Security (RLS)
-- ══════════════════════════════════════════════════════════
alter table public.users enable row level security;
alter table public.push_logs enable row level security;
alter table public.subscription_events enable row level security;
alter table public.metrics enable row level security;

-- 仅 service_role 可读写所有表（云函数使用 service_role key）
-- anon 角色无任何权限（微信用户不直接访问 Supabase）
create policy "service_role full access users"
    on public.users for all
    using (auth.role() = 'service_role')
    with check (auth.role() = 'service_role');

create policy "service_role full access push_logs"
    on public.push_logs for all
    using (auth.role() = 'service_role')
    with check (auth.role() = 'service_role');

create policy "service_role full access subscription_events"
    on public.subscription_events for all
    using (auth.role() = 'service_role')
    with check (auth.role() = 'service_role');

create policy "service_role full access metrics"
    on public.metrics for all
    using (auth.role() = 'service_role')
    with check (auth.role() = 'service_role');

-- ══════════════════════════════════════════════════════════
-- 9. 初始数据（可选）
-- ══════════════════════════════════════════════════════════
-- 插入示例监控指标（演示用，可删除）
insert into public.metrics (metric_name, metric_value, metric_tags)
values
    ('system.init', 1, '{"version": "6.4.0"}'::jsonb)
on conflict do nothing;

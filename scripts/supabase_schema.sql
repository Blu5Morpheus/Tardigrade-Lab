-- ────────────────────────────────────────────────────────────────────
-- Tardigrade Lab — Supabase schema
-- Run this in the Supabase SQL editor for your project. One-time setup.
-- ────────────────────────────────────────────────────────────────────

-- Per-demo configuration (admin → Demos tab).
create table if not exists demo_settings (
  slug          text primary key,
  enabled       boolean not null default true,
  display_order integer not null default 0,
  default_params jsonb not null default '{}'::jsonb,
  notes         text,
  updated_at    timestamptz not null default now()
);

-- Hero "// LAB STATUS" rows (admin → Lab status tab).
create table if not exists lab_status (
  id            uuid primary key default gen_random_uuid(),
  display_order integer not null,
  label         text not null,
  value         text not null,
  kind          text not null check (kind in ('ok', 'warn', 'default')),
  active        boolean not null default true,
  updated_at    timestamptz not null default now()
);

-- Orders ledger (post-shop-launch; schema ready in advance).
create table if not exists orders (
  stripe_payment_intent_id text primary key,
  customer_email text not null,
  product_slug   text not null,
  amount_cents   integer not null,
  status         text not null check (status in ('paid', 'shipped', 'delivered', 'refunded')),
  shipped_at     timestamptz,
  tracking_number text,
  notes          text,
  created_at     timestamptz not null default now()
);

-- ────────────────────────────────────────────────────────────────────
-- Seed data
-- ────────────────────────────────────────────────────────────────────

insert into demo_settings (slug, display_order, enabled) values
  ('vqe',           1, true),
  ('clifford',      2, true),
  ('amplituhedron', 3, true),
  ('lattice',       4, true),
  ('page-curve',    5, true),
  ('me-bot',        6, true)
on conflict (slug) do nothing;

insert into lab_status (display_order, label, value, kind, active) values
  (1, 'VQE / LIGO classifier',      'PREPRINT PENDING', 'ok',   true),
  (2, 'Clifford GA agent (Cl 3,1)', 'RUNNING',          'ok',   true),
  (3, 'NV-center MPCVD reactor',    'CALIBRATION',      'warn', true),
  (4, 'L-band interferometer',      'OPERATIONAL',      'ok',   true),
  (5, 'Yb⁺ Paul trap',              'ASSEMBLY',         'warn', true);

-- ────────────────────────────────────────────────────────────────────
-- Row-Level Security
-- ────────────────────────────────────────────────────────────────────

alter table demo_settings enable row level security;
alter table lab_status    enable row level security;
alter table orders        enable row level security;

-- Public read for the two surfaces the Astro build needs to fetch.
drop policy if exists "public read demo_settings" on demo_settings;
create policy "public read demo_settings" on demo_settings
  for select using (true);

drop policy if exists "public read lab_status" on lab_status;
create policy "public read lab_status" on lab_status
  for select using (active = true);

-- Orders are admin-only; service-role key bypasses RLS for writes.
-- (No public read policy on orders.)

-- Touch updated_at on every update.
create or replace function touch_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists demo_settings_touch on demo_settings;
create trigger demo_settings_touch before update on demo_settings
  for each row execute function touch_updated_at();

drop trigger if exists lab_status_touch on lab_status;
create trigger lab_status_touch before update on lab_status
  for each row execute function touch_updated_at();

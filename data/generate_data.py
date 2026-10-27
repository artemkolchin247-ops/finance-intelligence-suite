"""Builds data/finance_db.xlsx: the Excel stand-in for the PostgreSQL warehouse.

Sheets mirror warehouse tables and use the same raw field names as the reporting
logic (orders_count, orders_rub, sales_qty, reclama, ads_vk_c_rub, ...):
  dim_sku, fact_daily, fact_ext_daily, stocks_ms, creators_weekly, hypotheses.
Only raw values are stored; every ratio is derived in core/.
Run: python data/generate_data.py   (seeded, deterministic)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
START = pd.Timestamp("2026-03-01")
END = pd.Timestamp("2026-06-08")
PRE_DAYS = 12  # simulated before START so lagged buyouts exist on day one
OUT_PATH = Path(__file__).with_name("finance_db.xlsx")
MARKETPLACES = ["WB", "Ozon"]

SKU_TEXT = """
компбел-ж-бесшов/беж компбел-ж-бесшов/голуб компбел-ж-бесшов/корал компбел-ж-бесшов/розов компбел-ж-бесшов/рубин
компбел-ж-бесшов/свроз компбел-ж-бесшов/серый компбел-ж-бесшов/темкор компбел-ж-бесшов/темсин компбел-ж-бесшов/чер
компбел-ж-бесшов/affogato компбел-ж-бесшов/bright_yellow компбел-ж-бесшов/dark_gray компбел-ж-бесшов/dark_green
компбел-ж-бесшов/emerald_green компбел-ж-бесшов/ivory компбел-ж-бесшов/lavender компбел-ж-бесшов/light_brown
компбел-ж-бесшов/lime компбел-ж-бесшов/military_green компбел-ж-бесшов/white компбел-ж-бесшов/wine_red
alice/black alice/brown alice/nude alice/white alice/wine_red
audrey/americano audrey/black audrey/brown audrey/dark_beige audrey/fig audrey/graphite audrey/ivory audrey/light_brown
audrey/light_green audrey/light_pink audrey/nude audrey/total_white audrey/white audrey/wine_red
bella/black bella/brown bella/light_beige bella/white
charlotte/black charlotte/brown charlotte/dark_beige
eva/black eva/light_beige eva/white
joy/affogato joy/black joy/brown joy/brownie joy/dark_gray joy/dark_pink joy/dark_purple joy/ivory joy/lavender
joy/light_blue joy/light_purple joy/military_green joy/navy_blue joy/nude joy/pale_pink joy/peach joy/red
joy/shinny_pink joy/silver_grey joy/sky_blue joy/washed_blue joy/washed_khaki joy/white joy/wine_red
lana/black lana/dark_beige lana/light_beige lana/white lana/yellow
miafull/black miafull/dark_beige miafull/light_beige miafull/white
moon/affogato moon/black moon/bright_yellow moon/brown moon/dark_gray moon/dark_purple moon/emerald_green moon/ivory
moon/lavender moon/light_blue moon/military_green moon/nude moon/pale_green moon/pale_pink moon/peach moon/red
moon/shinny_pink moon/silver_grey moon/washed_black moon/washed_denim moon/washed_khaki moon/white moon/wine_red
moon2/affogato moon2/black moon2/bright_yellow moon2/brown moon2/brownie moon2/ivory moon2/light_brown moon2/nude
moon2/red moon2/washed_black moon2/washed_denim moon2/white moon2/wine_red
ruby/affogato ruby/black ruby/bright_yellow ruby/brown ruby/brownie ruby/dark_gray ruby/dark_purple ruby/emerald_green
ruby/flower2 ruby/heart_gray ruby/heart_pink ruby/ivory ruby/light_blue ruby/light_brown ruby/military_green
ruby/navy_blue ruby/nude ruby/pale_pink ruby/peach ruby/red ruby/shinny_pink ruby/shock_green ruby/silver_grey
ruby/total_love_red ruby/washed_barbie ruby/washed_black ruby/washed_blue ruby/washed_candy ruby/washed_denim
ruby/washed_khaki ruby/white ruby/wine_red
set_bella/black set_bella/black_light_beige_white set_bella/white
set_moon/black_white_nude set_moon/magic_red set_moon/nude set_moon/spring set_moon/white
set_moon2/black set_moon2/black_white_nude set_moon2/nude set_moon2/skin set_moon2/white
set_ruby/black set_ruby/skin set_ruby/spring set_ruby/white set_ruby/white_black_nude
set_vuki/black_white_nude set_vuki/magic_red set_vuki/total_love_red set_vuki/white
set_vuki2/calm set_vuki2/fall set_vuki2/skin
set_wendy/black_nude_white
set_wookiee/black_white_nude set_wookiee/nude
valery/black valery/brown valery/ivory valery/light_pink valery/nude valery/white
vuki/heart_gray vuki/heart_pink vuki/washed_barbie vuki/washed_black vuki/washed_brown vuki/washed_denim vuki/washed_khaki
vuki2/affogato vuki2/black vuki2/brown vuki2/dark_grey vuki2/ivory vuki2/leo_brown vuki2/light_blue vuki2/nude
vuki2/red vuki2/washed_black vuki2/washed_denim vuki2/white vuki2/wine_red
wendy/americano wendy/black wendy/blue wendy/brown wendy/coffee wendy/dark_beige wendy/fig wendy/graphite
wendy/light_beige wendy/light_brown wendy/light_grey wendy/light_pink wendy/nude wendy/pink wendy/white wendy/yellow
"""
# SKU articles whose model key differs from the model name
ARTICLE_MODEL = {
    "компбел-ж-бесшов": "Vuki", "vuki": "Vuki", "vuki2": "Vuki2", "set_vuki2": "Set Vuki2",
    "set_ruby": "set ruby",
}
# models in the report list that have no articles in the supplied SKU list
EXTRA_MODELS = ["JoyW", "MoonW", "MoonW2", "RubyP", "RubyPT", "RubyW", "Set MoonP", "Set VukiP",
                "VukiN2", "VukiP", "VukiW", "VukiW2"]
EXTRA_COLORS = ["black", "white", "nude", "brown", "ivory", "light_beige"]
DISPLAY = {m.lower().replace(" ", "_"): m for m in [
    "Alice", "Audrey", "Bella", "Charlotte", "Eva", "Joy", "Lana", "Miafull", "Moon", "Moon2", "Ruby",
    "Set Bella", "Set Moon", "Set Moon2", "Set Vuki", "Set Wendy", "Set Wookiee", "Valery", "Wendy",
]}
OUTGOING_RUBY = {"washed_barbie", "washed_candy", "washed_khaki", "total_love_red", "flower2", "heart_gray", "heart_pink"}

# orders/day on WB for the whole model, list price before SPP (rub), sebes share of price
MODEL_CFG = {
    "Vuki": (45, 1500, 0.220), "Moon": (10, 1500, 0.220), "Wendy": (60, 1700, 0.295), "Ruby": (70, 1600, 0.295),
    "Joy": (42, 1550, 0.295), "Audrey": (35, 1750, 0.285), "Lana": (22, 1400, 0.295), "Moon2": (20, 1600, 0.295),
    "Vuki2": (25, 1650, 0.295), "Alice": (14, 1500, 0.305), "Bella": (15, 1450, 0.295), "Charlotte": (10, 1800, 0.285),
    "Eva": (9, 1300, 0.295), "Miafull": (12, 1900, 0.275), "Valery": (15, 1750, 0.285), "Set Bella": (5, 2900, 0.275),
    "Set Moon": (7, 3000, 0.275), "Set Moon2": (5, 3100, 0.275), "set ruby": (6, 3200, 0.275), "Set Vuki": (6, 3000, 0.275),
    "Set Vuki2": (4, 3100, 0.275), "Set Wendy": (2.5, 3300, 0.275), "Set Wookiee": (3, 3200, 0.275),
}
EXT_CHANNELS = {  # prob active, drr of orders_rub (lo,hi), cpc (lo,hi), ctr (lo,hi), click->order CR (lo,hi)
    "b": (0.35, 0.020, 0.050, 15, 35, 0.008, 0.02, 0.02, 0.06),
    "ads_vk_c": (0.60, 0.004, 0.010, 8, 14, 0.008, 0.02, 0.02, 0.06),
    "ads_vk_s": (0.40, 0.005, 0.012, 12, 20, 0.006, 0.015, 0.02, 0.05),
    "ads_s_c": (0.40, 0.005, 0.012, 12, 20, 0.006, 0.015, 0.02, 0.05),
    "ads_ya_c": (0.70, 0.010, 0.020, 10, 18, 0.01, 0.025, 0.03, 0.07),
}

# Hypothesis effects baked into the data: (mp, model, from, to, only_status, effects)
# effects: price, orders, cr (click->order), clicks, buyout, int_cpc (abs, rub), int_drr (share of sales, abs),
#          ext (spend/views multiplier), ext_cr, ext_cpc (abs)
EVENTS = [
    ("WB", "Wendy", "2026-04-30", "2026-06-08", None, dict(int_drr=0.045, int_cpc=9.9)),
    ("WB", "Audrey", "2026-04-30", "2026-06-08", None, dict(cr=1.08)),
    ("WB", "Joy", "2026-04-30", "2026-06-08", None, dict(ext=2.4, ext_cr=0.45, orders=0.93)),
    ("WB", "Joy", "2026-05-07", "2026-06-08", None, dict(clicks=1.12)),
    ("WB", "Lana", "2026-04-30", "2026-06-08", None, dict(ext=1.9, ext_cpc=2.6)),
    ("WB", "Vuki", "2026-05-07", "2026-06-08", None, dict(price=0.80, orders=2.00, buyout=1.12, int_drr=0.035, ext=0.5)),
    ("WB", "Moon", "2026-05-07", "2026-05-11", None, dict(price=0.667, orders=3.44)),
    ("WB", "Moon", "2026-05-12", "2026-05-17", None, dict(price=0.80, orders=2.25)),
    ("WB", "Moon", "2026-05-18", "2026-06-08", None, dict(price=0.733, orders=2.70)),
    ("WB", "Wendy", "2026-05-10", "2026-06-08", None, dict(price=1.08, orders=1.12)),
    ("WB", "Ruby", "2026-05-13", "2026-06-08", "Выводим", dict(price=0.90, orders=0.93, buyout=1.06)),
]
CABINET_DIP = ("WB", "2026-05-19", "2026-05-21", 0.55)  # demand dip across the whole WB cabinet

HYPOTHESES = [
    ("29.04.2026", "WB", "Wendy", "Тест поискового трафика, возвращение старых кампаний", "Высокая стоимость рекламы",
     "ДРР от продаж 4,5% (-1,8), оптимизация дала все-таки результат, но не для всех артикулов (черный - рост ДРР, тербуется оптимизация). Итого с 30.04 о 07.05 -33к внутреннего бюджета, а заказов от рекламы +45шт или 150к (эффективность выросла). СРС 9,9р (-3,4р)",
     "Света", "05.05.2026", "В работе"),
    ("29.04.2026", "WB", "Audrey", "Добавить промокод", "Низкая конверсия в заказ, трафика много",
     "Отдала промокод подрядчику на всю модель кроме выводимых на 3%. Отчет ожидаем в пн, если по апи выгрузится",
     "Света", "05.05.2026", "В работе"),
    ("29.04.2026", "WB", "Joy", "Увеличить внешний трафик", "Мало трафика, нужно пробовать блогеров",
     "Увеличили на 216к показов, +2,3к переходов за счет адб и адс. но за период меньше заказов (в большей степени это выводимые и те, что заканчиваются на складах лайт браун к примеру). ДРР сильно вырос, кажется, что внешка не сработала как хотелось бы",
     "Света", "05.05.2026", "В работе"),
    ("29.04.2026", "WB", "Joy", "Оптимизировать SEO", "Мало кликов", "", "Настя", "07.05.2026", "В работе"),
    ("29.04.2026", "WB", "Lana", "Увеличить внешний трафик", "Мало трафика, нужно пробовать блогеров",
     "Увеличено показов на 65к, получено на 1,5к больше переходов из внешки по 2,6р, (АДБ и АДС) При этом ДРР от продаж 2,8%",
     "Света", "05.05.2026", "В работе"),
    ("07.05.2026", "WB", "Vuki", "Тест по снижению цены ~ на 20%",
     "Гипотеза: маржинальность ниже, но быстрее оборачиваемость и выше маржа",
     "Vuki прод.:\nЗаказы шт        +90%\nЗаказы руб        +55%\nМаржа руб        +96%\nМаржа до рекламы, %        23,5% -> 16%",
     "Артем", "12.05.2026", "Подтверждено"),
    ("07.05.2026", "WB", "Moon", "Тест по снижению цены ~ на 50% (до 1000 руб)",
     "Гипотеза: маржинальность ниже, но быстрее оборачиваемость и выше маржа",
     "Moon прод.:\nЗаказы шт        +244%\nЗаказы руб        +128%\nМаржа руб        +50%\nМаржа до рекламы, %        27% -> 10%.\n\nРазница по марже не такая большая, \nформально +50%, но сумма маленькая, поэтому \n- это на уровне погрешности. В Vuki при снижении \nцены на 20% сработало намного лучше, хоть \nприрост по заказам не такой большой, как у Moon. \nЕсть гипотеза, что можно цену немного поднять \nдо ~15% маржинальности.",
     "Артем", "12.05.2026", "Опровергнуто"),
    ("10.05.2026", "WB", "Wendy", "Повышение цены перед блогером", "-",
     "Маржинальность до рекламы в % выросла на 1,8 п.п. Маржа в руб (прогнозная) увеличилась от среднего значения. Выросли заказы и в рублях и в штуках.",
     "Артем", "10.05.2026", "Подтверждено"),
    ("12.05.2026", "WB", "Moon", "Повышение цены до ~20% маржинальности (1200 руб). Это ниже, чем до теста 07.05",
     "После снижения цены на 50%, заказы выросли в 2,5 раза, но маржа осталась на том же уровне.",
     "Маржинальность до рекламы в % выросла, но заказы снизились. Годовая доходность такая же, как с маржой 10%.\nГипотеза - снизить цену на 100 руб, чтобы маржа была около 15%.",
     "Артем", "18.05.2026", "Опровергнуто"),
    ("13.05.2026", "WB", "Ruby", "Снижение цены на выводимые ruby", "Падение заказов",
     "За первую неделю (14-20 мая) снижение цены не дало ни роста заказов, ни роста маржи (наоборот - спад). Единственный положительный эффект - вырос % выкупа, но он не перекрыл падение объема.\nПродление теста еще на 7 дней, возможно был шум.",
     "Артем", "01.06.2026", "В работе"),
    ("18.05.2026", "WB", "Moon", "Снижение цены до ~15% маржинальности (1100 руб)", "Цель - рост годовой доходности",
     "Цена 1 100 ₽ дала лучшее сочетание: больше заказов чем при 1 200, и при этом каждая продажа приносит максимальную маржу за весь тест (283 ₽).\nОставить 1 100 ₽ ещё на 7 дней. Последние 3 дня заказы упали с 30 до 13 в день, но это не из-за цены - то же падение видно по всему кабинету, это общая просадка спроса.",
     "Артем", "01.06.2026", "Подтверждено"),
]
HYP_COLUMNS = ["Дата", "Маркетплейс", "Модель", "Гипотеза", "Проблема", "Результат", "Ответственный",
               "Сроки выполнения", "Статус", "Ссылка Bitrix", "Ссылка на результат"]


# --------------------------------------------------------------------------- catalogue
def build_dim_sku(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for art in SKU_TEXT.split():
        prefix, color = art.split("/", 1)
        model = ARTICLE_MODEL.get(prefix) or DISPLAY.get(prefix) or prefix
        rows.append((art, model, color))
    for model in EXTRA_MODELS:
        key = model.lower().replace(" ", "_")
        for color in rng.choice(EXTRA_COLORS, size=int(rng.integers(3, 6)), replace=False):
            rows.append((f"{key}/{color}", model, color))
    dim = pd.DataFrame(rows, columns=["sku", "model", "color"])
    outgoing = (dim["model"] == "Ruby") & dim["color"].isin(OUTGOING_RUBY)
    outgoing |= rng.random(len(dim)) < 0.06
    dim["status"] = np.where(outgoing, "Выводим", "Продается")
    return dim[["sku", "model", "status"]]


def model_cfg(model: str) -> tuple[float, float, float]:
    if model in MODEL_CFG:
        return MODEL_CFG[model]
    base = model.rstrip("WNPT0123456789 ")
    seed_cfg = MODEL_CFG.get(base) or MODEL_CFG["Alice"]
    scale = 0.45 if model.startswith("Set") else 0.35
    return (max(3.0, seed_cfg[0] * scale), seed_cfg[1] * (1.05 if model.endswith("W") else 1.0), seed_cfg[2])


def event_arrays(dates: pd.DatetimeIndex, mp: str, model: str, status: str) -> dict[str, np.ndarray]:
    n = len(dates)
    arr = {k: np.ones(n) for k in ("price", "orders", "cr", "clicks", "buyout", "ext", "ext_cr")}
    arr["int_cpc"] = np.full(n, np.nan)
    arr["int_drr"] = np.full(n, np.nan)
    arr["ext_cpc"] = np.full(n, np.nan)
    for e_mp, e_model, d0, d1, only_status, eff in EVENTS:
        if e_mp != mp or e_model != model or (only_status and only_status != status):
            continue
        m = (dates >= pd.Timestamp(d0)) & (dates <= pd.Timestamp(d1))
        for k, v in eff.items():
            if k in ("int_cpc", "int_drr", "ext_cpc"):
                arr[k][m] = v
            else:
                arr[k][m] *= v
    c_mp, c0, c1, factor = CABINET_DIP
    if mp == c_mp:
        arr["orders"][(dates >= pd.Timestamp(c0)) & (dates <= pd.Timestamp(c1))] *= factor
    return arr


# --------------------------------------------------------------------------- simulation
def simulate_sku(rng, dates, sku, model, status, mp, weight) -> pd.DataFrame:
    base_orders, list_price, sebes_share = model_cfg(model)
    n = len(dates)
    ev = event_arrays(dates, mp, model, status)
    base = base_orders * weight * (1.0 if mp == "WB" else 0.45)
    dow = np.array([1.0, 0.98, 0.97, 1.0, 1.05, 1.12, 1.10])[dates.day_of_week]
    trend = 1 + 0.25 * np.arange(n) / n
    promo = np.ones(n)
    for start in rng.choice(np.arange(5, n - 4), size=3, replace=False):
        promo[start:start + 3] = 1.25
    sku_price = list_price * rng.uniform(0.96, 1.05) * (1.04 if mp == "Ozon" else 1.0)
    price = sku_price * ev["price"] * np.where(promo > 1, 0.95, 1.0) * rng.uniform(0.99, 1.01, n)
    if status == "Выводим":
        base *= 0.55
    lam = base * trend * dow * promo * ev["orders"] * rng.lognormal(0, 0.12, n)
    orders = rng.poisson(np.maximum(lam, 0)).astype(float)

    # sales lag orders by 3-9 days; buyout is a stable SKU property
    buyout = rng.uniform(0.58, 0.82) * ev["buyout"]
    lagged = np.zeros(n)
    for k in range(3, 10):
        lagged += np.roll(orders, k) / 7.0
    lagged[:9] = base * 1.0
    sales = rng.poisson(np.maximum(lagged * buyout, 0)).astype(float)

    spp_base = rng.uniform(0.19, 0.26) if mp == "WB" else rng.uniform(0.10, 0.17)
    spp = np.clip(spp_base + rng.normal(0, 0.008, n), 0.03, 0.4)
    orders_rub = orders * price
    sales_rub = sales * price

    sebes_unit = sku_price * sebes_share * rng.uniform(0.97, 1.03)
    comis_pct = rng.uniform(0.17, 0.215) if mp == "WB" else rng.uniform(0.15, 0.20)
    logist_unit = sku_price * rng.uniform(0.075, 0.10) * (1 if mp == "WB" else 0.92)
    storage_unit = sku_price * rng.uniform(0.012, 0.02)
    other_unit = sku_price * rng.uniform(0.008, 0.018)

    # funnel
    cr_click_order = rng.uniform(0.03, 0.05) * ev["cr"]
    clicks = np.where(orders > 0, orders / cr_click_order * rng.uniform(0.9, 1.1, n), base / cr_click_order * 0.6)
    clicks = clicks * ev["clicks"] * np.where(orders > 0, 1, 0.3)
    carts = np.maximum(orders, clicks * rng.uniform(0.09, 0.13) * rng.uniform(0.92, 1.08, n))

    # internal ads: DRR as share of orders_rub unless the event pins DRR of sales
    drr_orders = rng.uniform(0.032, 0.058) * np.where(np.isnan(ev["int_drr"]), 1.0, 1.0)
    reclama = orders_rub * drr_orders * rng.uniform(0.85, 1.15, n)
    pinned = ~np.isnan(ev["int_drr"])
    reclama = np.where(pinned, sales_rub * ev["int_drr"] * rng.uniform(0.9, 1.1, n), reclama)
    cpc = np.where(np.isnan(ev["int_cpc"]), rng.uniform(11, 16), ev["int_cpc"]) * rng.uniform(0.93, 1.07, n)
    adv_clicks = reclama / cpc
    adv_views = adv_clicks / rng.uniform(0.025, 0.04)
    adv_orders = np.minimum(np.round(adv_clicks * rng.uniform(0.03, 0.045)), orders)
    clicks = np.maximum(clicks, adv_clicks * 1.3)
    carts = np.maximum(carts, adv_orders)

    # stock of the marketplace warehouse
    rate = max(base * 0.9, 0.3)
    stock = np.empty(n)
    s = rate * rng.uniform(35, 70)
    for i in range(n):
        s -= sales[i]
        if s < rate * 9 and rng.random() < 0.12:
            s += rate * rng.uniform(40, 70)
        s = max(s, 0.0)
        stock[i] = s

    return pd.DataFrame({
        "date": dates, "mp": mp, "sku": sku, "model": model,
        "orders_count": orders, "orders_rub": orders_rub, "orders_spp_rub": orders_rub * (1 - spp),
        "sales_qty": sales, "sales_rub": sales_rub, "sebes_rub": sales * sebes_unit,
        "comis_spp": sales_rub * comis_pct, "logist": sales * logist_unit, "storage": sales * storage_unit,
        "other": sales * other_unit, "_spp": spp,
        "clicks_total": clicks, "cart_total": carts,
        "adv_views": adv_views, "adv_clicks": adv_clicks, "adv_orders": adv_orders, "reclama": reclama,
        "stock_mp": stock,
    })


def simulate_ext(rng, grp: pd.DataFrame, mp: str, model: str) -> pd.DataFrame | None:
    """Model-level external advertising for one marketplace: one row per date."""
    day = grp.groupby("date").agg(orders_rub=("orders_rub", "sum"), orders=("orders_count", "sum"))
    dates = day.index
    status_mix = "Продается"
    ev = event_arrays(dates, mp, model, status_mix)
    out = pd.DataFrame(index=dates)
    for ch, (prob, d_lo, d_hi, c_lo, c_hi, ctr_lo, ctr_hi, cr_lo, cr_hi) in EXT_CHANNELS.items():
        active = rng.random(len(dates)) < prob
        spend = np.where(active, day["orders_rub"].values * rng.uniform(d_lo, d_hi, len(dates)) * 0.65, 0.0) * ev["ext"]
        cpc = np.where(np.isnan(ev["ext_cpc"]), rng.uniform(c_lo, c_hi, len(dates)), ev["ext_cpc"])
        clicks = spend / cpc
        views = clicks / rng.uniform(ctr_lo, ctr_hi, len(dates))
        orders = np.minimum(np.round(clicks * rng.uniform(cr_lo, cr_hi, len(dates)) * ev["ext_cr"]),
                            day["orders"].values * 0.3)
        pre = "b" if ch == "b" else ch
        suffix = ("_rub", "_views", "_clicks", "_orders") if ch != "b" else ("_rub", "_views", "_clicks", "_orders")
        out[f"{pre}{suffix[0]}"], out[f"{pre}{suffix[1]}"] = spend, views
        out[f"{pre}{suffix[2]}"], out[f"{pre}{suffix[3]}"] = clicks, orders
        if ch == "b":
            out["b_carts"] = np.maximum(orders, clicks * rng.uniform(0.15, 0.3, len(dates)))
            has_stats = ~((spend > 0) & (rng.random(len(dates)) < 0.03))
            out["b_has_stats"] = has_stats.astype(int)
            for c in ("b_views", "b_clicks", "b_carts", "b_orders"):
                out.loc[~has_stats, c] = 0.0
    out = out.reset_index()
    out.insert(1, "mp", mp)
    out.insert(2, "model", model)
    return out


def simulate_creators(rng, dim: pd.DataFrame) -> pd.DataFrame:
    weeks = pd.date_range(START - pd.Timedelta(days=START.day_of_week), END, freq="7D")
    rows = []
    for model in ["Wendy", "Moon", "Ruby", "Joy", "Vuki", "Audrey", "Lana", "Bella"]:
        for mp in MARKETPLACES:
            for w in weeks:
                if rng.random() > (0.65 if mp == "WB" else 0.35):
                    continue
                spend = rng.uniform(15_000, 90_000) * (1 if mp == "WB" else 0.4)
                impr = spend / rng.uniform(0.25, 0.5)
                clicks = impr * rng.uniform(0.01, 0.03)
                carts = clicks * rng.uniform(0.12, 0.25)
                orders = carts * rng.uniform(0.18, 0.3)
                rows.append({
                    "week_start": w, "week_end": w + pd.Timedelta(days=6), "mp": mp, "model": model,
                    "cr_spend": round(spend), "cr_impr": round(impr), "cr_clicks": round(clicks),
                    "cr_carts": round(carts), "cr_orders": round(orders),
                    "cr_orders_rub": round(orders * rng.uniform(1400, 1800)), "cr_posts": int(rng.integers(2, 12)),
                })
    return pd.DataFrame(rows)


def allocate_ext(fact: pd.DataFrame, ext: pd.DataFrame) -> pd.Series:
    """Total external spend per fact row: model-day spend split by the SKU's share of orders."""
    spend_cols = ["b_rub", "ads_vk_c_rub", "ads_vk_s_rub", "ads_s_c_rub", "ads_ya_c_rub"]
    ext_total = ext.set_index(["date", "mp", "model"])[spend_cols].sum(axis=1).rename("ext_total")
    key = fact.join(ext_total, on=["date", "mp", "model"])
    orders_sum = key.groupby(["date", "mp", "model"])["orders_count"].transform("sum")
    n_sku = key.groupby(["date", "mp", "model"])["sku"].transform("count")
    share = np.where(orders_sum > 0, key["orders_count"] / orders_sum.replace(0, np.nan), 1.0 / n_sku)
    return (key["ext_total"].fillna(0.0) * share)


def main() -> None:
    rng = np.random.default_rng(SEED)
    dim = build_dim_sku(rng)
    dates = pd.date_range(START - pd.Timedelta(days=PRE_DAYS), END)

    weights = {}
    for model, g in dim.groupby("model"):
        w = rng.dirichlet(np.full(len(g), 2.0))
        weights.update(dict(zip(g["sku"], w)))
    frames = []
    for sku, model, status in dim.itertuples(index=False):
        for mp in MARKETPLACES:
            if mp == "Ozon" and rng.random() > 0.55:
                continue
            frames.append(simulate_sku(rng, dates, sku, model, status, mp, weights[sku]))
    fact = pd.concat(frames, ignore_index=True)

    ext_frames = [simulate_ext(rng, g, mp, model) for (mp, model), g in fact.groupby(["mp", "model"])]
    ext = pd.concat(ext_frames, ignore_index=True)
    ext_alloc = allocate_ext(fact, ext)

    nds = 0.05 * fact["sales_rub"] * (1 - fact["_spp"])
    fact["margin_rub"] = (fact["sales_rub"] - fact["sebes_rub"] - fact["comis_spp"] - fact["logist"]
                          - fact["storage"] - fact["other"] - nds - fact["reclama"] - ext_alloc)

    fact = fact[fact["date"] >= START].drop(columns=["_spp", "model"]).reset_index(drop=True)
    ext = ext[ext["date"] >= START].reset_index(drop=True)

    # "Мой склад": own-warehouse stock per SKU, shared by both marketplaces
    rate = fact.groupby("sku")["sales_qty"].mean()
    n = (END - START).days + 1
    ms_rows = []
    for sku, r in rate.items():
        walk = np.cumsum(rng.normal(0, max(r, 0.2) * 0.4, n)) + max(r, 0.2) * rng.uniform(10, 35)
        ms_rows.append(pd.DataFrame({"date": pd.date_range(START, END), "sku": sku,
                                     "qty": np.maximum(walk, 0).round()}))
    stocks_ms = pd.concat(ms_rows, ignore_index=True)

    num = fact.select_dtypes("number").columns
    fact[num] = fact[num].round(2)
    ext_num = ext.select_dtypes("number").columns
    ext[ext_num] = ext[ext_num].round(2)
    for c in ("orders_count", "sales_qty", "stock_mp"):
        fact[c] = fact[c].round().astype(int)

    hyp = pd.DataFrame(HYPOTHESES, columns=HYP_COLUMNS[:9])
    hyp["Ссылка Bitrix"] = ""
    hyp["Ссылка на результат"] = ""
    for c in ("Дата", "Сроки выполнения"):
        hyp[c] = pd.to_datetime(hyp[c], format="%d.%m.%Y")

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as xw:
        dim.to_excel(xw, sheet_name="dim_sku", index=False)
        fact.to_excel(xw, sheet_name="fact_daily", index=False)
        ext.to_excel(xw, sheet_name="fact_ext_daily", index=False)
        stocks_ms.to_excel(xw, sheet_name="stocks_ms", index=False)
        simulate_creators(rng, dim).to_excel(xw, sheet_name="creators_weekly", index=False)
        hyp.to_excel(xw, sheet_name="hypotheses", index=False)
    print(f"{OUT_PATH.name}: skus={len(dim)} models={dim['model'].nunique()} fact={len(fact)} ext={len(ext)} ms={len(stocks_ms)}")


if __name__ == "__main__":
    main()

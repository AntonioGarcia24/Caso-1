import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ================== CONFIG ==================
EXCEL_PATH = r"C:\Users\aanto\Documents\Python\TFM\TFM_datos_sinteticos_Granada_2023_2024_demandaAlta_PR60_modificado.xlsx"
SEQ_LEN, HORIZON = 48, 24
BATCH_SIZE, EPOCHS = 64, 15
OUT_DIR = "finalisima"

# FECHA OBJETIVO (acepta 'YYYY-MM-DD' o 'dd-mm-YYYY')
TARGET_DATE_STR = "2024-03-18"

# Gestión FV / preferencias
HARD_TARGET_MIN_PV = 2.0
ADAPTIVE_RATIO, MIN_FLOOR = 0.60, 0.5
HARD_CUTOFF_HOUR = 19                 # hora tope para que TERMINE el bloque
PREFERRED_WINDOW, PREFERRED_BONUS = (12, 15), 0.5
AVOID_OVERLAP = True

# Cargas flexibles (bloques ejemplo)
THERMO_BLOCK = np.array([1.1, 1.1], dtype=float)  # 2h termo
WASH_BLOCK   = np.array([0.5, 0.4], dtype=float)  # 2h lavadora
FLEX_BLOCKS  = [("termo", THERMO_BLOCK), ("lavadora", WASH_BLOCK)]

# Base mínima orientativa (garantiza margen nocturno al CAPAR la flex)
BASELINE_MIN_KWH = 0.20

# ===== Límites de desplazamiento =====
MAX_BLOCKS_PER_DAY   = 1      # nº máx de bloques a desplazar en el día
MAX_SHIFT_KWH        = 1.5    # tope de kWh totales a mover (None = sin tope)
MAX_SHIFT_FRACTION   = 0.50   # tope como % del total flexible original (None = sin tope)
ALLOW_PARTIAL_LAST_BLOCK = False  # si el último bloque excede cupo, ¿recortarlo proporcionalmente?
MAX_FLEX_PER_HOUR    = None   # tope simultáneo kWh/h (None = sin tope)

# Precio: ¿permitimos negativos?
ALLOW_NEGATIVE_PRICE = False   # pon True si quieres explotar precios negativos

np.random.seed(42); tf.random.set_seed(42)
os.makedirs(OUT_DIR, exist_ok=True)

# ================== UTILIDADES ==================
def cyclical_enc(x, T): return np.sin(2*np.pi*x/T), np.cos(2*np.pi*x/T)

def guess_columns(df):
    cols_l = [c.lower() for c in df.columns]
    t_idx = next((i for i,c in enumerate(cols_l) if any(k in c for k in ["time","fecha","date","timestamp"])), None)
    d_idx = next((i for i,c in enumerate(cols_l) if any(k in c for k in ["demand","demanda","load","consumo"])), None)
    p_idx = next((i for i,c in enumerate(cols_l) if any(k in c for k in ["pv","solar","fotov","generacion","generación","generation"])), None)
    return (df.columns[t_idx] if t_idx is not None else None,
            df.columns[d_idx] if d_idx is not None else None,
            df.columns[p_idx] if p_idx is not None else None)

def make_windows(features, target, seq_len=SEQ_LEN, horizon=HORIZON):
    X, y, T = [], [], len(features)
    for t in range(T - seq_len - horizon + 1):
        X.append(features[t:t+seq_len, :]); y.append(target[t+seq_len:t+seq_len+horizon])
    return np.array(X, np.float32), np.array(y, np.float32)

def build_lstm_model(input_dim, seq_len=SEQ_LEN, horizon=HORIZON, units=64, dropout=0.1):
    inp = layers.Input(shape=(seq_len, input_dim))
    x = layers.LSTM(units, return_sequences=True)(inp); x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units)(x); x = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon)(x)
    model = models.Model(inp, out); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    return model

def schedule_block_solar_then_price(block, nonflex_pred, pv_pred, price, scheduled_flex, hard_cutoff, avoid_overlap=True):
    """
    Primero usa excedente FV, y para lo que quede a red, minimiza coste horario.
    block: np.array (kWh/h) de longitud d (p.ej., [1.1,1.1])
    price: €/kWh por hora (longitud 24)
    """
    d = len(block)
    last_start = max(0, hard_cutoff - d)
    best_i, best_score = None, -1e18
    W_SOLAR, W_PRICE = 1000.0, 1.0  # prioridad fuerte a FV

    for i in range(0, last_start + 1):
        if avoid_overlap and np.any(scheduled_flex[i:i+d] > 1e-9):
            continue
        solar_avail = np.maximum(pv_pred[i:i+d] - (nonflex_pred[i:i+d] + scheduled_flex[i:i+d]), 0.0)
        solar_used = np.minimum(block, solar_avail)
        grid_needed = block - solar_used
        solar_kwh = float(solar_used.sum())
        grid_cost = float(np.dot(grid_needed, price[i:i+d]))  # € = kWh * €/kWh
        score = W_SOLAR * solar_kwh - W_PRICE * grid_cost
        if score > best_score:
            best_score, best_i = score, i
    return best_i

# ================== 1) CARGA ==================
raw = pd.read_excel(EXCEL_PATH)
t_col, d_col, p_col = guess_columns(raw)
if not all([t_col, d_col, p_col]):
    raise ValueError(f"No se detectaron columnas tiempo/demanda/PV. Detectado: tiempo={t_col}, demanda={d_col}, pv={p_col}")

# Columna de precio explícita
price_col = next((c for c in raw.columns if c.lower() == "precio_kwh"), None)
if price_col is None:
    raise ValueError("No se encontró la columna 'precio_kwh' en el Excel.")

df = raw[[t_col, d_col, p_col, price_col]].copy()
df.columns = ["timestamp", "demand_kWh", "pv_kWh", "price_eur_kwh"]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp").asfreq("h")

# Interpolaciones suaves
df["demand_kWh"] = df["demand_kWh"].interpolate()
df["pv_kWh"]     = df["pv_kWh"].interpolate()
df["price_eur_kwh"] = df["price_eur_kwh"].interpolate()

# ================== 2) DATASET ==================
hours_num = df.index.hour.values; doy_num = df.index.dayofyear.values
h_sin, h_cos = cyclical_enc(hours_num, 24); d_sin, d_cos = cyclical_enc(doy_num, 365.25)
feat = np.column_stack([df["demand_kWh"].values, df["pv_kWh"].values, h_sin, h_cos, d_sin, d_cos])
target_load, target_pv = df["demand_kWh"].values, df["pv_kWh"].values

# ================== 2.b) FECHA OBJETIVO Y VENTANA ==================
try:
    TARGET_DATE = pd.to_datetime(TARGET_DATE_STR, format="%Y-%m-%d").normalize()
except ValueError:
    TARGET_DATE = pd.to_datetime(TARGET_DATE_STR, dayfirst=True).normalize()

if TARGET_DATE not in df.index:
    years_avail = sorted(set(df.index.year))
    same_md = [pd.Timestamp(year=int(y), month=TARGET_DATE.month, day=TARGET_DATE.day) for y in years_avail]
    same_md = [d for d in same_md if d in df.index]
    msg = f"La fecha {TARGET_DATE.date()} no está en el índice."
    if same_md: msg += f" Disponibles con mismo mes/día: {[d.date() for d in same_md]}"
    raise ValueError(msg)

pos_start = df.index.get_loc(TARGET_DATE)
if pos_start - SEQ_LEN < 0 or pos_start + HORIZON > len(df):
    raise ValueError("No hay suficiente historial/futuro para la ventana del día objetivo.")

# Entrena con histórico previo (sin fuga temporal)
N_safe = pos_start - SEQ_LEN - HORIZON + 1
if N_safe < 200:
    print(f"[ADVERTENCIA] Pocas ventanas históricas antes de {TARGET_DATE.date()}: N_safe={N_safe}")
X_all_hist, yL_all_hist = make_windows(feat[:pos_start], target_load[:pos_start], SEQ_LEN, HORIZON)
_,              yP_all_hist = make_windows(feat[:pos_start], target_pv[:pos_start],   SEQ_LEN, HORIZON)
X_all_hist, yL_all_hist, yP_all_hist = X_all_hist[:N_safe], yL_all_hist[:N_safe], yP_all_hist[:N_safe]
N = len(X_all_hist)
if N < 50: raise ValueError("Muy pocos patrones históricos. Asegura ~100 ventanas previas.")

n_train = int(0.80*N); n_val = int(0.10*N)
X_train, yL_train, yP_train = X_all_hist[:n_train], yL_all_hist[:n_train], yP_all_hist[:n_train]
X_val,   yL_val,   yP_val   = X_all_hist[n_train:n_train+n_val], yL_all_hist[n_train:n_train+n_val], yP_all_hist[n_train:n_train+n_val]
X_test,  yL_test,  yP_test  = X_all_hist[n_train+n_val:], yL_all_hist[n_train+n_val:], yP_all_hist[n_train+n_val:]

# ================== 3) MODELOS ==================
input_dim = X_all_hist.shape[-1]
model_load = build_lstm_model(input_dim)
model_pv   = build_lstm_model(input_dim)
cb = [callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
      callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5)]

print("Entrenando LSTM — Demanda...")
model_load.fit(X_train, yL_train, validation_data=(X_val, yL_val),
               epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=1)
print("Entrenando LSTM — FV...")
model_pv.fit(X_train, yP_train, validation_data=(X_val, yP_val),
             epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=1)

print(f">> Test MAE Demanda: {model_load.evaluate(X_test, yL_test, verbose=0):.3f} kWh")
print(f">> Test MAE FV:      {model_pv.evaluate(X_test, yP_test, verbose=0):.3f} kWh")

date_tag = TARGET_DATE.strftime("%Y%m%d")
model_load.save(os.path.join(OUT_DIR, f"modelo_demanda_{date_tag}.h5"))
model_pv.save(os.path.join(OUT_DIR, f"modelo_pv_{date_tag}.h5"))

# ================== 4) PREDICCIÓN DÍA OBJETIVO ==================
X_target = feat[pos_start-SEQ_LEN:pos_start, :][None, :, :]
pred_demand = model_load.predict(X_target, verbose=0)[0]
pred_pv     = model_pv.predict(X_target,   verbose=0)[0]

real_demand_day = df["demand_kWh"].iloc[pos_start:pos_start+HORIZON].values
real_pv_day     = df["pv_kWh"].iloc[pos_start:pos_start+HORIZON].values
price_day       = df["price_eur_kwh"].iloc[pos_start:pos_start+HORIZON].values.astype(float)
hours_pred_index = df.index[pos_start:pos_start+HORIZON]

# === 2) Normalización del precio horario ===
price_day = price_day.astype(float)
if not ALLOW_NEGATIVE_PRICE:
    price_day_eff = np.maximum(price_day, 0.0)  # suelo a 0 si hay horas negativas
else:
    price_day_eff = price_day.copy()
print(f"[PRECIO] bruto  min={price_day.min():.4f}  max={price_day.max():.4f}  mean={price_day.mean():.4f} €/kWh")
print(f"[PRECIO] usado  min={price_day_eff.min():.4f}  max={price_day_eff.max():.4f}  mean={price_day_eff.mean():.4f} €/kWh")

# Umbral solar informativo
pv_day_max = float(pred_pv.max())
MIN_PV_FOR_LOAD = HARD_TARGET_MIN_PV if pv_day_max >= HARD_TARGET_MIN_PV else max(MIN_FLOOR, ADAPTIVE_RATIO * pv_day_max)
print(f"[INFO] ({TARGET_DATE.date()}) Umbral PV: {MIN_PV_FOR_LOAD:.2f} kWh (pico {pv_day_max:.2f})")

# ================== 5) GESTIÓN DE CARGAS — FV → PRECIO con LÍMITES ==================
# Flex original (horas de partida; sirve para cuantificar energía desplazable)
orig_flex_raw = np.zeros(HORIZON)
# ejemplo base (ajústalo a tu caso real)
orig_flex_raw[21]+=1.1; orig_flex_raw[22]+=1.1   # termo 21-23
orig_flex_raw[20]+=0.5; orig_flex_raw[21]+=0.4   # lavadora 20-22

# CAPADO: la flex no puede comerse la base mínima
headroom_pred = np.maximum(pred_demand - BASELINE_MIN_KWH, 0.0)
orig_flex = np.minimum(orig_flex_raw, headroom_pred)

# Base no flexible (no subirla artificialmente)
nonflex_pred = np.clip(pred_demand - orig_flex, 0.0, None)

# Cupo permitido de energía a desplazar
total_flex_energy = float(orig_flex.sum())
allowed_energy = total_flex_energy
if MAX_SHIFT_KWH is not None:
    allowed_energy = min(allowed_energy, MAX_SHIFT_KWH)
if MAX_SHIFT_FRACTION is not None:
    allowed_energy = min(allowed_energy, MAX_SHIFT_FRACTION * total_flex_energy)

remaining_energy = allowed_energy
blocks_scheduled = 0
scheduled = np.zeros(HORIZON)

# === 3) Programación de bloques con FV→Precio y límites ===
for name, block in FLEX_BLOCKS:
    if blocks_scheduled >= MAX_BLOCKS_PER_DAY or remaining_energy <= 1e-6:
        break
    bsum = float(block.sum())
    if bsum > remaining_energy:
        if not ALLOW_PARTIAL_LAST_BLOCK:
            print(f"[SKIP] {name}: supera cupo restante ({remaining_energy:.2f} kWh).")
            continue
        block_to_place = block * (remaining_energy / bsum)
    else:
        block_to_place = block.copy()

    start = schedule_block_solar_then_price(
        block=block_to_place,
        nonflex_pred=nonflex_pred,
        pv_pred=pred_pv,
        price=price_day_eff,            # << precio efectivo (con o sin negativos)
        scheduled_flex=scheduled,
        hard_cutoff=HARD_CUTOFF_HOUR,
        avoid_overlap=AVOID_OVERLAP
    )
    if start is None:
        print(f"[SKIP] {name}: sin hueco válido.")
        continue

    scheduled[start:start+len(block_to_place)] += block_to_place
    if MAX_FLEX_PER_HOUR is not None:
        scheduled[start:start+len(block_to_place)] = np.minimum(
            scheduled[start:start+len(block_to_place)], MAX_FLEX_PER_HOUR
        )
    placed = float(block_to_place.sum())
    remaining_energy -= placed
    blocks_scheduled += 1
    print(f"[SCHEDULE] {name} @ {start:02d}:00 → {placed:.2f} kWh (restan {remaining_energy:.2f} kWh)")

# Nada de reescalar a total_flex_energy: respetamos el límite impuesto
moved_energy = float(scheduled.sum())
leftover_orig = orig_flex.copy()
if total_flex_energy > 1e-9:
    moved_ratio = min(moved_energy / total_flex_energy, 1.0)
    leftover_orig = orig_flex * (1.0 - moved_ratio)

# Consumos (misma energía diaria, distinta distribución)
cons_base_pred = nonflex_pred + orig_flex           # == pred_demand hora a hora
cons_mng_pred  = nonflex_pred + leftover_orig + scheduled
assert np.isclose(cons_base_pred.sum(), cons_mng_pred.sum(), atol=1e-6), "No se conserva la energía diaria."

# === Balance energético ANTES vs DESPUÉS (PREDICCIONES) ===
net_before = cons_base_pred - pred_pv
imp_before = np.clip(net_before, 0, None)
exp_before = np.clip(-net_before, 0, None)
autoc_before = cons_base_pred - imp_before

net_after = cons_mng_pred - pred_pv
imp_after = np.clip(net_after, 0, None)
exp_after = np.clip(-net_after, 0, None)
autoc_after = cons_mng_pred - imp_after

print("\n=== Balance energético (kWh) — PREDICCIONES ===")
print(f"Importación ANTES   : {imp_before.sum():.3f} kWh")
print(f"Importación DESPUÉS : {imp_after.sum():.3f} kWh   (Δ {imp_after.sum()-imp_before.sum():+.3f} kWh)")
print(f"Autoconsumo ANTES   : {autoc_before.sum():.3f} kWh")
print(f"Autoconsumo DESPUÉS : {autoc_after.sum():.3f} kWh  (Δ {autoc_after.sum()-autoc_before.sum():+.3f} kWh)")
print(f"Vertido ANTES       : {exp_before.sum():.3f} kWh")
print(f"Vertido DESPUÉS     : {exp_after.sum():.3f} kWh   (Δ {exp_after.sum()-exp_before.sum():+.3f} kWh)")

# === Métricas de error (1 día) DEMANDA y FV (MAPE robusto) ===
def compute_metrics(y_true, y_pred, mask=None):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if mask is not None:
        y_true = y_true[mask]; y_pred = y_pred[mask]
    err  = y_pred - y_true
    mae  = np.mean(np.abs(err)) if y_true.size else np.nan
    rmse = np.sqrt(np.mean(err**2)) if y_true.size else np.nan
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, y_true)
    mape = np.nanmean(np.abs(err/denom))*100
    return mae, rmse, mape

mae_d, rmse_d, mape_d = compute_metrics(real_demand_day, pred_demand)       # Demanda (todas las horas)
sun_mask = real_pv_day > 0.1
mae_p, rmse_p, mape_p = compute_metrics(real_pv_day, pred_pv, mask=sun_mask)  # FV (solo horas con sol)

# === Excel con resultados horarios, totales y métricas ===
excel_path = os.path.join(OUT_DIR, f"resumen_gestion_{date_tag}.xlsx")

# Pestaña 1: Horario
df_hourly = pd.DataFrame({
    "FechaHora":                hours_pred_index,
    "Consumo_base_kWh":         np.round(cons_base_pred, 3),
    "Consumo_gestionado_kWh":   np.round(cons_mng_pred, 3),
    "FV_pred_kWh":              np.round(pred_pv, 3),
    "Precio_(€/kWh)":           np.round(price_day_eff, 4),
    "Importe_red_ANTES_kWh":    np.round(imp_before, 3),
    "Vertido_ANTES_kWh":        np.round(exp_before, 3),
    "Autoconsumo_ANTES_kWh":    np.round(autoc_before, 3),
    "Importe_red_DESPUES_kWh":  np.round(imp_after, 3),
    "Vertido_DESPUES_kWh":      np.round(exp_after, 3),
    "Autoconsumo_DESPUES_kWh":  np.round(autoc_after, 3),
    # (opcionales) reales del día:
    "Consumo_real_kWh":         np.round(real_demand_day, 3),
    "FV_real_kWh":              np.round(real_pv_day, 3),
})

# Pestaña 2: Totales del día
df_tot = pd.DataFrame({
    "Métrica": [
        "Consumo estimado (base)", "Consumo gestionado",
        "FV estimada (día)",
        "Importe red ANTES", "Importe red DESPUÉS",
        "Autoconsumo ANTES", "Autoconsumo DESPUÉS",
        "Vertido ANTES", "Vertido DESPUÉS"
    ],
    "kWh": [
        cons_base_pred.sum(), cons_mng_pred.sum(),
        pred_pv.sum(),
        imp_before.sum(), imp_after.sum(),
        autoc_before.sum(), autoc_after.sum(),
        exp_before.sum(), exp_after.sum()
    ]
}).round(3)

# Pestaña 3: Métricas
df_metrics = pd.DataFrame({
    "Serie":   ["Demanda (24h)", "FV (solo horas con sol)"],
    "MAE_kWh": [mae_d, mae_p],
    "RMSE_kWh":[rmse_d, rmse_p],
    "MAPE_%":  [mape_d, mape_p],
})
df_metrics["MAE_kWh"]  = df_metrics["MAE_kWh"].round(3)
df_metrics["RMSE_kWh"] = df_metrics["RMSE_kWh"].round(3)
df_metrics["MAPE_%"]   = df_metrics["MAPE_%"].round(1)

with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    df_hourly.to_excel(writer, sheet_name="Horario", index=False)
    df_tot.to_excel(writer,    sheet_name="Totales", index=False)
    df_metrics.to_excel(writer, sheet_name="Metricas", index=False)

print(f"\n>>> Excel generado: {os.path.abspath(excel_path)}")


# ================== 6) FIGURAS ==================
# 6.1 Predicción pura (sin cargas)
tot_dem_est = float(pred_demand.sum()); tot_pv_est = float(pred_pv.sum())
pv_vs_dem_pct = (tot_pv_est / tot_dem_est * 100) if tot_dem_est > 0 else float("nan")

plt.figure(figsize=(11,4))
plt.plot(range(HORIZON), pred_demand, label=f"Consumo estimado — {tot_dem_est:.2f} kWh", linewidth=2)
plt.plot(range(HORIZON), pred_pv,     label=f"Producción FV estimada — {tot_pv_est:.2f} kWh ({pv_vs_dem_pct:.1f}%)",
         linewidth=2, linestyle="--", color="gold")
plt.title(f"Demanda vs FV estimadas (sin cargas) — {TARGET_DATE.date()}")
plt.xlabel("Hora"); plt.ylabel("kWh"); plt.xticks(range(HORIZON))
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=9, frameon=True, framealpha=0.9)
ax = plt.gca()
ax.text(0.99, 0.98, f"Totales\n• Consumo estimado: {tot_dem_est:.2f} kWh\n• Producción FV estimada: {tot_pv_est:.2f} kWh",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"fig1_pred_demanda_vs_pv_{date_tag}.png"), dpi=180)
plt.close()

# === 4) Figura final: Consumo base vs gestionado + FV + Precio ===
h = np.arange(HORIZON)
fig, ax1 = plt.subplots(figsize=(12,4.8)); bar_w = 0.4

ax1.bar(h - bar_w/2, cons_base_pred, width=bar_w,
        label=f"Consumo estimado — {cons_base_pred.sum():.2f} kWh", alpha=0.90)
ax1.bar(h + bar_w/2, cons_mng_pred,  width=bar_w,
        label=f"Consumo gestionado — {cons_mng_pred.sum():.2f} kWh", alpha=0.90)
ax1.plot(h, pred_pv, linestyle="--", linewidth=2,
         label=f"FV estimada — {tot_pv_est:.2f} kWh", color="gold")


ax1.set_title(f"Comparativa de consumo estimado y consumo gestionado bajo tarifa dinámica de precios")
ax1.set_xlabel("Hora"); ax1.set_ylabel("kWh"); ax1.set_xticks(h)
ax1.grid(True, linestyle="--", alpha=0.35)

# Eje derecho: precio (€/kWh)
ax2 = ax1.twinx()
ax2.plot(h, price_day_eff, linewidth=2, label="Precio (€/kWh)",color="gray")
ax2.set_ylabel("€/kWh")
if not ALLOW_NEGATIVE_PRICE:
    ax2.set_ylim(bottom=0)  # evita valores negativos en el eje derecho

# Compacta el rango del precio para que no “domine” el gráfico
p_low, p_high = np.quantile(price_day_eff, [0.05, 0.95])
margin = 0.05
ax2.set_ylim(max(0, p_low - margin), p_high + margin)  # ajusta si quieres p. ej. (0.12, 0.22)


# Leyenda combinada
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="upper left", ncol=1, frameon=True, fontsize=9, framealpha=0.9)

plt.tight_layout()
final_fig = os.path.join(OUT_DIR, f"fig5_consumo_fv_precio_{date_tag}.png")
plt.savefig(final_fig, dpi=180)
plt.close()
print(">> Figura final con precio:", final_fig)

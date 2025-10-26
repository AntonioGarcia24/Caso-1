import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ================== CONFIG ==================
EXCEL_PATH = r"C:\Users\aanto\Documents\Python\TFM\TFM_datos_sinteticos_Granada_2023_2024_demandaAlta_PR60_modificado.xlsx"
SEQ_LEN, HORIZON = 48, 24
BATCH_SIZE, EPOCHS = 64, 15
OUT_DIR = "Caso1Marzo"

# --- FECHA OBJETIVO ---
TARGET_DATE_STR = "2024-03-18"      #AÑO/MES/DÍA

# Política de gestión
HARD_TARGET_MIN_PV = 2.0
ADAPTIVE_RATIO, MIN_FLOOR = 0.60, 0.5
HARD_CUTOFF_HOUR = 19
PREFERRED_WINDOW, PREFERRED_BONUS = (12, 15), 0.5
AVOID_OVERLAP = True

# Cargas flexibles
THERMO_DURATION, THERMO_POWER_PER_H = 2, 1.1
WASH_DURATION = 2  # (0.5 + 0.4)

# Suavizado flex
SMOOTH_KERNEL = np.array([0.25, 0.5, 0.25], dtype=float)

# >>> NUEVO: mínimo no flexible por hora (asegura consumo nocturno)
BASELINE_MIN_KWH = 0.8

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

def choose_block_strict(pred_pv_arr, pred_load_arr, duration, preferred_window, preferred_bonus,
                        hard_cutoff, min_pv_for_load, forbidden_hours=None):
    best_i, best_s = None, float("-inf"); pw_s, pw_e = preferred_window; last_start = max(0, hard_cutoff - duration)
    for i in range(0, last_start + 1):
        H = range(i, i+duration)
        if any(h >= hard_cutoff for h in H): continue
        if any(pred_pv_arr[h] < min_pv_for_load for h in H): continue
        if forbidden_hours and any(h in forbidden_hours for h in H): continue
        base = pred_pv_arr[i:i+duration].sum() - pred_load_arr[i:i+duration].sum()
        bonus = sum(1 for h in H if pw_s <= h <= pw_e) * preferred_bonus
        s = base + bonus
        if s > best_s: best_s, best_i = s, i
    return best_i

# ================== 1) CARGA ==================
raw = pd.read_excel(EXCEL_PATH)
t_col, d_col, p_col = guess_columns(raw)
if not all([t_col, d_col, p_col]):
    raise ValueError(f"No se detectaron columnas tiempo/demanda/PV. Detectado: tiempo={t_col}, demanda={d_col}, pv={p_col}")

df = raw[[t_col, d_col, p_col]].copy()
df.columns = ["timestamp", "demand_kWh", "pv_kWh"]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.dropna().sort_values("timestamp").set_index("timestamp")
df = df.asfreq("h")
df["demand_kWh"] = df["demand_kWh"].interpolate()
df["pv_kWh"]     = df["pv_kWh"].interpolate()

# ================== 2) DATASET ==================
hours_num = df.index.hour.values; doy_num = df.index.dayofyear.values
h_sin, h_cos = cyclical_enc(hours_num, 24); d_sin, d_cos = cyclical_enc(doy_num, 365.25)

feat = np.column_stack([df["demand_kWh"].values, df["pv_kWh"].values, h_sin, h_cos, d_sin, d_cos])
target_load, target_pv = df["demand_kWh"].values, df["pv_kWh"].values

# ================== 2.b) FECHA OBJETIVO Y VENTANA ==================
TARGET_DATE = pd.Timestamp(TARGET_DATE_STR).normalize()
if TARGET_DATE not in df.index:
    years_avail = sorted(set(df.index.year))
    same_md = [pd.Timestamp(year=int(y), month=TARGET_DATE.month, day=TARGET_DATE.day) for y in years_avail]
    same_md = [d for d in same_md if d in df.index]
    msg = f"La fecha {TARGET_DATE.date()} no está en el índice del dataset."
    if same_md: msg += f" Disponibles con mismo mes/día: {[d.date() for d in same_md]}"
    raise ValueError(msg)

pos_start = df.index.get_loc(TARGET_DATE)  # 1-abr 00:00
if pos_start - SEQ_LEN < 0 or pos_start + HORIZON > len(df):
    raise ValueError("No hay suficiente historial o futuro en el dataset para la ventana del 1 de abril.")

# Ventanas solo con horizonte COMPLETO antes del objetivo (sin fuga temporal)
N_safe = pos_start - SEQ_LEN - HORIZON + 1
if N_safe < 200:
    print(f"[ADVERTENCIA] Pocas ventanas históricas antes de {TARGET_DATE.date()}: N_safe={N_safe}")

X_all_hist, yL_all_hist = make_windows(feat[:pos_start], target_load[:pos_start], SEQ_LEN, HORIZON)
_,              yP_all_hist = make_windows(feat[:pos_start], target_pv[:pos_start],   SEQ_LEN, HORIZON)
X_all_hist, yL_all_hist, yP_all_hist = X_all_hist[:N_safe], yL_all_hist[:N_safe], yP_all_hist[:N_safe]

N = len(X_all_hist)
if N < 50:
    raise ValueError("Muy pocos patrones históricos tras el recorte temporal. Asegura al menos ~100 ventanas previas.")

n_train = int(0.80*N); n_val = int(0.10*N)
X_train, yL_train, yP_train = X_all_hist[:n_train], yL_all_hist[:n_train], yP_all_hist[:n_train]
X_val,   yL_val,   yP_val   = X_all_hist[n_train:n_train+n_val], yL_all_hist[n_train:n_train+n_val], yP_all_hist[n_train:n_train+n_val]
X_test,  yL_test,  yP_test  = X_all_hist[n_train+n_val:], yL_all_hist[n_train+n_val:], yP_all_hist[n_train+n_val:]

# ================== 3) MODELOS ==================
input_dim = X_all_hist.shape[-1]
model_load = build_lstm_model(input_dim)  # Demanda
model_pv   = build_lstm_model(input_dim)  # FV
cb = [callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
      callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5)]

print("Entrenando LSTM — Demanda (histórico hasta el día anterior a objetivo)...")
model_load.fit(X_train, yL_train, validation_data=(X_val, yL_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=1)
print("Entrenando LSTM — FV (histórico hasta el día anterior a objetivo)...")
model_pv.fit(X_train, yP_train, validation_data=(X_val, yP_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=1)

mae_L = model_load.evaluate(X_test, yL_test, verbose=0)
mae_P = model_pv.evaluate(X_test, yP_test, verbose=0)
print(f">> Test MAE Demanda (kWh): {mae_L:.3f}")
print(f">> Test MAE FV      (kWh): {mae_P:.3f}")

date_tag = pd.to_datetime(TARGET_DATE_STR).strftime("%Y%m%d")
model_load.save(os.path.join(OUT_DIR, f"modelo_demanda_{date_tag}.h5"))
model_pv.save(os.path.join(OUT_DIR, f"modelo_pv_{date_tag}.h5"))

# ================== 4) PREDICCIÓN PARA 1-ABR (24h) ==================
X_target = feat[pos_start-SEQ_LEN:pos_start, :][None, :, :]
pred_demand = model_load.predict(X_target, verbose=0)[0]  # (24,)
pred_pv     = model_pv.predict(X_target,   verbose=0)[0]

# Reales del propio día
real_demand_day = df["demand_kWh"].iloc[pos_start:pos_start+HORIZON].values
real_pv_day     = df["pv_kWh"].iloc[pos_start:pos_start+HORIZON].values
hours_pred_index = df.index[pos_start:pos_start+HORIZON]

# Umbral solar adaptativo
pv_day_max = float(pred_pv.max())
MIN_PV_FOR_LOAD = HARD_TARGET_MIN_PV if pv_day_max >= HARD_TARGET_MIN_PV else max(MIN_FLOOR, ADAPTIVE_RATIO * pv_day_max)
print(f"[INFO] ({TARGET_DATE.date()}) Umbral PV usado: {MIN_PV_FOR_LOAD:.2f} kWh (pico predicho {pv_day_max:.2f})")

# === 4.b) Métricas de error (1 de abril) ===
import numpy as np

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))

    # MAPE robusto: ignora horas con y_true ~ 0 para no explotar
    mask = np.abs(y_true) > 1e-6
    if mask.any():
        mape = np.mean(np.abs(err[mask] / y_true[mask])) * 100.0
    else:
        mape = np.nan

    # sMAPE como alternativa estable (útil si hay ceros)
    smape = np.mean(2.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100.0
    return mae, rmse, mape, smape

# Demandas (kWh/h)
mae_d, rmse_d, mape_d, smape_d = compute_metrics(real_demand_day, pred_demand)
# Fotovoltaica (kWh/h)
mae_p, rmse_p, mape_p, smape_p = compute_metrics(real_pv_day, pred_pv)

print(f"[1-abr] DEMANDA — MAE: {mae_d:.3f} kWh | RMSE: {rmse_d:.3f} kWh | MAPE: {mape_d:.1f}% | sMAPE: {smape_d:.1f}%")
print(f"[1-abr]   FOTOV — MAE: {mae_p:.3f} kWh | RMSE: {rmse_p:.3f} kWh | MAPE: {mape_p:.1f}% | sMAPE: {smape_p:.1f}%")

# (Opcional) Error en TOTales diarios (un solo punto, no tiene RMSE)
tot_err_dem = float(pred_demand.sum() - real_demand_day.sum())
tot_err_pv  = float(pred_pv.sum()     - real_pv_day.sum())
tot_pct_dem = (abs(tot_err_dem) / max(real_demand_day.sum(), 1e-9)) * 100.0
tot_pct_pv  = (abs(tot_err_pv)  / max(real_pv_day.sum(), 1e-9)) * 100.0
print(f"[1-abr] Error total día — Demanda: {tot_err_dem:+.2f} kWh ({tot_pct_dem:.1f}%) | FV: {tot_err_pv:+.2f} kWh ({tot_pct_pv:.1f}%)")


# ================== 5) GESTIÓN DE CARGAS — SIEMPRE QUEDA BASE ==================
# Flex "original" teórica
orig_flex_raw = np.zeros(HORIZON)
orig_flex_raw[21]+=1.1; orig_flex_raw[22]+=1.1; orig_flex_raw[20]+=0.5; orig_flex_raw[21]+=0.4

# CAPADO por hora: no puede comer más que (pred - baseline_min)
headroom_pred = np.maximum(pred_demand - BASELINE_MIN_KWH, 0.0)
orig_flex = np.minimum(orig_flex_raw, headroom_pred)

# Flex gestionada candidata
managed_flex_base = np.zeros(HORIZON); forbidden = set()
# TERMO
thermo_start = choose_block_strict(pred_pv, pred_demand, THERMO_DURATION, PREFERRED_WINDOW, PREFERRED_BONUS,
                                   HARD_CUTOFF_HOUR, MIN_PV_FOR_LOAD, forbidden_hours=forbidden)
if thermo_start is not None:
    managed_flex_base[thermo_start:thermo_start+THERMO_DURATION] += THERMO_POWER_PER_H
    if AVOID_OVERLAP: forbidden |= set(range(thermo_start, thermo_start+THERMO_DURATION))
# LAVADORA
wsh_start = choose_block_strict(pred_pv, pred_demand, WASH_DURATION, PREFERRED_WINDOW, PREFERRED_BONUS,
                                HARD_CUTOFF_HOUR, MIN_PV_FOR_LOAD, forbidden_hours=forbidden)
if wsh_start is not None:
    managed_flex_base[wsh_start] += 0.5; managed_flex_base[wsh_start+1] += 0.4

# Filtro + suavizado + límite por PV
hours = np.arange(HORIZON)
managed_flex_base[(pred_pv < MIN_PV_FOR_LOAD) | (hours >= HARD_CUTOFF_HOUR)] = 0.0
smooth = np.convolve(managed_flex_base, SMOOTH_KERNEL, mode="same") if managed_flex_base.sum()>0 else managed_flex_base.copy()
managed_flex_base = np.minimum(smooth, np.maximum(0.0, pred_pv))
managed_flex_base = np.clip(managed_flex_base, 0.0, None)

# Reescalados para conservar energía flexible (pred y real)
managed_flex_pred = managed_flex_base.copy()
if managed_flex_pred.sum() > 0:
    managed_flex_pred *= (orig_flex.sum() / managed_flex_pred.sum())

# En real también dejamos BASELINE_MIN_KWH por hora
headroom_real = np.maximum(real_demand_day - BASELINE_MIN_KWH, 0.0)
orig_flex_cap_real = np.minimum(orig_flex_raw, headroom_real)
managed_flex_real = managed_flex_base.copy()
if managed_flex_real.sum() > 0:
    managed_flex_real *= (orig_flex_cap_real.sum() / managed_flex_real.sum())

# ================== 6) MÉTRICAS (redistribución, sin sumar kWh) ==================
cons_before_real = real_demand_day
cons_after_real  = (real_demand_day - orig_flex_cap_real) + managed_flex_real
assert np.isclose(cons_before_real.sum(), cons_after_real.sum(), atol=1e-6), "No se conserva energía (real)"

net_before      = cons_before_real - real_pv_day
import_before   = np.clip(net_before, 0, None)
export_before   = np.clip(-net_before, 0, None)
selfcons_before = cons_before_real - import_before

net_after       = cons_after_real - real_pv_day
import_after    = np.clip(net_after, 0, None)
export_after    = np.clip(-net_after, 0, None)
selfcons_after  = cons_after_real - import_after

# ================== 7) TABLAS ==================
summary = pd.DataFrame({
    "Hora": [t.strftime("%Y-%m-%d %H:%M") for t in hours_pred_index],
    "Demanda_real_kWh":        np.round(real_demand_day, 3),
    "PV_real_kWh":             np.round(real_pv_day, 3),
    "Demanda_pred_kWh":        np.round(pred_demand, 3),
    "PV_pred_kWh":             np.round(pred_pv, 3),
    "Flex_original_pred_kWh":  np.round(orig_flex, 3),
    "Flex_gestionada_pred_kWh":np.round(managed_flex_pred, 3),
    "Flex_original_real_cap_kWh": np.round(orig_flex_cap_real, 3),
    "Flex_gestionada_real_kWh":   np.round(managed_flex_real, 3),
    "Importe_red_antes_kWh":   np.round(import_before, 3),
    "Importe_red_despues_kWh": np.round(import_after, 3),
    "Autoconsumo_antes_kWh":   np.round(selfcons_before, 3),
    "Autoconsumo_despues_kWh": np.round(selfcons_after, 3),
})
totals = pd.DataFrame({
    "Métrica":[
        "Consumo total REAL (día)","PV total REAL (día)",
        "Flex original (pred capada)","Flex gestionada (pred esc)",
        "Flex original (real capada)","Flex gestionada (real esc)",
        "Importe red ANTES","Importe red DESPUÉS",
        "Autoconsumo ANTES","Autoconsumo DESPUÉS",
        "Vertido ANTES","Vertido DESPUÉS"
    ],
    "kWh":[
        cons_before_real.sum(), real_pv_day.sum(),
        orig_flex.sum(), managed_flex_pred.sum(),
        orig_flex_cap_real.sum(), managed_flex_real.sum(),
        import_before.sum(), import_after.sum(),
        selfcons_before.sum(), selfcons_after.sum(),
        export_before.sum(), export_after.sum()
    ]
}).round(3)

summary_fp = os.path.join(OUT_DIR, f"detalle_horario_{date_tag}.csv")
totals_fp  = os.path.join(OUT_DIR, f"resumen_totales_{date_tag}.csv")
summary.to_csv(summary_fp, index=False, encoding="utf-8")
totals.to_csv(totals_fp, index=False, encoding="utf-8")
print(f"\n>> CSVs guardados en: {os.path.abspath(summary_fp)}  y  {os.path.abspath(totals_fp)}")

# ================== 8) GRÁFICAS ==================
# 1) Demanda estimada vs FV estimada — SIN cargas
tot_dem_est = float(pred_demand.sum()); tot_pv_est  = float(pred_pv.sum())
pv_vs_dem_pct = (tot_pv_est / tot_dem_est * 100) if tot_dem_est > 0 else float("nan")

plt.figure(figsize=(11,4))
plt.plot(range(HORIZON), pred_demand, label=f"Consumo estimado — Total día: {tot_dem_est:.2f} kWh", linewidth=2)
plt.plot(range(HORIZON), pred_pv,     label=f"Producción FV estimada — Total día: {tot_pv_est:.2f} kWh",
         linewidth=2, linestyle="--", color="gold")
plt.title(f"Consumo estimado y Producción FV estimada")
plt.xlabel("Hora"); plt.ylabel("kWh"); plt.xticks(range(HORIZON))
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=9, frameon=True, framealpha=0.9)
ax = plt.gca()
ax.text(0.99, 0.98, f"Totales del día\n• Consumo: {tot_dem_est:.2f} kWh\n• FV: {tot_pv_est:.2f} kWh",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"fig1_pred_demanda_vs_pv_{date_tag}.png"), dpi=180)
plt.close()

# 2) Columnas: consumo estimado base vs gestionado
h = np.arange(HORIZON)
nonflex_pred    = np.maximum(pred_demand - orig_flex, 0.0)   # >= BASELINE_MIN_KWH por construcción
cons_base_pred  = nonflex_pred + orig_flex                   # == pred_demand
cons_mng_pred   = nonflex_pred + managed_flex_pred           # nunca < nonflex_pred

assert np.isclose(cons_base_pred.sum(), cons_mng_pred.sum(), atol=1e-6), "No se conserva energía (pred)"

fig, ax1 = plt.subplots(figsize=(12,4.2))
bar_w = 0.4
ax1.bar(h - bar_w/2, cons_base_pred, width=bar_w, color="#1f77b4",
        label=f"Consumo estimado — Total día: {cons_base_pred.sum():.2f} kWh", alpha=0.90)
ax1.bar(h + bar_w/2, cons_mng_pred, width=bar_w, color="#ff7f0e",
        label=f"Consumo gestionado — Total día: {cons_mng_pred.sum():.2f} kWh",  alpha=0.90)
ax1.plot(h, pred_pv, linestyle="--", linewidth=2, color="gold", label="Producción FV estimada")
ax1.set_title(f"Comparativa entre consumo estimado y consumo gestionado por algoritmo de IA")
ax1.set_xlabel("Hora"); ax1.set_ylabel("kWh"); ax1.set_xticks(h); ax1.grid(True, linestyle="--", alpha=0.35)
handles, labels = ax1.get_legend_handles_labels(); bylabel = dict(zip(labels, handles))
ax1.legend(bylabel.values(), bylabel.keys(), loc="upper left", ncol=1, frameon=True, fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"fig5_columnas_horarias_demanda_y_pv_{date_tag}.png"), dpi=180)
plt.close()

print(">> Figuras guardadas en:\n  ",
      os.path.join(OUT_DIR, f"fig1_pred_demanda_vs_pv_{date_tag}.png"),
      "\n  ",
      os.path.join(OUT_DIR, f"fig5_columnas_horarias_demanda_y_pv_{date_tag}.png"))

# 3) Consumo REAL vs FV REAL — 1 de abril
real_tot_dem = float(real_demand_day.sum()); real_tot_pv  = float(real_pv_day.sum())
real_cover   = (real_tot_pv / real_tot_dem * 100) if real_tot_dem > 0 else float("nan")

plt.figure(figsize=(11,4))
plt.plot(range(HORIZON), real_demand_day, label=f"Consumo — Total: {real_tot_dem:.2f} kWh", linewidth=2)
plt.plot(range(HORIZON), real_pv_day,     label=f"Producción FV  — Total: {real_tot_pv:.2f} kWh",
         linewidth=2, linestyle="--", color="gold")
plt.title(f"Consumo y Producción FV reales")
plt.xlabel("Hora"); plt.ylabel("kWh")
plt.xticks(range(HORIZON))
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=9, frameon=True, framealpha=0.9)
ax = plt.gca()
ax.text(0.99, 0.98, f"Totales del día\n• Consumo: {real_tot_dem:.2f} kWh\n• FV: {real_tot_pv:.2f} kWh",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"fig_real_consumo_vs_pv_{date_tag}.png"), dpi=180)
plt.close()

# ================== 9) BALANCE ENERGÉTICO CON IDENTIDADES ==================
autoc_before = float(selfcons_before.sum())
autoc_after  = float(selfcons_after.sum())
imp_before   = float(import_before.sum())
imp_after    = float(import_after.sum())
exp_before   = float(export_before.sum())
exp_after    = float(export_after.sum())

cons_before = autoc_before + imp_before
cons_after  = autoc_after  + imp_after
pv_total_b  = autoc_before + exp_before
pv_total_a  = autoc_after  + exp_after

fig, ax = plt.subplots(figsize=(11,5))
labels = ["Consumo", "Producción FV total"]
x = np.arange(len(labels)); width = 0.36

ax.bar(x[0]-width/2, autoc_before, width, label="Autoconsumo ANTES")
ax.bar(x[0]-width/2, imp_before,  width, bottom=autoc_before, label="Importe red ANTES")
ax.bar(x[0]+width/2, autoc_after, width, label="Autoconsumo DESPUÉS")
ax.bar(x[0]+width/2, imp_after,  width, bottom=autoc_after,  label="Importe red DESPUÉS")

ax.bar(x[1]-width/2, autoc_before, width)
ax.bar(x[1]-width/2, exp_before,  width, bottom=autoc_before, label="Vertido ANTES")
ax.bar(x[1]+width/2, autoc_after, width)
ax.bar(x[1]+width/2, exp_after,  width, bottom=autoc_after,  label="Vertido DESPUÉS")

ax.set_title(f"Antes vs Después — {TARGET_DATE.date()}\nIdentidades: Consumo=Autoconsumo+Red · FV=Autoconsumo+Vertido")
ax.set_ylabel("Energía (kWh)")
ax.set_xticks(x, labels); ax.grid(axis="y", linestyle="--", alpha=0.35)

def annotate_total(xpos, total):
    ax.text(xpos, total + 0.15, f"{total:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

annotate_total(x[0]-width/2, cons_before)
annotate_total(x[0]+width/2, cons_after)
annotate_total(x[1]-width/2, pv_total_b)
annotate_total(x[1]+width/2, pv_total_a)

ax.text(0.5, 1.02,
        "Consumo = Autoconsumo + Importe red     ·     Producción FV = Autoconsumo + Vertido",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"))

handles, labels_ = ax.get_legend_handles_labels(); bylabel = dict(zip(labels_, handles))
ax.legend(bylabel.values(), bylabel.keys(), ncol=3, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"fig_balance_identidades_{date_tag}.png"), dpi=180)
plt.close()
print(">> Figuras en:", os.path.join(OUT_DIR, f"fig1_pred_demanda_vs_pv_{date_tag}.png"),
      "|", os.path.join(OUT_DIR, f"fig5_columnas_horarias_demanda_y_pv_{date_tag}.png"),
      "|", os.path.join(OUT_DIR, f"fig_real_consumo_vs_pv_{date_tag}.png"))

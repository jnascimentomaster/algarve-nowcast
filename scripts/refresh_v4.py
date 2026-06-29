#!/usr/bin/env python3
"""
refresh_v4.py
Pipeline do Algarve Nowcast v4 (setorial, Denton hibrido + ECM).

Faz tudo de ponta a ponta:
  Fase 1  serie anual de VAB setorial 1995-2024 (INE, base 2021)
  Fase 2  desagregacao trimestral Denton hibrida (84 trimestres, 2004-2024)
  Fase 3  equacoes ponte v4 (receita como ancora nominal, ECM na construcao)
  Fase 4  nowcast do trimestre corrente, backtest expansivel, data.json

Uso:
  python3 refresh_v4.py                  # corre tudo, escreve public/data.json
  python3 refresh_v4.py --no-fetch       # reutiliza cache em data/ se existir

Pensado para correr semanalmente via GitHub Actions. Os dados intermedios
ficam em data/ para servir de continuidade entre execucoes.

Convencoes INE confirmadas:
  Algarve NUTS II  Dim2=15        aeroporto Faro  Dim2=LPFR
  anual S7A{ano}   trimestral S5A{ano}{trim}   mensal S3A{ano}{mes:02d}
  base 2021 / NUTS 2024, serie consistente desde 1995 (sem splicing manual)
"""

import json, csv, time, urllib.request, argparse, os, sys
from collections import defaultdict
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------------
# Configuracao
# ----------------------------------------------------------------------------
BASE = "https://www.ine.pt/ine/json_indicador/pindica.jsp"
DATA_DIR = "data"
OUT_JSON = "public/data.json"

SECTORS = ["304", "309", "307", "203", "308", "REST"]
SECTOR_NAMES = {
    "304": "Comercio e Turismo", "309": "Administracao Publica",
    "307": "Imobiliario", "203": "Construcao",
    "308": "Consultoria", "REST": "Outros Setores",
}
# Trimestres COVID tratados com dummy (base perto de zero distorce a homologa)
COVID = {"2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4", "2021-Q1", "2021-Q2"}

# Janela de desagregacao trimestral (limitada pelo aeroporto, que comeca em 2004)
DISAGG_START, DISAGG_END = 2004, 2024

# Especificacoes das pontes (Spec E: ancoras nominais cointegradas)
#   cada setor mapeia para a lista de indicadores usados na regressao de nivel
BRIDGE_SPECS = {
    "304": ["revenue"],            # turismo: faturacao turistica (nominal)
    "308": ["revenue"],            # consultoria segue o ciclo do turismo
    "307": ["htx", "revenue"],     # imobiliario: transacoes + receita
    "309": ["unemp", "trend"],     # admin publica: aciclico, tendencia
    "REST": ["unemp", "trend"],    # residual
    # 203 (construcao) usa ECM proprio, nao entra aqui
}
GDP_FACTOR = 1.08   # PIB = VAB x (1 + impostos liquidos sobre produtos)


# ----------------------------------------------------------------------------
# Fetch INE (paralelo, com retries)
# ----------------------------------------------------------------------------
def _fetch(url, timeout=12, tries=2):
    for _ in range(tries):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=timeout).read())
        except Exception:
            time.sleep(0.3)
    return None


def _parse(d, want_dim3=None):
    """Extrai {periodo: valor} de uma resposta INE. want_dim3 filtra por setor."""
    out = {}
    if not (d and isinstance(d, list) and "Dados" in d[0]):
        return out
    for _, vals in d[0]["Dados"].items():
        for v in vals:
            if not v.get("valor"):
                continue
            if want_dim3 and v.get("dim_3") != want_dim3:
                continue
            out[want_dim3 or "v"] = float(v["valor"].replace(",", "."))
    return out


def fetch_series(code, dim2, extra, periods, label=""):
    """Puxa uma serie INE para uma lista de periodos, em paralelo."""
    import concurrent.futures
    def one(p):
        url = f"{BASE}?op=2&varcd={code}&Dim1={p}&Dim2={dim2}{extra}&lang=EN"
        d = _fetch(url)
        if d and isinstance(d, list) and "Dados" in d[0]:
            for _, vals in d[0]["Dados"].items():
                for v in vals:
                    if v.get("valor"):
                        return (p, float(v["valor"].replace(",", ".")))
        return None
    res = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(one, periods):
            if r:
                res[r[0]] = r[1]
    # passo de retry para periodos que falharam (rede instavel no INE)
    for _ in range(2):
        missing = [p for p in periods if p not in res]
        if not missing:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for r in ex.map(one, missing):
                if r:
                    res[r[0]] = r[1]
    if label:
        print(f"  {label}: {len(res)}/{len(periods)} periodos")
    return res


def months(y0, y1):
    return [f"S3A{y}{m:02d}" for y in range(y0, y1 + 1) for m in range(1, 13)]


def quarters(y0, y1):
    return [f"S5A{y}{q}" for y in range(y0, y1 + 1) for q in range(1, 5)]


# ----------------------------------------------------------------------------
# Conversoes de periodo
# ----------------------------------------------------------------------------
def s3a_to_ym(p):      # S3A202403 -> 2024-03
    return f"{p[3:7]}-{p[7:9]}"

def s5a_to_q(p):       # S5A20243 -> 2024-Q3
    return f"{p[3:7]}-Q{p[7]}"

def ym_to_q(ym, store, val):
    y, m = ym.split("-")
    qn = (int(m) - 1) // 3 + 1
    store[f"{y}-Q{qn}"].append(val)

def monthly_to_quarterly(md, agg="sum"):
    q = defaultdict(list)
    for ym, val in md.items():
        ym_to_q(ym, q, val)
    return {k: (sum(v) if agg == "sum" else float(np.mean(v)))
            for k, v in q.items() if len(v) == 3}


# ----------------------------------------------------------------------------
# Fase 1  serie anual de VAB setorial 1995-2024
# ----------------------------------------------------------------------------
def phase1_annual(fetch=True):
    path = f"{DATA_DIR}/sector_gva_annual_1995.csv"
    if not fetch and os.path.exists(path):
        annual = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                annual[int(row["year"])] = {s: float(row[s]) for s in SECTORS + ["TOT"]}
        return annual

    print("Fase 1  serie anual 1995-2024")
    # VAB total a precos correntes (NUTS III Algarve = 150)
    tot_raw = fetch_series("0014113", "150", "&Dim3=TOT",
                           [f"S7A{y}" for y in range(1995, 2025)], "VAB total")
    gva_tot = {int(p[3:7]): v for p, v in tot_raw.items()}

    # Quotas setoriais A10 (Dim2=15), 1995-2023
    shares = {}
    import concurrent.futures
    def pull_shares(y):
        d = _fetch(f"{BASE}?op=2&varcd=0014109&Dim1=S7A{y}&Dim2=15&lang=EN")
        sh = {}
        if d and "Dados" in d[0]:
            for _, vals in d[0]["Dados"].items():
                for v in vals:
                    if v.get("valor") and v.get("dim_3"):
                        sh[v["dim_3"]] = float(v["valor"].replace(",", "."))
        return y, sh
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for y, sh in ex.map(pull_shares, range(1995, 2024)):
            if sh:
                shares[y] = sh
    shares[2024] = dict(shares[2023])   # 2024 ainda sem quotas, herda 2023

    annual = {}
    for y in sorted(gva_tot):
        if y not in shares:
            continue
        tot, sh = gva_tot[y], shares[y]
        modeled = sum(sh.get(s, 0) for s in ["304", "309", "307", "203", "308"])
        rec = {"TOT": tot}
        for s in ["304", "309", "307", "203", "308"]:
            rec[s] = tot * sh.get(s, 0) / 100
        rec["REST"] = tot * (100 - modeled) / 100
        annual[y] = rec

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["year"] + SECTORS + ["TOT"])
        for y in sorted(annual):
            w.writerow([y] + [round(annual[y][s], 1) for s in SECTORS + ["TOT"]])
    print(f"  {len(annual)} anos, agregado bate com INE por construcao")
    return annual


# ----------------------------------------------------------------------------
# Fase 2  desagregacao Denton hibrida
# ----------------------------------------------------------------------------
def denton_proportional(annual_vals, indicator_q, years):
    """Trimestre proporcional ao indicador dentro de cada ano (preserva soma)."""
    out = {}
    for y in years:
        qs = [f"{y}-Q{q}" for q in range(1, 5)]
        if y in annual_vals and all(q in indicator_q for q in qs):
            tot = sum(indicator_q[q] for q in qs)
            if tot > 0:
                for q in qs:
                    out[q] = annual_vals[y] * indicator_q[q] / tot
    return out


def denton_smooth(annual_vals, years):
    """Minimiza a soma de (delta trimestral)^2 sujeito as somas anuais (KKT)."""
    yrs = [y for y in years if y in annual_vals]
    n = len(yrs) * 4
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i] = -1; D[i, i + 1] = 1
    C = np.zeros((len(yrs), n))
    for i in range(len(yrs)):
        C[i, i * 4:(i + 1) * 4] = 1
    ya = np.array([annual_vals[y] for y in yrs])
    K = np.block([[2 * D.T @ D, C.T], [C, np.zeros((len(yrs), len(yrs)))]])
    sol = np.linalg.solve(K, np.concatenate([np.zeros(n), ya]))
    quarters = [f"{y}-Q{q}" for y in yrs for q in range(1, 5)]
    return {quarters[i]: sol[i] for i in range(n)}


def phase2_disaggregate(annual, ind, fetch=True):
    path = f"{DATA_DIR}/sector_qgva_v4.csv"
    if not fetch and os.path.exists(path):
        gva = {s: {} for s in SECTORS}
        with open(path) as f:
            for row in csv.DictReader(f):
                for s in SECTORS:
                    gva[s][row["quarter"]] = float(row[s])
        return gva

    print("Fase 2  desagregacao Denton hibrida")
    years = list(range(DISAGG_START, DISAGG_END + 1))
    av = lambda s: {y: annual[y][s] for y in annual}
    gva = {}
    # ciclicos: Denton proporcional
    gva["304"] = denton_proportional(av("304"), ind["airport_q"], years)
    gva["308"] = denton_proportional(av("308"), ind["airport_q"], years)
    gva["203"] = denton_proportional(av("203"), ind["cost_q"], years)
    # imobiliario: transacoes 2009+, suave antes
    gva["307"] = {**denton_smooth(av("307"), range(2004, 2009)),
                  **denton_proportional(av("307"), ind["htx"], range(2009, 2025))}
    # aciclicos: Denton suave
    gva["309"] = denton_smooth(av("309"), years)
    gva["REST"] = denton_smooth(av("REST"), years)

    out_q = [f"{y}-Q{q}" for y in years for q in range(1, 5)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["quarter"] + SECTORS + ["TOTAL"])
        for q in out_q:
            vals = [gva[s].get(q, 0) for s in SECTORS]
            w.writerow([q] + [round(v, 1) for v in vals] + [round(sum(vals), 1)])

    # validacao: agregado vs INE
    maxd = max(abs(sum(sum(gva[s].get(f"{y}-Q{q}", 0) for q in range(1, 5))
                       for s in SECTORS) / annual[y]["TOT"] - 1) * 100
               for y in years)
    print(f"  84 trimestres, agregado vs INE: diferenca max {maxd:.3f}%")
    return gva


# ----------------------------------------------------------------------------
# Indicadores (mensais e trimestrais)
# ----------------------------------------------------------------------------
def load_indicators(fetch=True, now_year=2026):
    """Devolve dict de indicadores em frequencia trimestral, mais trend."""
    print("Indicadores")
    cache = f"{DATA_DIR}/indicators_cache.json"
    if not fetch and os.path.exists(cache):
        raw = json.load(open(cache))
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        raw = {}
        # mensais
        emb = fetch_series("0000861", "LPFR", "&Dim3=T&Dim4=T", months(2004, now_year), "aeroporto emb")
        dis = fetch_series("0000862", "LPFR", "&Dim3=T&Dim4=T", months(2004, now_year), "aeroporto dis")
        airport = {s3a_to_ym(p): emb.get(p, 0) + dis.get(p, 0) for p in set(emb) | set(dis)}
        cost = {s3a_to_ym(p): v for p, v in fetch_series("0011748", "PT", "&Dim3=T", months(2000, now_year), "custo").items()}
        revenue = {s3a_to_ym(p): v for p, v in fetch_series("0009813", "15", "&Dim3=T", months(2017, now_year), "receita").items()}
        # trimestrais
        unemp = {s5a_to_q(p): v for p, v in fetch_series("0012136", "15", "&Dim3=T", quarters(2011, now_year), "desemprego").items()}
        wages = {s5a_to_q(p): v for p, v in fetch_series("0012134", "15", "&Dim3=B-F", quarters(2011, now_year), "salarios").items()}
        htx = {s5a_to_q(p): v for p, v in fetch_series("0012786", "15", "&Dim3=H1&Dim4=T&Dim5=T", quarters(2009, now_year), "transacoes").items()}
        raw = {"airport_m": airport, "cost_m": cost, "revenue_m": revenue,
               "unemp": unemp, "wages": wages, "htx": htx}
        json.dump(raw, open(cache, "w"))

    ind = {
        "airport_q": monthly_to_quarterly(raw["airport_m"], "sum"),
        "cost_q":    monthly_to_quarterly(raw["cost_m"], "mean"),
        "revenue":   monthly_to_quarterly(raw["revenue_m"], "sum"),
        "unemp":     raw["unemp"], "wages": raw["wages"], "htx": raw["htx"],
    }
    allq = [f"{y}-Q{q}" for y in range(2004, now_year + 1) for q in range(1, 5)]
    ind["trend"] = {q: i for i, q in enumerate(allq)}
    return ind


# ----------------------------------------------------------------------------
# Fase 3 e 4  pontes, nowcast, backtest
# ----------------------------------------------------------------------------
def _matrix(gva, ind, sector, feats, cut=None):
    train_q = [f"{y}-Q{q}" for y in range(DISAGG_START, DISAGG_END + 1) for q in range(1, 5)]
    qs = [q for q in train_q if q in gva[sector] and all(q in ind[f] for f in feats)]
    if cut:
        qs = [q for q in qs if q <= cut]
    return qs


def bridge_fit_predict(gva, ind, sector, feats, target_q, cut=None):
    qs = _matrix(gva, ind, sector, feats, cut)
    if len(qs) < 10 or not all(target_q in ind[f] for f in feats):
        return None, None
    X = np.array([[ind[f][q] for f in feats] + [1.0 if q in COVID else 0.0] for q in qs])
    y = np.array([gva[sector][q] for q in qs])
    ms, ss = X.mean(0), X.std(0) + 1e-9
    m = Ridge(alpha=1.0).fit((X - ms) / ss, y)
    xn = np.array([[ind[f][target_q] for f in feats] + [1.0 if target_q in COVID else 0.0]])
    pred = m.predict((xn - ms) / ss)[0]
    fitted = m.predict((X - ms) / ss)
    r2 = 1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    p = X.shape[1]; r2adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - p - 1)
    rmse = float(np.sqrt(np.mean((y - fitted) ** 2)))
    return pred, {"r2_adj": round(r2adj, 4), "rmse": round(rmse, 1),
                  "n": len(qs), "features": feats}


def ecm_fit_predict(gva, ind, target_q, cut=None):
    """Construcao 203: ECM (salarios LP, delta custo CP, dummy COVID)."""
    wages, cost = ind["wages"], ind["cost_q"]
    qs = sorted([q for q in _matrix(gva, ind, "203", []) if q in wages and q in cost])
    if cut:
        qs = [q for q in qs if q <= cut]
    if len(qs) < 10 or target_q not in cost or target_q not in wages:
        return None, None
    y = np.array([gva["203"][q] for q in qs])
    w = np.array([wages[q] for q in qs]); c = np.array([cost[q] for q in qs])
    A = np.column_stack([np.ones(len(w)), w])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)        # longo prazo
    ecm = y - A @ b
    dy, dc, el = np.diff(y), np.diff(c), ecm[:-1]    # curto prazo
    cov = np.array([1.0 if qs[i + 1] in COVID else 0.0 for i in range(len(dy))])
    Xs = np.column_stack([dc, el, cov]); ms, ss = Xs.mean(0), Xs.std(0) + 1e-9
    m = Ridge(alpha=0.5).fit((Xs - ms) / ss, dy)
    last = qs[-1]; ecm_last = gva["203"][last] - (b[0] + b[1] * wages[last])
    feat = np.array([[cost[target_q] - cost[last], ecm_last,
                      1.0 if target_q in COVID else 0.0]])
    pred = gva["203"][last] + m.predict((feat - ms) / ss)[0]
    fitd = {qs[0]: y[0]}
    for i in range(1, len(qs)):
        f = np.array([[c[i] - c[i - 1], ecm[i - 1], 1.0 if qs[i] in COVID else 0.0]])
        fitd[qs[i]] = y[i - 1] + m.predict((f - ms) / ss)[0]
    rmse = float(np.sqrt(np.mean([(fitd[q] - gva["203"][q]) ** 2 for q in qs])))
    return pred, {"r2_adj": None, "rmse": round(rmse, 1), "n": len(qs),
                  "features": ["salarios (LP)", "custo (CP)", "ECM", "covid"]}


def predict_sector(gva, ind, s, q, cut=None):
    if s == "203":
        return ecm_fit_predict(gva, ind, q, cut)[0]
    return bridge_fit_predict(gva, ind, s, BRIDGE_SPECS[s], q, cut)[0]


def backtest(gva, ind, y0=2019, y1=2024):
    """Janela expansivel out-of-sample, exclui COVID. Devolve vies e MAE."""
    errs = []
    for ty in range(y0, y1 + 1):
        for q in range(1, 5):
            tq = f"{ty}-Q{q}"
            if tq in COVID:
                continue
            pr = {s: predict_sector(gva, ind, s, tq, f"{ty-1}-Q4") for s in SECTORS}
            if any(v is None for v in pr.values()):
                continue
            actual = sum(gva[s][tq] for s in SECTORS)
            errs.append((sum(pr.values()) - actual) / actual * 100)
    errs = np.array(errs)
    return float(np.mean(errs)), float(np.mean(np.abs(errs))), len(errs)


def carry_forward(ind, now_q):
    """Para indicadores trimestrais sem o trimestre corrente, herda o homologo."""
    for name in ["unemp", "wages", "htx", "revenue"]:
        d = ind[name]
        if now_q not in d:
            y, qn = now_q.split("-Q")
            py = f"{int(y)-1}-Q{qn}"
            d[now_q] = d.get(py, d[sorted(d)[-1]])


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run(fetch=True, now_q="2026-Q1"):
    now_year = int(now_q[:4])
    annual = phase1_annual(fetch)
    ind = load_indicators(fetch, now_year)
    gva = phase2_disaggregate(annual, ind, fetch)
    carry_forward(ind, now_q)

    print("Fase 3 e 4  pontes, nowcast, backtest")
    bias, mae, n = backtest(gva, ind)
    corr = 1 / (1 + bias / 100)        # fator de correcao de vies
    py_q = f"{now_year-1}-Q{now_q[-1]}"

    now, prev, diag = {}, {}, {}
    for s in SECTORS:
        if s == "203":
            now[s], diag[s] = ecm_fit_predict(gva, ind, now_q)
            prev[s], _ = ecm_fit_predict(gva, ind, py_q)
        else:
            now[s], diag[s] = bridge_fit_predict(gva, ind, s, BRIDGE_SPECS[s], now_q)
            prev[s], _ = bridge_fit_predict(gva, ind, s, BRIDGE_SPECS[s], py_q)

    tot = sum(now[s] for s in SECTORS)
    tp = sum(prev[s] for s in SECTORS)
    rmse_agg = float(np.sqrt(sum(diag[s]["rmse"] ** 2 for s in SECTORS)))

    data = {
        "updated": time.strftime("%Y-%m-%d"),
        "nowcast_quarter": now_q,
        "version": "Algarve Nowcast v4 Setorial",
        "aggregate": {
            "gva_meur": round(tot, 1), "gva_corrected_meur": round(tot * corr, 1),
            "gdp_meur": round(tot * GDP_FACTOR, 1),
            "gdp_corrected_meur": round(tot * corr * GDP_FACTOR, 1),
            "yoy_pct": round((tot / tp - 1) * 100, 1),
            "rmse_meur": round(rmse_agg, 1),
            "lower_90": round(tot * corr - 1.645 * rmse_agg, 1),
            "upper_90": round(tot * corr + 1.645 * rmse_agg, 1),
            "bias_correction_pct": round(bias, 1),
        },
        "sectors": {s: {
            "point": round(now[s], 1), "point_corrected": round(now[s] * corr, 1),
            "weight_pct": round(now[s] / tot * 100, 1),
            "prev_year": round(prev[s], 1),
            "yoy_pct": round((now[s] / prev[s] - 1) * 100, 1),
            "prev_year_q": py_q, "name": SECTOR_NAMES[s],
        } for s in SECTORS},
        "diagnostics": {s: {**diag[s], "dw": None} for s in SECTORS},
        "validation": {
            "mae_pct": round(mae, 1), "bias_pct": round(bias, 1),
            "n_quarters": n,
            "method": "Janela expansivel out-of-sample 2019-2024, exclui COVID",
        },
        "sector_quarterly_gva": {
            s: {q: round(gva[s][q], 1) for q in sorted(gva[s])} for s in SECTORS},
        "indicators": {
            name: {q: round(ind[key][q], 1) for q in sorted(ind[key])
                   if "2017-Q1" <= q <= now_q}
            for name, key in [("revenue", "revenue"), ("airport", "airport_q"),
                              ("unemp", "unemp"), ("wages", "wages"),
                              ("htx", "htx"), ("cost", "cost_q")]},
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(data, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)

    print(f"\nNowcast {now_q}: VAB {tot:.0f}M (corr {tot*corr:.0f}M), "
          f"PIB {tot*GDP_FACTOR:.0f}M (corr {tot*corr*GDP_FACTOR:.0f}M), "
          f"homologa {(tot/tp-1)*100:+.1f}%")
    print(f"Backtest: vies {bias:+.1f}%, MAE {mae:.1f}% (n={n})")
    print(f"Escrito: {OUT_JSON}")
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true",
                    help="reutiliza cache em data/ em vez de puxar do INE")
    ap.add_argument("--quarter", default="2026-Q1", help="trimestre a estimar")
    args = ap.parse_args()
    run(fetch=not args.no_fetch, now_q=args.quarter)

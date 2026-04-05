#!/usr/bin/env python3
"""
refresh.py — Automated data refresh for Algarve Nowcast v3
============================================================
Run weekly via GitHub Actions. Pulls latest INE data, reruns models,
writes public/data.json for static site.

Usage: python scripts/refresh.py
"""

import numpy as np
import pandas as pd
import requests
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from scipy import linalg
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

INE_BASE = "https://www.ine.pt/ine/json_indicador/pindica.jsp"
DATA_DIR = Path("data")
PUBLIC_DIR = Path("public")
DATA_DIR.mkdir(exist_ok=True)
PUBLIC_DIR.mkdir(exist_ok=True)

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

# ── INE API ──

def ine_fetch(varcd, dim1, dim2, dim3=None, dim4=None, dim5=None, retries=3):
    params = {'op': '2', 'varcd': varcd, 'Dim1': dim1, 'Dim2': dim2, 'lang': 'EN'}
    if dim3: params['Dim3'] = dim3
    if dim4: params['Dim4'] = dim4
    if dim5: params['Dim5'] = dim5
    for attempt in range(retries):
        try:
            r = requests.get(INE_BASE, params=params, timeout=20)
            data = r.json()
            if isinstance(data, list) and data:
                dados = data[0].get('Dados', {})
                for period, entries in dados.items():
                    if isinstance(entries, list):
                        for e in entries:
                            v = e.get('valor')
                            if v and v != '-':
                                return float(v)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  WARN: INE fetch failed {varcd}/{dim1}: {e}", file=sys.stderr)
    return None


def pull_monthly(varcd, dim2, dim3, col_name, start_year=2017):
    """Pull a monthly INE series."""
    records = []
    end_month = CURRENT_MONTH if CURRENT_YEAR == 2026 else 12  # conservative
    for year in range(start_year, CURRENT_YEAR + 1):
        max_m = min(end_month, 12) if year == CURRENT_YEAR else 12
        for month in range(1, max_m + 1):
            val = ine_fetch(varcd, f'S3A{year}{month:02d}', dim2, dim3)
            if val is not None:
                records.append({'date': f'{year}-{month:02d}-01', col_name: val})
            time.sleep(0.2)
    return pd.DataFrame(records)


def pull_quarterly(varcd, dim2, col_name, dim3=None, dim4=None, dim5=None, start_year=2017):
    """Pull a quarterly INE series."""
    records = []
    for year in range(start_year, CURRENT_YEAR + 1):
        for q in range(1, 5):
            val = ine_fetch(varcd, f'S5A{year}{q}', dim2, dim3, dim4, dim5)
            if val is not None:
                records.append({'quarter': f'{year}-Q{q}', col_name: val})
            time.sleep(0.2)
    return pd.DataFrame(records)


# ── Chow-Lin ──

def ar1_vcv(n, rho):
    rho = np.clip(rho, -0.999, 0.999)
    idx = np.arange(n)
    V = rho ** np.abs(idx[:, None] - idx[None, :])
    return V / (1.0 - rho**2)


def chowlin(y_annual, x_quarterly):
    n_a, n_q = len(y_annual), len(y_annual) * 4
    x = x_quarterly.reshape(-1, 1) if x_quarterly.ndim == 1 else x_quarterly
    C = np.zeros((n_a, n_q))
    for i in range(n_a):
        C[i, i*4:(i+1)*4] = 1.0
    X = np.column_stack([np.ones(n_q), x])
    Xa = C @ X

    def neg_ll(rho):
        if abs(rho) >= 0.999: return 1e10
        try:
            V = ar1_vcv(n_q, rho); Va = C @ V @ C.T; Va_inv = linalg.inv(Va)
            beta = linalg.inv(Xa.T @ Va_inv @ Xa) @ Xa.T @ Va_inv @ y_annual
            resid = y_annual - Xa @ beta
            sign, logdet = np.linalg.slogdet(Va)
            return 1e10 if sign <= 0 else 0.5 * (logdet + resid @ Va_inv @ resid)
        except:
            return 1e10

    rho = minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method='bounded').x
    V = ar1_vcv(n_q, rho); Va_inv = linalg.inv(C @ V @ C.T)
    beta = linalg.inv(Xa.T @ Va_inv @ Xa) @ Xa.T @ Va_inv @ y_annual
    D = V @ C.T @ Va_inv
    return X @ beta + D @ (y_annual - Xa @ beta), rho


# ── Helpers ──

def m2q(df, val_col, method='sum', min_months=3):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    q = df[[val_col]].resample('QS').sum() if method == 'sum' else df[[val_col]].resample('QS').mean()
    counts = df[[val_col]].resample('QS').count()
    q = q[counts[val_col] >= min_months]
    q.index = [f"{d.year}-Q{(d.month-1)//3+1}" for d in q.index]
    return q[val_col]


def extrapolate_q(df, val_col, target_q, method='sum'):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['qtr'] = [f"{d.year}-Q{(d.month-1)//3+1}" for d in df.index]
    df['miq'] = [(d.month - 1) % 3 + 1 for d in df.index]
    cur = df[df['qtr'] == target_q]
    if len(cur) == 0:
        return None, 0
    months = len(cur)
    year, qn = int(target_q.split('-')[0]), int(target_q.split('Q')[1])
    ratios = []
    for y in range(year - 3, year):
        hd = df[df['qtr'] == f"{y}-Q{qn}"]
        if len(hd) == 3:
            partial = hd[hd['miq'] <= months]
            if len(partial) > 0 and partial[val_col].sum() > 0:
                r = hd[val_col].sum() / partial[val_col].sum() if method == 'sum' \
                    else hd[val_col].mean() / partial[val_col].mean()
                ratios.append(r)
    scale = np.median(ratios) if ratios else (3.0 / months if method == 'sum' else 1.0)
    est = cur[val_col].sum() * scale if method == 'sum' else cur[val_col].mean() * scale
    return est, months / 3.0


def fit_bridge(y, X, alpha=1.0):
    common = X.index.intersection(y.index)
    X_fit = X.loc[common].copy()
    y_fit = y.loc[common].copy()
    mask = X_fit.notna().all(axis=1) & y_fit.notna()
    X_fit, y_fit = X_fit[mask], y_fit[mask]
    if len(X_fit) < 8:
        return None
    X_fit['covid'] = [1.0 if '2020' in q or ('2021' in q and 'Q1' in q) else 0.0
                      for q in X_fit.index]
    feats = list(X_fit.columns)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_fit)
    model = Ridge(alpha=alpha).fit(X_s, y_fit)
    pred = model.predict(X_s)
    rmse = np.sqrt(mean_squared_error(y_fit, pred))
    ss_res = np.sum((y_fit.values - pred) ** 2)
    ss_tot = np.sum((y_fit.values - y_fit.mean()) ** 2)
    n, k = len(X_fit), len(feats)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2
    resid = y_fit.values - pred
    dw = np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2) if np.sum(resid ** 2) > 0 else None
    return {'model': model, 'scaler': scaler, 'feats': feats, 'r2_adj': r2_adj,
            'rmse': rmse, 'n': n, 'dw': dw, 'r2': r2}


# ═══════════════════════════════════════════════════════════
# MAIN REFRESH
# ═══════════════════════════════════════════════════════════

def main():
    print(f"Algarve Nowcast refresh — {datetime.now().isoformat()}")

    # ── 1. Pull all data ──
    print("\n1. Pulling INE data...")

    # Total GVA annual (0014113)
    gva_records = []
    for year in range(1995, CURRENT_YEAR + 1):
        val = ine_fetch('0014113', f'S7A{year}', '150', 'TOT')
        if val: gva_records.append({'year': year, 'gva': val})
        time.sleep(0.15)
    gva_df = pd.DataFrame(gva_records)
    gva_total = {g['year']: g['gva'] for g in gva_records}
    print(f"  GVA annual: {len(gva_records)} years")

    # Sector shares (0014109)
    SECTORS = ['304', '309', '307', '203', '308', '101', '202', '305', '306', '310']
    shares = {}
    for s in SECTORS:
        shares[s] = {}
        for year in range(1995, CURRENT_YEAR + 1):
            val = ine_fetch('0014109', f'S7A{year}', '15', s)
            if val: shares[s][year] = val
            time.sleep(0.1)
    print(f"  Sector shares: {len(shares)} sectors")

    # Monthly series
    nights = pull_monthly('0009808', '15', 'T', 'overnight_stays')
    hotel = pull_monthly('0009808', '15', '01', 'hotel_nights')
    al = pull_monthly('0009808', '15', '02', 'al_nights')
    revenue = pull_monthly('0009813', '15', 'T', 'revenue_keur')
    permits = pull_monthly('0012098', '15', None, 'building_permits')

    # Airport passengers
    emb = pull_monthly('0000861', 'LPFR', 'T', 'embarked')
    dis = pull_monthly('0000862', 'LPFR', 'T', 'disembarked')
    if len(emb) and len(dis):
        airport = emb.merge(dis, on='date')
        airport['airport_pax'] = airport['embarked'] + airport['disembarked']
    else:
        airport = pd.DataFrame()

    # Quarterly series
    htx = pull_quarterly('0012786', '15', 'housing_tx_keur', 'H1', 'T', 'T')
    unemp = pull_quarterly('0012136', '15', 'unemployment_rate', 'T')

    print(f"  Nights: {len(nights)}, Revenue: {len(revenue)}, Permits: {len(permits)}")
    print(f"  Airport: {len(airport)}, Housing TX: {len(htx)}, Unemployment: {len(unemp)}")

    # Save raw data
    for name, df in [('nights', nights), ('hotel_nights', hotel), ('al_nights', al),
                     ('revenue_full', revenue), ('building_permits', permits),
                     ('housing_transactions', htx), ('unemployment', unemp)]:
        if len(df): df.to_csv(DATA_DIR / f'{name}.csv', index=False)
    if len(airport): airport.to_csv(DATA_DIR / f'airport.csv', index=False)
    with open(DATA_DIR / 'gva.json', 'w') as f:
        json.dump(gva_records, f)
    with open(DATA_DIR / 'sector_shares.json', 'w') as f:
        json.dump(shares, f, default=str)

    # ── 2. Compute sector GVA ──
    print("\n2. Computing sector GVA...")
    KEY_SECTORS = ['304', '309', '307', '203', '308']
    sector_gva = {}
    for s in KEY_SECTORS:
        sector_gva[s] = {}
        for year in range(1995, CURRENT_YEAR + 1):
            sh = shares[s].get(year)
            if sh and year in gva_total:
                sector_gva[s][year] = sh / 100 * gva_total[year]
        # Extend latest year with previous year's shares if needed
        max_share_year = max(shares[s].keys()) if shares[s] else 0
        if max_share_year < CURRENT_YEAR and CURRENT_YEAR in gva_total:
            sector_gva[s][CURRENT_YEAR] = shares[s][max_share_year] / 100 * gva_total[CURRENT_YEAR]

    sector_gva['REST'] = {}
    for year in range(1995, CURRENT_YEAR + 1):
        if year in gva_total:
            modelled = sum(sector_gva[s].get(year, 0) for s in KEY_SECTORS)
            if modelled > 0:
                sector_gva['REST'][year] = gva_total[year] - modelled

    # ── 3. Quarterly aggregation ──
    print("\n3. Building quarterly features...")
    nights_q = m2q(nights, 'overnight_stays')
    hotel_q = m2q(hotel, 'hotel_nights')
    al_q = m2q(al, 'al_nights')
    permits_q = m2q(permits, 'building_permits')
    revenue_q = m2q(revenue, 'revenue_keur')
    airport_q = m2q(airport, 'airport_pax') if len(airport) else pd.Series(dtype=float)
    htx_q = htx.set_index('quarter')['housing_tx_keur'] if len(htx) else pd.Series(dtype=float)
    unemp_q = unemp.set_index('quarter')['unemployment_rate'] if len(unemp) else pd.Series(dtype=float)

    all_q = pd.DataFrame({'nights': nights_q, 'hotel_nights': hotel_q, 'al_nights': al_q,
                           'building_permits': permits_q, 'revenue': revenue_q,
                           'airport_pax': airport_q, 'housing_tx': htx_q,
                           'unemployment': unemp_q})
    mask = all_q['revenue'].notna() & (all_q['nights'] > 0)
    all_q.loc[mask, 'revpar'] = all_q.loc[mask, 'revenue'] / all_q.loc[mask, 'nights'] * 1000
    all_q['al_share'] = all_q['al_nights'] / all_q['nights']

    # ── 4. Chow-Lin per sector ──
    print("\n4. Chow-Lin disaggregation...")
    START_YEAR = 2017
    END_YEAR = max(y for y in gva_total.keys() if y <= CURRENT_YEAR)
    years = list(range(START_YEAR, END_YEAR + 1))
    n_q = len(years) * 4
    q_dates = pd.date_range(f"{START_YEAR}-01-01", periods=n_q, freq='QS')
    q_labels = [f"{d.year}-Q{(d.month-1)//3+1}" for d in q_dates]

    INTERPOLATORS = {'304': 'nights', '309': 'trend', '307': 'housing_tx',
                     '203': 'building_permits', '308': 'nights', 'REST': 'trend'}

    sector_quarterly = {}
    sector_rhos = {}
    for s in KEY_SECTORS + ['REST']:
        y_ann = np.array([sector_gva[s].get(y, np.nan) for y in years])
        valid = ~np.isnan(y_ann)
        if not valid.all():
            vi = np.where(valid)[0]
            y_ann = np.interp(np.arange(len(years)), vi, y_ann[valid])
        itype = INTERPOLATORS[s]
        if itype == 'trend':
            x_q = np.linspace(0, 1, n_q)
        elif itype in all_q.columns:
            x_q = np.array([all_q[itype].get(ql, np.nan) for ql in q_labels])
            nans = np.isnan(x_q)
            if nans.any():
                vi = np.where(~nans)[0]
                x_q[nans] = np.interp(np.where(nans)[0], vi, x_q[vi]) if len(vi) >= 2 \
                    else np.linspace(0, 1, n_q)[nans]
        else:
            x_q = np.linspace(0, 1, n_q)
        gva_q, rho = chowlin(y_ann, x_q)
        sector_quarterly[s] = pd.Series(gva_q, index=q_labels)
        sector_rhos[s] = rho
        print(f"  {s}: ρ={rho:.3f}")

    # ── 5. Bridge equations ──
    print("\n5. Fitting bridges...")
    bridges = {}
    bridges['304'] = fit_bridge(sector_quarterly['304'],
                                all_q[['nights', 'revpar', 'al_share']])
    bridges['309'] = fit_bridge(sector_quarterly['309'],
                                pd.DataFrame({'trend': np.arange(n_q, dtype=float)}, index=q_labels))
    bridges['307'] = fit_bridge(sector_quarterly['307'],
                                pd.DataFrame({'housing_tx': htx_q,
                                              'trend': np.arange(len(q_labels), dtype=float)},
                                             index=q_labels))
    bridges['203'] = fit_bridge(sector_quarterly['203'],
                                all_q[['building_permits']])
    bridges['308'] = fit_bridge(sector_quarterly['308'],
                                all_q[['nights', 'revpar']])
    bridges['REST'] = fit_bridge(sector_quarterly['REST'],
                                 pd.DataFrame({'trend': np.arange(n_q, dtype=float)}, index=q_labels))

    for s, b in bridges.items():
        if b:
            print(f"  {s}: R²adj={b['r2_adj']:.4f}, RMSE={b['rmse']:.1f}")

    # ── 6. Nowcast current quarter ──
    # Determine current nowcast quarter
    nowcast_q_num = (CURRENT_MONTH - 1) // 3 + 1
    nowcast_q = f"{CURRENT_YEAR}-Q{nowcast_q_num}"
    print(f"\n6. Nowcasting {nowcast_q}...")

    X_now = {}
    for label, df_src, col in [('nights', nights, 'overnight_stays'),
                                ('hotel_nights', hotel, 'hotel_nights'),
                                ('al_nights', al, 'al_nights'),
                                ('building_permits', permits, 'building_permits'),
                                ('revenue', revenue, 'revenue_keur')]:
        est, comp = extrapolate_q(df_src, col, nowcast_q)
        if est: X_now[label] = est

    if len(airport):
        est_a, _ = extrapolate_q(airport, 'airport_pax', nowcast_q)
        if est_a: X_now['airport_pax'] = est_a

    if X_now.get('nights', 0) > 0:
        X_now['al_share'] = X_now.get('al_nights', 0) / X_now['nights']
        if 'revenue' in X_now:
            X_now['revpar'] = X_now['revenue'] / X_now['nights'] * 1000

    X_now['housing_tx'] = htx_q.iloc[-1] if len(htx_q) else None
    X_now['unemployment'] = unemp_q.iloc[-1] if len(unemp_q) else None
    trend_val = float(n_q)  # next quarter beyond training

    SECTOR_NAMES = {'304': 'Trade/Tourism', '309': 'Public Admin', '307': 'Real Estate',
                    '203': 'Construction', '308': 'Consultancy', 'REST': 'Other Sectors'}

    sector_nowcasts = {}
    total_var = 0
    for s, bridge in bridges.items():
        if not bridge: continue
        x_vec = []
        ok = True
        for f in bridge['feats']:
            if f == 'covid': x_vec.append(0.0)
            elif f == 'trend': x_vec.append(trend_val)
            elif f in X_now and X_now[f] is not None: x_vec.append(X_now[f])
            else: ok = False; break
        if not ok: continue
        x_s = bridge['scaler'].transform(np.array(x_vec).reshape(1, -1))
        point = bridge['model'].predict(x_s)[0]
        sector_nowcasts[s] = {'point': round(point, 1), 'rmse': round(bridge['rmse'], 1),
                              'r2_adj': round(bridge['r2_adj'], 4)}
        total_var += bridge['rmse'] ** 2

    total_gva = sum(sn['point'] for sn in sector_nowcasts.values())
    total_rmse = np.sqrt(total_var)

    print(f"\n  Aggregate GVA: €{total_gva:,.1f}M ± €{total_rmse:,.1f}M")

    # ── 7. Build data.json ──
    print("\n7. Writing public/data.json...")
    output = {
        'updated': datetime.now().strftime('%Y-%m-%d'),
        'nowcast_quarter': nowcast_q,
        'nowcast': {
            'gva_meur': round(total_gva, 1),
            'gdp_meur': round(total_gva * 1.08, 1),
            'rmse_meur': round(total_rmse, 1),
            'lower_90': round(total_gva - 1.645 * total_rmse, 1),
            'upper_90': round(total_gva + 1.645 * total_rmse, 1),
        },
        'sectors': {s: {**sn, 'name': SECTOR_NAMES.get(s, 'Other'),
                         'weight_pct': round(sn['point'] / total_gva * 100, 1)}
                    for s, sn in sector_nowcasts.items()},
        'diagnostics': {s: {'r2_adj': round(b['r2_adj'], 4), 'rmse': round(b['rmse'], 1),
                             'dw': round(b['dw'], 3) if b['dw'] else None,
                             'n': b['n'], 'features': b['feats']}
                        for s, b in bridges.items() if b},
        'chowlin_rho': {s: round(r, 3) for s, r in sector_rhos.items()},
        'sector_quarterly': {},
        'indicators': {},
        'sector_shares': {s: {str(y): round(v, 1) for y, v in vals.items()}
                          for s, vals in shares.items()},
        'gva_annual': {str(g['year']): g['gva'] for g in gva_records},
    }

    # Sector quarterly GVA for charts
    for s in KEY_SECTORS + ['REST']:
        output['sector_quarterly'][f'gva_{s}'] = {
            q: round(v, 1) for q, v in sector_quarterly[s].items()
        }
    # Total
    output['sector_quarterly']['gva_total_sectoral'] = {
        q: round(sum(sector_quarterly[s][q] for s in KEY_SECTORS + ['REST']), 1)
        for q in q_labels
    }

    # Indicator series
    for col in ['nights', 'revpar', 'al_share', 'building_permits',
                'airport_pax', 'housing_tx', 'unemployment']:
        if col in all_q.columns:
            s = all_q[col].dropna()
            output['indicators'][col] = {q: round(v, 2) for q, v in s.items()}

    with open(PUBLIC_DIR / 'data.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. data.json written ({os.path.getsize(PUBLIC_DIR / 'data.json')} bytes)")
    return 0


if __name__ == '__main__':
    sys.exit(main())

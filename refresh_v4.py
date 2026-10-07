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
                                         # (deteta sozinho o ultimo trimestre completo)
  python3 refresh_v4.py --quarter 2026-Q2   # forca o trimestre a estimar
  python3 refresh_v4.py --no-fetch       # reutiliza cache em data/ se existir
  python3 refresh_v4.py --no-probe       # salta a sonda inicial ao INE (6 pedidos)
  python3 refresh_v4.py --no-flash       # sem estimativa antecipada do trimestre seguinte

Pensado para correr semanalmente via GitHub Actions. Os dados intermedios
ficam em data/ para servir de continuidade entre execucoes.

Protecao: se a deteccao automatica der um trimestre mais antigo do que o do
public/data.json ja publicado (tipicamente porque o INE nao respondeu e a cache
acaba antes), a execucao falha sem escrever nada. --quarter ignora a protecao.

Convencoes INE confirmadas:
  Algarve NUTS II  Dim2=15        aeroporto Faro  Dim2=LPFR
  anual S7A{ano}   trimestral S5A{ano}{trim}   mensal S3A{ano}{mes:02d}
  base 2021 / NUTS 2024, serie consistente desde 1995 (sem splicing manual)
"""

import json, csv, time, urllib.request, urllib.error, argparse, os, sys, threading, atexit, copy
from collections import defaultdict, Counter
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

# Indicadores publicados no data.json: (nome no site, chave interna)
INPUT_KEYS = [("revenue", "revenue"), ("airport", "airport_q"),
              ("unemp", "unemp"), ("wages", "wages"),
              ("htx", "htx"), ("cost", "cost_q")]

# Indicadores mensais que ancoram as pontes. Um trimestre so existe nestas series
# quando tem os 3 meses, por isso sao eles que decidem o trimestre a estimar.
# Os trimestrais lentos (desemprego, salarios, transacoes) podem ser herdados
# do trimestre homologo (ver carry_forward) e ficam assinalados no data.json.
FAST_INPUTS = ["revenue", "cost_q"]


# ----------------------------------------------------------------------------
# Fetch INE (paralelo, com retries)
# ----------------------------------------------------------------------------
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
}

# Diagnostico do INE: o que falha, e quanto tempo cada pedido demora. Nao altera
# o que o pipeline faz, so regista (ver ine_report e ine_probe).
_ERR = Counter()        # classe de erro -> n
_ERR_T = []             # duracao (s) dos pedidos falhados
_OK_T = []              # duracao (s) dos pedidos bem sucedidos
_LOCK = threading.Lock()

def _err_name(e):
    """Classe curta do erro, para contar (sem URLs nem valores)."""
    if isinstance(e, urllib.error.HTTPError):
        ra = e.headers.get("Retry-After") if e.headers else None
        return f"HTTP {e.code}" + (f" (Retry-After {ra})" if ra else "")
    if isinstance(e, urllib.error.URLError):
        return "URLError: " + str(e.reason)[:60]
    if isinstance(e, ValueError):
        return "resposta nao e JSON"        # p.ex. pagina de erro com HTTP 200
    return f"{type(e).__name__}: {str(e)[:60]}"


# Ritmo dos pedidos ao INE. Medido no GitHub (2026-10-07): 6 pedidos sequenciais
# com pausa de 0.5s passaram todos; 16 em paralelo deram 15 x HTTP 429 e, a
# seguir, o INE deixou de responder (timeouts) durante varios minutos.
PACE = 0.5                    # s minimos entre o inicio de dois pedidos
COOLDOWNS = (30, 60, 120)     # s de espera apos a 1a, 2a, 3a+ falha seguida
WAIT_BUDGET = 300             # s de espera total por execucao; esgotado, usa-se a cache
_NET = threading.RLock()      # um pedido de cada vez, mesmo se houver threads
_S = {"last": 0.0, "streak": 0, "waited": 0.0, "since": 0.0, "gave_up": False}


def _fetch(url, timeout=8, tries=3):
    """Um pedido ao INE, com pausa entre pedidos e espera apos falhas.

    Falha transitoria (429, 5xx, timeout, ligacao cortada, resposta que nao e
    JSON): espera 30s, depois 60s, depois 120s, e repete ate `tries` vezes. Uma
    falha clara do pedido (400, 404) devolve None sem esperar. Se a espera total
    passar WAIT_BUDGET, desiste de todos os pedidos seguintes (fica a cache).
    """
    with _NET:
        for attempt in range(tries):
            if _S["gave_up"]:
                return None
            gap = PACE - (time.time() - _S["last"])
            if gap > 0:
                time.sleep(gap)
            _S["last"] = t0 = time.time()
            try:
                req = urllib.request.Request(url, headers=_HEADERS)
                out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            except Exception as e:
                with _LOCK:
                    _ERR[_err_name(e)] += 1
                    _ERR_T.append(time.time() - t0)
                if (isinstance(e, urllib.error.HTTPError)
                        and e.code != 429 and e.code < 500):
                    return None                  # pedido recusado, esperar nao ajuda
                if not _S["streak"]:
                    _S["since"] = t0
                wait = COOLDOWNS[min(_S["streak"], len(COOLDOWNS) - 1)]
                ra = e.headers.get("Retry-After") if isinstance(e, urllib.error.HTTPError) else None
                if ra and str(ra).isdigit():
                    wait = max(wait, min(int(ra), 120))
                _S["streak"] += 1
                if attempt == tries - 1:
                    return None
                if _S["waited"] + wait > WAIT_BUDGET:
                    _S["gave_up"] = True
                    print(f"  AVISO INE: {_S['waited']:.0f}s de espera sem resposta; "
                          f"os pedidos que faltam ficam pela cache")
                    return None
                _S["waited"] += wait
                time.sleep(wait)
                continue
            with _LOCK:
                _OK_T.append(time.time() - t0)
            if _S["streak"]:
                print(f"  INE voltou a responder apos {time.time() - _S['since']:.0f}s "
                      f"({_S['streak']} falhas seguidas)")
                _S["streak"] = 0
            return out
    return None


def ine_report():
    """Resumo dos pedidos ao INE nesta execucao (corre no fim, mesmo se houver erro)."""
    n_ok, n_err = len(_OK_T), len(_ERR_T)
    if not (n_ok or n_err):
        return
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else 0.0
    print(f"INE, resumo dos pedidos: {n_ok} ok, {n_err} falhados")
    if _OK_T:
        print(f"  ok: latencia media {np.mean(_OK_T):.1f}s, p90 {q(_OK_T, .9):.1f}s, "
              f"max {max(_OK_T):.1f}s")
    if _ERR_T:
        print(f"  falhas: duracao media {np.mean(_ERR_T):.1f}s, max {max(_ERR_T):.1f}s")
        for k, v in _ERR.most_common():
            print(f"    {v:4d} x {k}")
    if _S["waited"]:
        print(f"  esperas apos falhas: {_S['waited']:.0f}s no total"
              + ("; desistiu do INE (orcamento esgotado)" if _S["gave_up"] else ""))


def _probe_one(p):
    """Um pedido de teste (receita, Algarve) a um mes ja publicado. (ok, segundos, erro)."""
    url = f"{BASE}?op=2&varcd=0009813&Dim1={p}&Dim2=15&Dim3=T&lang=EN"
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        json.loads(urllib.request.urlopen(req, timeout=20).read())
        return (True, time.time() - t0, "")
    except Exception as e:
        return (False, time.time() - t0, _err_name(e))


def _probe_summary(label, res):
    ok = [t for good, t, _ in res if good]
    errs = Counter(e for good, _, e in res if not good)
    slow = sum(t > 8 for t in ok)                 # o pipeline desiste aos 8s
    s = f"  {label}: {len(ok)}/{len(res)} ok"
    if ok:
        s += f", latencia media {np.mean(ok):.1f}s, max {max(ok):.1f}s"
        if slow:
            s += f" ({slow} acima de 8s)"
    print(s)
    for k, v in errs.most_common():
        print(f"      {v} x {k}")


def ine_probe():
    """Sonda curta ao INE antes de puxar dados: 6 pedidos sequenciais, pausa de
    0.5s, a meses ja publicados. Mostra se o INE responde a esta maquina e a que
    velocidade. A sonda com 16 pedidos em paralelo (fase B) ja cumpriu o papel:
    o INE responde 429 e deixa de responder, por isso nao se repete.
    """
    print(f"Sonda INE ({time.strftime('%H:%M:%S', time.gmtime())} UTC)")
    res = []
    for p in [f"S3A2025{m:02d}" for m in range(1, 7)]:
        res.append(_probe_one(p)); time.sleep(PACE)
    _probe_summary(f"sequencial (pausa {PACE}s)", res)


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
    """Puxa uma serie INE para uma lista de periodos, um de cada vez.

    O INE responde 429 e deixa de responder se receber pedidos em paralelo, por
    isso o ritmo e as esperas ficam em _fetch. Aqui distingue-se o INE responder
    sem valor (periodo ainda nao publicado) de o pedido falhar: um mes perdido
    por falha faria o trimestre parecer incompleto e o nowcast recuar um trimestre.
    """
    def one(p):
        url = f"{BASE}?op=2&varcd={code}&Dim1={p}&Dim2={dim2}{extra}&lang=EN"
        d = _fetch(url)
        if d is None:
            return (p, None, True)                       # pedido falhou
        if isinstance(d, list) and d and "Dados" in d[0]:
            for _, vals in d[0]["Dados"].items():
                for v in vals:
                    if v.get("valor"):
                        return (p, float(v["valor"].replace(",", ".")), False)
        return (p, None, False)                          # respondeu sem valor
    res, failed = {}, []
    for p in periods:
        _, val, errored = one(p)
        if val is not None:
            res[p] = val
        elif errored:
            failed.append(p)
    if failed:
        print(f"  AVISO {label or code}: {len(failed)} periodos sem resposta do INE "
              f"({failed[0]} ...)")
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
def _load_annual_cache(path):
    annual = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            annual[int(row["year"])] = {s: float(row[s]) for s in SECTORS + ["TOT"]}
    return annual


def _split_year(tot, sh):
    modeled = sum(sh.get(s, 0) for s in ["304", "309", "307", "203", "308"])
    rec = {"TOT": tot}
    for s in ["304", "309", "307", "203", "308"]:
        rec[s] = tot * sh.get(s, 0) / 100
    rec["REST"] = tot * (100 - modeled) / 100
    return rec


def _save_annual(annual, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["year"] + SECTORS + ["TOT"])
        for y in sorted(annual):
            w.writerow([y] + [round(annual[y][s], 1) for s in SECTORS + ["TOT"]])


def phase1_annual(fetch=True, now_year=2026):
    """Serie anual de VAB setorial. Incremental: o historico vem da cache,
    so se vao buscar ao INE os anos recentes (que mudam com atraso anual)."""
    path = f"{DATA_DIR}/sector_gva_annual_1995.csv"
    cached = _load_annual_cache(path) if os.path.exists(path) else None
    if not fetch:
        if cached:
            return cached
        raise RuntimeError("--no-fetch mas nao ha cache em " + path)

    print("Fase 1  serie anual")
    rec_years = list(range(now_year - 2, now_year + 1))   # anos recentes a verificar

    def pull_shares(y):
        d = _fetch(f"{BASE}?op=2&varcd=0014109&Dim1=S7A{y}&Dim2=15&lang=EN")
        sh = {}
        if d and "Dados" in d[0]:
            for _, vals in d[0]["Dados"].items():
                for v in vals:
                    if v.get("valor") and v.get("dim_3"):
                        sh[v["dim_3"]] = float(v["valor"].replace(",", "."))
        return sh

    if cached:
        # incremental: so os anos recentes, e so se houver total E quotas validas
        # (anos com total publicado mas quotas ainda em falta mantem a cache)
        print("  incremental: a verificar anos recentes")
        tot_raw = fetch_series("0014113", "150", "&Dim3=TOT",
                               [f"S7A{y}" for y in rec_years], "")
        annual = dict(cached)
        updated = 0
        for y in rec_years:
            tot = tot_raw.get(f"S7A{y}")
            sh = pull_shares(y)
            if tot is not None and sh:
                annual[y] = _split_year(tot, sh)
                updated += 1
        if updated:
            _save_annual(annual, path)
        print(f"  {len(annual)} anos ({updated} atualizados, resto da cache)")
        return annual

    # ---- primeira vez, sem cache: pull completo 1995-2024 ----
    print("  pull completo 1995-2024")
    tot_raw = fetch_series("0014113", "150", "&Dim3=TOT",
                           [f"S7A{y}" for y in range(1995, 2025)], "VAB total")
    gva_tot = {int(p[3:7]): v for p, v in tot_raw.items()}
    shares = {}
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for y, sh in zip(range(1995, 2024), ex.map(pull_shares, range(1995, 2024))):
            if sh:
                shares[y] = sh
    if not gva_tot or not shares:
        raise RuntimeError("INE nao respondeu no pull inicial e nao ha cache")
    if 2023 in shares:
        shares[2024] = dict(shares[2023])
    annual = {y: _split_year(gva_tot[y], shares[y])
              for y in sorted(gva_tot) if y in shares}
    os.makedirs(DATA_DIR, exist_ok=True)
    _save_annual(annual, path)
    print(f"  {len(annual)} anos")
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
def _fetch_monthly(code, dim2, extra, periods, label):
    """Devolve {YYYY-MM: valor} para uma lista de periodos S3A."""
    return {s3a_to_ym(p): v for p, v in fetch_series(code, dim2, extra, periods, label).items()}


def load_indicators(fetch=True, now_year=2026):
    """Devolve indicadores em frequencia trimestral, mais trend.

    Estrategia: a serie historica (2004+) nao muda e vive na cache do repo.
    Quando ha cache, so se vai ao INE buscar a janela recente (ano anterior e
    ano corrente), umas dezenas de chamadas em vez de centenas. Se o INE nao
    responder, fica-se com a cache. Sem cache (primeira vez), faz o pull completo.
    """
    print("Indicadores")
    cache = f"{DATA_DIR}/indicators_cache.json"
    cached = json.load(open(cache)) if os.path.exists(cache) else None

    if not fetch:
        if cached:
            raw = cached
        else:
            raise RuntimeError("--no-fetch mas nao ha cache em " + cache)

    elif cached:
        # ---- modo incremental: so a janela recente ----
        print("  incremental: a buscar so periodos recentes ao INE")
        raw = {k: dict(v) for k, v in cached.items()}
        now_ym = time.strftime("%Y%m")
        rm = [p for p in months(now_year - 1, now_year) if p[3:9] <= now_ym]  # nao pedir meses futuros
        rq = quarters(now_year - 1, now_year)        # ~8 trimestres recentes
        emb = _fetch_monthly("0000861", "LPFR", "&Dim3=T&Dim4=T", rm, "")
        dis = _fetch_monthly("0000862", "LPFR", "&Dim3=T&Dim4=T", rm, "")
        for ym in set(emb) | set(dis):
            raw["airport_m"][ym] = emb.get(ym, 0) + dis.get(ym, 0)
        for ym, v in _fetch_monthly("0011748", "PT", "&Dim3=T", rm, "").items():
            raw["cost_m"][ym] = v
        for ym, v in _fetch_monthly("0009813", "15", "&Dim3=T", rm, "").items():
            raw["revenue_m"][ym] = v
        for p, v in fetch_series("0012136", "15", "&Dim3=T", rq, "").items():
            raw["unemp"][s5a_to_q(p)] = v
        for p, v in fetch_series("0012134", "15", "&Dim3=B-F", rq, "").items():
            raw["wages"][s5a_to_q(p)] = v
        for p, v in fetch_series("0012786", "15", "&Dim3=H1&Dim4=T&Dim5=T", rq, "").items():
            raw["htx"][s5a_to_q(p)] = v
        n_new = len(raw["airport_m"]) - len(cached["airport_m"])
        print(f"  {max(0, n_new)} meses novos de aeroporto; cache atualizada")
        json.dump(raw, open(cache, "w"))

    else:
        # ---- primeira vez, sem cache: pull completo ----
        os.makedirs(DATA_DIR, exist_ok=True)
        emb = _fetch_monthly("0000861", "LPFR", "&Dim3=T&Dim4=T", months(2004, now_year), "aeroporto emb")
        dis = _fetch_monthly("0000862", "LPFR", "&Dim3=T&Dim4=T", months(2004, now_year), "aeroporto dis")
        airport = {ym: emb.get(ym, 0) + dis.get(ym, 0) for ym in set(emb) | set(dis)}
        cost = _fetch_monthly("0011748", "PT", "&Dim3=T", months(2000, now_year), "custo")
        revenue = _fetch_monthly("0009813", "15", "&Dim3=T", months(2017, now_year), "receita")
        unemp = {s5a_to_q(p): v for p, v in fetch_series("0012136", "15", "&Dim3=T", quarters(2011, now_year), "desemprego").items()}
        wages = {s5a_to_q(p): v for p, v in fetch_series("0012134", "15", "&Dim3=B-F", quarters(2011, now_year), "salarios").items()}
        htx = {s5a_to_q(p): v for p, v in fetch_series("0012786", "15", "&Dim3=H1&Dim4=T&Dim5=T", quarters(2009, now_year), "transacoes").items()}
        if not airport or not revenue or not unemp:
            raise RuntimeError("INE nao respondeu no pull inicial e nao ha cache")
        raw = {"airport_m": airport, "cost_m": cost, "revenue_m": revenue,
               "unemp": unemp, "wages": wages, "htx": htx}
        json.dump(raw, open(cache, "w"))

    ind = {
        "airport_q": monthly_to_quarterly(raw["airport_m"], "sum"),
        "cost_q":    monthly_to_quarterly(raw["cost_m"], "mean"),
        "revenue":   monthly_to_quarterly(raw["revenue_m"], "sum"),
        "unemp":     raw["unemp"], "wages": raw["wages"], "htx": raw["htx"],
        # meses soltos: so o trimestre completo entra nas series acima, e o flash
        # precisa dos meses do trimestre ainda incompleto
        "revenue_m": raw["revenue_m"], "cost_m": raw["cost_m"],
    }
    allq = [f"{y}-Q{q}" for y in range(2004, now_year + 1) for q in range(1, 5)]
    ind["trend"] = {q: i for i, q in enumerate(allq)}
    return ind

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
    """Janela expansivel out-of-sample, exclui COVID.
    Devolve (vies, MAE, n, detalhe por trimestre)."""
    errs, detail = [], []
    for ty in range(y0, y1 + 1):
        for q in range(1, 5):
            tq = f"{ty}-Q{q}"
            if tq in COVID:
                continue
            pr = {s: predict_sector(gva, ind, s, tq, f"{ty-1}-Q4") for s in SECTORS}
            if any(v is None for v in pr.values()):
                continue
            actual = sum(gva[s][tq] for s in SECTORS)
            predicted = sum(pr.values())
            errs.append((predicted - actual) / actual * 100)
            detail.append({"quarter": tq, "actual": round(actual, 1),
                           "predicted": round(predicted, 1),
                           "error": round(predicted - actual, 1),
                           "error_pct": round((predicted - actual) / actual * 100, 1)})
    errs = np.array(errs)
    rmse_meur = float(np.sqrt(np.mean([(d["predicted"] - d["actual"]) ** 2
                                       for d in detail]))) if detail else 0.0
    bias = float(np.mean(errs)) if len(errs) else 0.0
    mae = float(np.mean(np.abs(errs))) if len(errs) else 0.0
    return bias, mae, len(errs), detail, rmse_meur


SHRINK_K = 2.0   # peso (em n. de observacoes) do vies global sobre o de cada trimestre

def seasonal_bias(detail, k=SHRINK_K):
    """Vies medio (%) por trimestre do ano, encolhido para o vies global:
        b_q = (soma dos erros do trimestre q + k * vies global) / (n_q + k)
    Com 3 a 4 observacoes por trimestre, o encolhimento evita ajustar ao ruido;
    sem observacoes, b_q cai para o vies global."""
    errs = [d["error_pct"] for d in detail]
    pooled = float(np.mean(errs)) if errs else 0.0
    by_q, n_by_q = {}, {}
    for qn in "1234":
        e = [d["error_pct"] for d in detail if d["quarter"][-1] == qn]
        n_by_q[qn] = len(e)
        by_q[qn] = (sum(e) + k * pooled) / (len(e) + k)
    return by_q, n_by_q, pooled


def loo_mae(detail, k=SHRINK_K):
    """MAE (%) deixando cada trimestre de fora: e corrigido com o vies estimado
    SEM ele. Devolve (MAE com vies global, MAE com vies sazonal)."""
    pooled_err, seas_err = [], []
    for i, d in enumerate(detail):
        by_q, _, pooled = seasonal_bias(detail[:i] + detail[i + 1:], k)
        for bias, out in ((pooled, pooled_err), (by_q[d["quarter"][-1]], seas_err)):
            out.append(abs(d["predicted"] / (1 + bias / 100) / d["actual"] - 1) * 100)
    return float(np.mean(pooled_err)), float(np.mean(seas_err))


def detect_now_q(ind):
    """Ultimo trimestre completo nos indicadores mensais que ancoram as pontes."""
    common = set.intersection(*(set(ind[k]) for k in FAST_INPUTS))
    if not common:
        raise RuntimeError("sem nenhum trimestre completo nos indicadores mensais")
    return max(common)


def carry_forward(ind, now_q):
    """Para indicadores trimestrais sem o trimestre corrente, herda o homologo.
    Devolve {indicador: trimestre de onde o valor foi herdado}."""
    carried = {}
    for name in ["unemp", "wages", "htx", "revenue"]:
        d = ind[name]
        if now_q not in d:
            y, qn = now_q.split("-Q")
            py = f"{int(y)-1}-Q{qn}"
            src = py if py in d else sorted(d)[-1]
            d[now_q] = d[src]
            carried[name] = src
    return carried


# ----------------------------------------------------------------------------
# Estimativa antecipada (flash) do trimestre seguinte ao ultimo completo
# ----------------------------------------------------------------------------
# O INE publica a receita turistica ~30 dias depois do fim do mes e o indice de
# custos ~39. Com 2 meses de receita e 1 de custos, ja se estima o trimestre:
#   receita  total = meses observados / quota media desses meses no trimestre
#            (mesmo trimestre de anos de referencia, sem os anos COVID)
#   custos   o ultimo mes observado prolongado pela variacao mensal media recente
#   lentos   desemprego, salarios e transacoes herdados do trimestre homologo
# O erro de usar este atalho mede-se em trimestres passados (flash_validation).
FLASH_MIN_REV_MONTHS = 2
FLASH_MIN_COST_MONTHS = 1
FLASH_REF_FROM = 2017               # a receita mensal comeca em 2017
FLASH_EXCLUDE_YEARS = {2020, 2021}  # quotas mensais distorcidas pelo COVID
FLASH_COST_WINDOW = 6               # variacoes mensais usadas para prolongar os custos
FLASH_MIN_REFS = 3                  # anos de referencia minimos para a quota
FLASH_VALID_FROM = 2022             # primeiro ano da validacao do atalho
FLASH_SLOW = ("unemp", "wages", "htx")


def _next_q(q):
    y, qn = int(q[:4]), int(q[-1])
    return f"{y + (qn == 4)}-Q{qn % 4 + 1}"


def _q_months(q):
    y, qn = int(q[:4]), int(q[-1])
    return [f"{y}-{m:02d}" for m in range(3 * (qn - 1) + 1, 3 * qn + 1)]


def _leading(ks, monthly):
    """Quantos meses do trimestre existem, contados desde o primeiro."""
    n = 0
    for k in ks:
        if k not in monthly:
            break
        n += 1
    return n


def _revenue_total_est(rev_m, q, n_rev):
    """Receita do trimestre q estimada a partir dos primeiros n_rev meses."""
    ks = _q_months(q)
    obs = sum(rev_m[k] for k in ks[:n_rev])
    y, qn = int(q[:4]), int(q[-1])
    shares = []
    for r in range(FLASH_REF_FROM, max(int(k[:4]) for k in rev_m) + 1):
        if r == y or r in FLASH_EXCLUDE_YEARS:
            continue
        kr = _q_months(f"{r}-Q{qn}")
        if not all(k in rev_m for k in kr):
            continue
        tot = sum(rev_m[k] for k in kr)
        if tot > 0:
            shares.append(sum(rev_m[k] for k in kr[:n_rev]) / tot)
    if len(shares) < FLASH_MIN_REFS:
        return None
    return obs / float(np.mean(shares))


def _cost_q_est(cost_m, q, n_cost):
    """Indice de custos do trimestre q: n_cost meses observados, o resto
    prolongado pela variacao mensal media das ultimas FLASH_COST_WINDOW."""
    ks = _q_months(q)
    last = ks[n_cost - 1]
    hist = sorted(k for k in cost_m if k <= last)
    if len(hist) < FLASH_COST_WINDOW + 1:
        return None
    step = float(np.mean(np.diff([cost_m[k] for k in hist[-(FLASH_COST_WINDOW + 1):]])))
    vals, v = [cost_m[k] for k in ks[:n_cost]], cost_m[last]
    for _ in range(3 - n_cost):
        v += step
        vals.append(v)
    return float(np.mean(vals))


def flash_plan(ind, last_q):
    """Trimestre seguinte a last_q, se ja tem meses suficientes para estimar.
    Devolve None se nao tem, ou se esta completo (nesse caso e o detect_now_q)."""
    fq = _next_q(last_q)
    if fq not in ind["trend"]:           # alem do ano carregado: sem tendencia para as pontes
        return None
    ks = _q_months(fq)
    n_rev, n_cost = _leading(ks, ind["revenue_m"]), _leading(ks, ind["cost_m"])
    if n_rev < FLASH_MIN_REV_MONTHS or n_cost < FLASH_MIN_COST_MONTHS:
        return None
    if n_rev >= 3 and n_cost >= 3:
        return None
    rev = _revenue_total_est(ind["revenue_m"], fq, n_rev)
    cost = _cost_q_est(ind["cost_m"], fq, n_cost)
    if rev is None or cost is None:
        return None
    return {"quarter": fq, "months": ks, "n_rev": n_rev, "n_cost": n_cost,
            "revenue_q": rev, "cost_q": cost,
            "revenue_obs": sum(ind["revenue_m"][k] for k in ks[:n_rev])}


def _total(gva, ind, q):
    vals = [predict_sector(gva, ind, s, q) for s in SECTORS]
    return None if any(v is None for v in vals) else float(sum(vals))


def flash_validation(gva, ind, n_rev, n_cost, last_q):
    """Erro (%) do atalho em trimestres passados: estimativa feita so com os
    primeiros n_rev meses de receita e n_cost de custos (lentos herdados do ano
    anterior) contra a feita com o trimestre completo, no mesmo modelo.
    Mede o custo de faltarem dados, nao a exatidao face ao valor real.
    As quotas de referencia incluem anos posteriores ao trimestre testado."""
    errs = []
    for y in range(FLASH_VALID_FROM, int(last_q[:4]) + 1):
        for qn in range(1, 5):
            q, py = f"{y}-Q{qn}", f"{y-1}-Q{qn}"
            if q > last_q or q in COVID:
                continue
            ks = _q_months(q)
            if not (q in ind["revenue"] and q in ind["cost_q"]
                    and all(k in ind["revenue_m"] and k in ind["cost_m"] for k in ks)
                    and all(q in ind[n] and py in ind[n] for n in FLASH_SLOW)):
                continue
            rev = _revenue_total_est(ind["revenue_m"], q, n_rev)
            cost = _cost_q_est(ind["cost_m"], q, n_cost)
            full = _total(gva, ind, q)
            if rev is None or cost is None or full is None:
                continue
            i2 = copy.deepcopy(ind)
            i2["revenue"][q], i2["cost_q"][q] = rev, cost
            for n in FLASH_SLOW:
                i2[n][q] = i2[n][py]
            fl = _total(gva, i2, q)
            if fl is not None:
                errs.append((fl / full - 1) * 100)
    return errs


# ----------------------------------------------------------------------------
# Protecao da publicacao
# ----------------------------------------------------------------------------
def _published_quarter(path=None):
    """Trimestre do nowcast que esta publicado, ou None se nao ha ficheiro legivel."""
    try:
        with open(path or OUT_JSON) as f:
            return json.load(f).get("nowcast_quarter")
    except (OSError, ValueError, AttributeError):
        return None


def _guard_publication(now_q, path=None):
    """Recusa substituir o data.json publicado por um trimestre mais antigo.
    Quando o INE nao responde, os pedidos que faltam ficam pela cache, que acaba
    antes do trimestre publicado, e a deteccao automatica recua. Escrever isso
    deixava o site pior sem ninguem dar por isso; falhar torna a execucao vermelha.
    Trimestres iguais ou mais recentes passam (ex.: antecipado -> completo)."""
    pub = _published_quarter(path)
    if pub and now_q < pub:
        motivo = ("o INE nao respondeu a tempo (ver resumo dos pedidos abaixo)"
                  if _S["gave_up"] else "faltam dados do INE")
        sys.exit(f"ERRO: a deteccao automatica deu {now_q}, mas o data.json publicado e {pub}. "
                 f"Nao se escreve um trimestre mais antigo ({motivo}). "
                 f"Nada foi escrito. Para forcar de proposito: --quarter {now_q}")


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run(fetch=True, now_q=None, probe=True, flash=True):
    # now_q=None: o trimestre a estimar e detetado a partir dos dados.
    # Ano a puxar do INE: o corrente, ou o do trimestre pedido se for posterior.
    fetch_year = max(time.localtime().tm_year, int(now_q[:4]) if now_q else 0)
    if fetch and probe:
        ine_probe()
    annual = phase1_annual(fetch, fetch_year)
    ind = load_indicators(fetch, fetch_year)
    gva = phase2_disaggregate(annual, ind, fetch)

    auto = now_q is None
    if auto:
        now_q = detect_now_q(ind)
    # ultimo trimestre realmente observado, antes de qualquer heranca ou estimativa
    data_through = {name: max(ind[key]) for name, key in INPUT_KEYS}
    last_q = now_q                     # ultimo trimestre completo
    # Estimativa antecipada do trimestre seguinte, se ja ha meses suficientes
    plan = flash_plan(ind, last_q) if (auto and flash) else None
    if plan:
        now_q = plan["quarter"]
        ind["revenue"][now_q], ind["cost_q"][now_q] = plan["revenue_q"], plan["cost_q"]
    now_year = int(now_q[:4])
    carried = carry_forward(ind, now_q)
    if plan:
        print(f"Trimestre a estimar: {now_q} (ANTECIPADO; ultimo completo: {last_q})")
        print(f"  receita: {plan['n_rev']}/3 meses observados, trimestre estimado em "
              f"{plan['revenue_q']:,.0f}; custos: {plan['n_cost']}/3 meses, "
              f"indice estimado {plan['cost_q']:.1f}")
    else:
        print(f"Trimestre a estimar: {now_q} ({'detetado' if auto else 'forcado'})")
    for name, src in carried.items():
        print(f"  AVISO: {name} sem {now_q}, herdado de {src}")
    if auto:
        _guard_publication(now_q)

    print("Fase 3 e 4  pontes, nowcast, backtest")
    bias, mae, n, bt_detail, bt_rmse = backtest(gva, ind)
    # Correcao de vies por trimestre do ano (nao um fator unico): o backtest mostra
    # erros muito diferentes por trimestre (T2 e T4 muito abaixo, T1 perto de zero).
    bias_q, n_q, _ = seasonal_bias(bt_detail)
    mae_loo_pooled, mae_loo_seasonal = loo_mae(bt_detail)
    corr_of = lambda q: 1 / (1 + bias_q[q[-1]] / 100)
    corr = corr_of(now_q)              # fator de correcao do trimestre a estimar
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

    # Intervalo a 90%: erro do modelo; no flash, soma-se (em quadratura) o erro de
    # estimar com dados em falta, medido em trimestres passados.
    half_90, flash_info = 1.645 * rmse_agg, None
    if plan:
        errs = np.array(flash_validation(gva, ind, plan["n_rev"], plan["n_cost"], last_q))
        enough = len(errs) >= 8
        rms = float(np.sqrt(np.mean(errs ** 2))) if enough else 3.0   # sem amostra, assume 3%
        flash_half = 1.645 * rms / 100 * tot * corr
        half_90 = float(np.sqrt((1.645 * rmse_agg) ** 2 + flash_half ** 2))
        # ultimo trimestre completo, para mostrar ao lado do antecipado
        lpy = f"{int(last_q[:4]) - 1}-Q{last_q[-1]}"
        tot_l, tp_l = _total(gva, ind, last_q), _total(gva, ind, lpy)
        last_complete = None
        if tot_l and tp_l:
            cl = corr_of(last_q)
            last_complete = {
                "quarter": last_q,
                "gva_meur": round(tot_l, 1), "gva_corrected_meur": round(tot_l * cl, 1),
                "gdp_meur": round(tot_l * GDP_FACTOR, 1),
                "gdp_corrected_meur": round(tot_l * cl * GDP_FACTOR, 1),
                "yoy_pct": round((tot_l / tp_l - 1) * 100, 1),
                "lower_90": round(tot_l * cl - 1.645 * rmse_agg, 1),
                "upper_90": round(tot_l * cl + 1.645 * rmse_agg, 1),
                "bias_correction_pct": round(bias_q[last_q[-1]], 1),
            }
        py_rev = ind["revenue"].get(py_q)
        flash_info = {
            "quarter": now_q, "last_complete_quarter": last_q,
            "months_observed": {"revenue": plan["months"][:plan["n_rev"]],
                                "cost": plan["months"][:plan["n_cost"]]},
            "estimated": {
                "revenue_quarter": round(plan["revenue_q"]),
                "revenue_months_missing": round(plan["revenue_q"] - plan["revenue_obs"]),
                "revenue_yoy_pct": round((plan["revenue_q"] / py_rev - 1) * 100, 1) if py_rev else None,
                "cost_quarter": round(plan["cost_q"], 1)},
            "validation": {
                "n": int(len(errs)), "rms_pct": round(rms, 1),
                "mae_pct": round(float(np.mean(np.abs(errs))), 1) if len(errs) else None,
                "bias_pct": round(float(np.mean(errs)), 1) if len(errs) else None,
                "worst_pct": round(float(errs[np.argmax(np.abs(errs))]), 1) if len(errs) else None,
                "assumed_rms": not enough,
                "note": "estimativa so com os meses disponiveis contra a do trimestre completo, "
                        "mesmo modelo, trimestres passados; mede o custo dos dados em falta, "
                        "nao a exatidao face ao valor real"},
            "interval": {"model_half_meur": round(1.645 * rmse_agg, 1),
                         "flash_half_meur": round(flash_half, 1),
                         "total_half_meur": round(half_90, 1)},
            "last_complete": last_complete,
        }

    # Serie agregada para o grafico: observado (soma de setores) + previsao 2025-2026
    gva_quarterly = {}
    for q in sorted(gva["304"]):
        if q >= "2017-Q1":
            gva_quarterly[q] = {"type": "actual",
                                "value": round(sum(gva[s][q] for s in SECTORS), 1)}
    fc_quarters = [f"{y}-Q{q}" for y in range(DISAGG_END + 1, now_year) for q in range(1, 5)]
    fc_quarters += [f"{now_year}-Q{q}" for q in range(1, int(now_q[-1]) + 1)]
    for q in fc_quarters:
        vals = [predict_sector(gva, ind, s, q) for s in SECTORS]
        if all(v is not None for v in vals):
            gva_quarterly[q] = {"type": "forecast",
                                "value": round(sum(vals) * corr_of(q), 1)}

    if plan and now_q in gva_quarterly:
        gva_quarterly[now_q]["type"] = "flash"

    data = {
        "updated": time.strftime("%Y-%m-%d"),
        "nowcast_quarter": now_q,
        "nowcast_status": "flash" if plan else "complete",
        "nowcast_quarter_source": "auto" if auto else "manual",
        "data_through": data_through,
        "inputs_carried_forward": carried,
        "version": "Algarve Nowcast v4 Setorial",
        "aggregate": {
            "gva_meur": round(tot, 1), "gva_corrected_meur": round(tot * corr, 1),
            "gdp_meur": round(tot * GDP_FACTOR, 1),
            "gdp_corrected_meur": round(tot * corr * GDP_FACTOR, 1),
            "yoy_pct": round((tot / tp - 1) * 100, 1),
            "rmse_meur": round(rmse_agg, 1),
            "lower_90": round(tot * corr - half_90, 1),
            "upper_90": round(tot * corr + half_90, 1),
            "bias_correction_pct": round(bias_q[now_q[-1]], 1),   # o aplicado neste trimestre
            "bias_correction_pooled_pct": round(bias, 1),         # o antigo fator unico
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
            "bias_by_quarter": {qn: round(v, 1) for qn, v in bias_q.items()},
            "n_by_quarter": n_q, "shrinkage_k": SHRINK_K,
            "mae_loo_pct_pooled": round(mae_loo_pooled, 1),
            "mae_loo_pct_seasonal": round(mae_loo_seasonal, 1),
            "rmse_meur": round(bt_rmse, 1), "n_quarters": n,
            "method": "Janela expansivel out-of-sample 2019-2024, exclui COVID",
            "detail": bt_detail,
        },
        **({"flash": flash_info} if plan else {}),
        "gva_quarterly": gva_quarterly,
        "sector_quarterly_gva": {
            s: {q: round(gva[s][q], 1) for q in sorted(gva[s])} for s in SECTORS},
        "indicators": {
            name: {q: round(ind[key][q], 1) for q in sorted(ind[key])
                   if "2017-Q1" <= q <= min(now_q, data_through[name])}
            for name, key in INPUT_KEYS},
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(data, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)

    print(f"\nNowcast {now_q}{' (antecipado)' if plan else ''}: VAB {tot:.0f}M (corr {tot*corr:.0f}M), "
          f"PIB {tot*GDP_FACTOR:.0f}M (corr {tot*corr*GDP_FACTOR:.0f}M), "
          f"homologa {(tot/tp-1)*100:+.1f}%")
    if plan:
        v = flash_info["validation"]
        print(f"  IC 90%: [{tot*corr-half_90:.0f}M ; {tot*corr+half_90:.0f}M]  "
              f"(modelo ±{1.645*rmse_agg:.0f}M, dados em falta ±{flash_half:.0f}M)")
        print(f"  atalho validado em {v['n']} trimestres: vies {v['bias_pct']:+.1f}%, "
              f"MAE {v['mae_pct']:.1f}%, pior {v['worst_pct']:+.1f}%")
    print(f"Backtest: vies global {bias:+.1f}%, MAE {mae:.1f}% (n={n})")
    print("  vies por trimestre (encolhido): " +
          ", ".join(f"T{qn} {bias_q[qn]:+.1f}% (n={n_q[qn]})" for qn in "1234"))
    print(f"  MAE fora da amostra: fator unico {mae_loo_pooled:.1f}% "
          f"vs sazonal {mae_loo_seasonal:.1f}%")
    print(f"Escrito: {OUT_JSON}")
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true",
                    help="reutiliza cache em data/ em vez de puxar do INE")
    ap.add_argument("--quarter", default=None,
                    help="trimestre a estimar, ex. 2026-Q2 (por omissao, deteta o ultimo completo)")
    ap.add_argument("--no-flash", action="store_true",
                    help="nao faz a estimativa antecipada do trimestre seguinte ao ultimo completo")
    ap.add_argument("--no-probe", action="store_true",
                    help="nao faz a sonda inicial ao INE (~30 pedidos de teste)")
    args = ap.parse_args()
    atexit.register(ine_report)      # resumo dos pedidos ao INE, mesmo se a execucao falhar
    run(fetch=not args.no_fetch, now_q=args.quarter, probe=not args.no_probe,
        flash=not args.no_flash)

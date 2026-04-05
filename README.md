# Algarve Economy Nowcast

Real-time GDP nowcast for the Algarve region, Portugal. Sectoral decomposition model using official INE statistics.

**Live site**: *(deploy to get your URL)*

## Architecture

```
┌─────────────────┐      weekly cron       ┌──────────────┐
│  INE API        │ ──────────────────────► │ GitHub       │
│  Google Trends  │   scripts/refresh.py   │ Actions      │
└─────────────────┘                        └──────┬───────┘
                                                  │ commit
                                                  ▼
                                           ┌──────────────┐
                                           │ public/      │
                                           │  index.html  │ ──► Vercel CDN
                                           │  data.json   │
                                           └──────────────┘
```

No backend. No build step. The Python script pulls data and runs models weekly. The HTML dashboard reads `data.json` on page load. Vercel serves static files.

## Deploy

### Option A: Vercel (recommended)

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com), import the repo
3. It auto-detects the `vercel.json` config
4. Done — site is live at `your-project.vercel.app`

### Option B: Any static host

Copy the `public/` folder to any web server (Netlify, GitHub Pages, S3, etc). The only requirement is that `data.json` is served alongside `index.html`.

### Enable auto-refresh

The GitHub Actions workflow (`.github/workflows/refresh.yml`) runs every Monday at 08:00 UTC. It:

1. Pulls latest data from INE API
2. Reruns Chow-Lin disaggregation + bridge equations
3. Writes updated `public/data.json`
4. Commits and pushes (triggers Vercel redeploy)

To trigger manually: Go to Actions → "Refresh Nowcast Data" → "Run workflow".

## Local development

```bash
# Install Python dependencies
pip install numpy pandas scipy scikit-learn requests

# Run the refresh script
python scripts/refresh.py

# Serve locally
cd public && python -m http.server 8000
# Open http://localhost:8000
```

## Model summary

6 sectors modelled individually, aggregated to total GVA:

| Sector | % GVA | Method | R²adj |
|--------|-------|--------|-------|
| Trade/Tourism (304) | 40% | nights + RevPAR + AL share | 0.927 |
| Public Admin (309) | 15% | linear trend | 0.963 |
| Real Estate (307) | 14% | housing transactions + trend | 0.923 |
| Consultancy (308) | 8% | tourism proxy | 0.852 |
| Construction (203) | 6% | building permits | 0.281 |
| Other sectors | 16% | linear trend | 0.829 |

Chow-Lin temporal disaggregation: 8 annual observations (2017–2024) → 32 quarterly.

## Data sources

All from INE (Statistics Portugal) public API:

- `0014109` — GVA sector shares (annual, 1995–2023)
- `0014113` — Total GVA (annual, 1995–2024)
- `0009808` — Overnight stays by segment (monthly, 2017+)
- `0009813` — Total revenue NUTS-2013 (monthly, 2017+)
- `0012098` — Building permits (monthly, 2017+)
- `0012786` — Housing transactions (quarterly, 2017+)
- `0012136` — Unemployment rate (quarterly, 2017+)
- `0000861/62` — Faro airport passengers (monthly, 2017+)

## License

Data: INE open data. Code: MIT.

#!/usr/bin/env python3
"""web_server.py — OilEnergy web server.

A self-contained HTTP server (stdlib only, no Flask/Django required) that
exposes the OilEnergy forecasting pipeline over a browser-friendly web UI
and a JSON REST API.

Usage:
  python web_server.py                          # http://localhost:8080
  python web_server.py --port 5000
  python web_server.py --host 0.0.0.0 --port 8080   # Docker / cloud

Endpoints:
  GET  /                → Web UI dashboard
  GET  /api/commodities → JSON list of available commodities
  POST /api/forecast    → Run forecast; body: JSON or form-encoded parameters
  GET  /api/audit       → Latest audit JSON files
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from oilenergy import run_pipeline
from oilenergy.commodities import list_commodities
from oilenergy.pipeline import AVAILABLE_MODELS

# ---------------------------------------------------------------------------
# HTML template — single-page app served at GET /
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OilEnergy Forecast</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:system-ui,sans-serif;background:#0f1117;color:#e0e0e0;min-height:100vh}}
    header{{background:#1a1d27;padding:16px 24px;border-bottom:1px solid #2a2d3e;display:flex;align-items:center;gap:12px}}
    header h1{{font-size:1.3rem;font-weight:700;color:#fff}}
    header span{{font-size:0.8rem;color:#888;margin-left:8px}}
    .container{{display:grid;grid-template-columns:280px 1fr;gap:0;height:calc(100vh - 57px)}}
    .sidebar{{background:#13151f;padding:20px;overflow-y:auto;border-right:1px solid #2a2d3e}}
    .main{{padding:24px;overflow-y:auto;display:flex;flex-direction:column;gap:20px}}
    label{{display:block;font-size:0.78rem;color:#aaa;margin-bottom:4px;margin-top:12px;text-transform:uppercase;letter-spacing:.05em}}
    select,input[type=number]{{width:100%;padding:7px 10px;background:#1e2130;border:1px solid #3a3d50;border-radius:6px;color:#e0e0e0;font-size:0.9rem}}
    .checks{{display:flex;flex-direction:column;gap:6px;margin-top:4px}}
    .checks label{{font-size:0.85rem;text-transform:none;letter-spacing:0;color:#ccc;display:flex;align-items:center;gap:8px;margin-top:0;cursor:pointer}}
    button{{margin-top:18px;width:100%;padding:10px;background:#2563eb;border:none;border-radius:8px;color:#fff;font-size:0.95rem;font-weight:600;cursor:pointer}}
    button:hover{{background:#1d4ed8}}
    button:disabled{{background:#374151;cursor:not-allowed}}
    .card{{background:#13151f;border:1px solid #2a2d3e;border-radius:10px;padding:20px}}
    .card h2{{font-size:0.9rem;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px}}
    .metrics{{display:flex;gap:16px;flex-wrap:wrap}}
    .metric{{background:#1a1d27;border-radius:8px;padding:12px 18px;flex:1;min-width:120px}}
    .metric .label{{font-size:0.72rem;color:#888;text-transform:uppercase;letter-spacing:.05em}}
    .metric .value{{font-size:1.4rem;font-weight:700;color:#60a5fa;margin-top:2px}}
    table{{width:100%;border-collapse:collapse;font-size:0.85rem}}
    th{{text-align:left;padding:8px 12px;background:#1a1d27;color:#aaa;font-weight:600;font-size:0.75rem;text-transform:uppercase}}
    td{{padding:8px 12px;border-bottom:1px solid #1e2130;color:#ccc}}
    td.up{{color:#34d399}}td.down{{color:#f87171}}
    .badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:600;background:#1e3a5f;color:#60a5fa}}
    .badge.proxy{{background:#3b2a1a;color:#fb923c}}
    #chart-canvas{{width:100%;height:320px;background:#1a1d27;border-radius:8px;margin-top:4px}}
    .summary{{font-size:0.85rem;line-height:1.6;color:#ccc;white-space:pre-wrap}}
    .err{{color:#f87171;font-size:0.9rem;padding:12px;background:#2d1515;border-radius:8px;border:1px solid #7f1d1d}}
    #spinner{{display:none;margin-top:16px;text-align:center;color:#888;font-size:0.85rem}}
  </style>
</head>
<body>
  <header>
    <h1>⛽ OilEnergy Forecast</h1>
    <span>Middle East Energy Pipeline</span>
  </header>
  <div class="container">
    <aside class="sidebar">
      <label>Category</label>
      <select id="category" onchange="updateCommodities()">
        <option value="oil">Oil</option>
        <option value="gas">Gas</option>
      </select>

      <label>Commodity</label>
      <select id="commodity"></select>

      <label>Model</label>
      <select id="model">{model_options}</select>

      <label>Horizon (days)</label>
      <input type="number" id="horizon" value="5" min="1" max="30">

      <label>Feature Groups</label>
      <div class="checks">
        <label><input type="checkbox" id="ft_seasonality" checked> Seasonality</label>
        <label><input type="checkbox" id="ft_weather"> Weather (Doha temp)</label>
        <label><input type="checkbox" id="ft_demand"> Demand (CDD/HDD proxy)</label>
        <label><input type="checkbox" id="ft_external"> External correlation</label>
      </div>

      <label style="margin-top:14px">Options</label>
      <div class="checks">
        <label><input type="checkbox" id="demo_fallback"> Allow demo fallback</label>
      </div>

      <button id="run-btn" onclick="runForecast()">Run Forecast</button>
      <div id="spinner">Running forecast…</div>
    </aside>

    <main class="main" id="main-panel">
      <div class="card">
        <h2>Welcome</h2>
        <p style="color:#888;font-size:.85rem">
          Select a commodity and model on the left, then click <strong>Run Forecast</strong>
          to generate a price forecast with audit trail.
        </p>
      </div>
    </main>
  </div>

  <script>
    const COMMODITIES = {commodities_json};

    function updateCommodities() {{
      const cat = document.getElementById('category').value;
      const sel = document.getElementById('commodity');
      const opts = COMMODITIES.filter(c => c.type === cat);
      sel.innerHTML = opts.map(c =>
        `<option value="${{c.key}}">${{c.name}}${{c.is_proxy ? ' (proxy)' : ''}}</option>`
      ).join('');
    }}

    function runForecast() {{
      const btn = document.getElementById('run-btn');
      btn.disabled = true;
      document.getElementById('spinner').style.display = 'block';

      const features = ['base'];
      if (document.getElementById('ft_seasonality').checked) features.push('seasonality');
      if (document.getElementById('ft_weather').checked) features.push('weather');
      if (document.getElementById('ft_demand').checked) features.push('demand');
      if (document.getElementById('ft_external').checked) features.push('external');

      const payload = {{
        commodity: document.getElementById('commodity').value,
        category: document.getElementById('category').value,
        model_name: document.getElementById('model').value,
        horizon_days: parseInt(document.getElementById('horizon').value),
        features: features.join(','),
        allow_demo_fallback: document.getElementById('demo_fallback').checked,
      }};

      fetch('/api/forecast', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload),
      }})
      .then(r => r.json())
      .then(data => renderResults(data))
      .catch(err => renderError(String(err)))
      .finally(() => {{
        btn.disabled = false;
        document.getElementById('spinner').style.display = 'none';
      }});
    }}

    function renderResults(data) {{
      if (data.error) {{ renderError(data.error); return; }}
      const m = data.model_audit;
      const fc = m.multi_day_forecast || [];
      const interp = data.interpretation || {{}};

      const proxyBadge = m.is_proxy_commodity
        ? `<span class="badge proxy">proxy for: ${{m.proxy_for}}</span>`
        : '<span class="badge">real data</span>';

      const metricsHtml = `
        <div class="metric"><div class="label">MAE</div><div class="value">${{m.test_metrics.mae.toFixed(4)}}</div></div>
        <div class="metric"><div class="label">RMSE</div><div class="value">${{m.test_metrics.rmse.toFixed(4)}}</div></div>
        <div class="metric"><div class="label">Dir. Accuracy</div><div class="value">${{(m.test_metrics.directional_accuracy*100).toFixed(1)}}%</div></div>
        <div class="metric"><div class="label">Train samples</div><div class="value">${{m.train_sample_count}}</div></div>
      `;

      const fcRows = fc.map(r => {{
        const dir = r.predicted_direction_up_vs_previous_day;
        return `<tr>
          <td>${{r.step_day}}</td>
          <td>${{r.forecast_date}}</td>
          <td>${{r.predicted_price.toFixed(4)}}</td>
          <td class="${{dir?'up':'down'}}">${{dir?'↑ up':'↓ down'}}</td>
        </tr>`;
      }}).join('');

      const summary = (interp.summary || '').trim();

      const panel = document.getElementById('main-panel');
      panel.innerHTML = `
        <div class="card">
          <h2>${{m.commodity_name}} — ${{m.model_name}} &nbsp; ${{proxyBadge}}</h2>
          <div class="metrics">${{metricsHtml}}</div>
        </div>
        <div class="card">
          <h2>Multi-day Forecast</h2>
          <canvas id="chart-canvas"></canvas>
          <table style="margin-top:16px">
            <thead><tr><th>Day</th><th>Date</th><th>Forecast Price</th><th>Direction</th></tr></thead>
            <tbody>${{fcRows}}</tbody>
          </table>
        </div>
        ${{summary ? `<div class="card"><h2>AI Interpretation</h2><div class="summary">${{summary}}</div></div>` : ''}}
      `;

      drawChart(fc, m.latest_prediction);
    }}

    function drawChart(fc, latest) {{
      const canvas = document.getElementById('chart-canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.offsetWidth * dpr;
      canvas.height = canvas.offsetHeight * dpr;
      ctx.scale(dpr, dpr);
      const W = canvas.offsetWidth, H = canvas.offsetHeight;
      const pad = {{t:20,r:20,b:40,l:60}};

      const prices = [latest.latest_observation_price, ...fc.map(f=>f.predicted_price)];
      const labels = [latest.latest_observation_date, ...fc.map(f=>f.forecast_date)];
      const minP = Math.min(...prices) * 0.995;
      const maxP = Math.max(...prices) * 1.005;

      const xStep = (W - pad.l - pad.r) / (prices.length - 1);
      const yScale = (H - pad.t - pad.b) / (maxP - minP);

      function toX(i) {{ return pad.l + i * xStep; }}
      function toY(p) {{ return H - pad.b - (p - minP) * yScale; }}

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#1a1d27';
      ctx.fillRect(0, 0, W, H);

      // Grid lines
      ctx.strokeStyle = '#2a2d3e'; ctx.lineWidth = 1;
      for (let i=0; i<prices.length; i++) {{
        const x = toX(i);
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H-pad.b); ctx.stroke();
      }}

      // Price line
      ctx.strokeStyle = '#60a5fa'; ctx.lineWidth = 2;
      ctx.beginPath();
      prices.forEach((p, i) => {{ i===0 ? ctx.moveTo(toX(i),toY(p)) : ctx.lineTo(toX(i),toY(p)); }});
      ctx.stroke();

      // Points
      prices.forEach((p, i) => {{
        ctx.fillStyle = i===0 ? '#60a5fa' : (fc[i-1]?.predicted_direction_up_vs_previous_day ? '#34d399' : '#f87171');
        ctx.beginPath(); ctx.arc(toX(i), toY(p), 4, 0, 2*Math.PI); ctx.fill();
      }});

      // Labels
      ctx.fillStyle = '#888'; ctx.font = '11px system-ui'; ctx.textAlign = 'center';
      labels.forEach((l, i) => {{ ctx.fillText(l, toX(i), H-pad.b+16); }});
      ctx.textAlign = 'right';
      prices.forEach((p, i) => {{
        if (i % Math.max(1, Math.floor(prices.length/4)) === 0) {{
          ctx.fillStyle = '#888'; ctx.fillText(p.toFixed(2), pad.l-6, toY(p)+4);
        }}
      }});
    }}

    function renderError(msg) {{
      document.getElementById('main-panel').innerHTML =
        `<div class="err">Error: ${{msg}}</div>`;
    }}

    // Init commodity dropdown
    updateCommodities();
  </script>
</body>
</html>
"""


def _build_html() -> str:
    commodities = list_commodities()
    model_options = "\n".join(
        f'        <option value="{k}">{k} — {v}</option>'
        for k, v in sorted(AVAILABLE_MODELS.items())
    )
    return _HTML.format(
        model_options=model_options,
        commodities_json=json.dumps(commodities),
    )


def _json_response(handler: BaseHTTPRequestHandler, data: dict, status: int = 200) -> None:
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, html: str) -> None:
    body = html.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class ForecastHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.address_string()}] {fmt % args}")

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/":
            _html_response(self, _build_html())
        elif path == "/api/commodities":
            _json_response(self, {"commodities": list_commodities()})
        elif path == "/api/audit":
            audits_dir = PROJECT_ROOT / "audits"
            result: dict = {}
            for name in ("data_audit.json", "model_audit.json", "correlation_audit.json"):
                fp = audits_dir / name
                if fp.exists():
                    result[name] = json.loads(fp.read_text(encoding="utf-8"))
            _json_response(self, result)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path != "/api/forecast":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        ct = self.headers.get("Content-Type", "")

        try:
            if "application/json" in ct:
                params = json.loads(raw.decode("utf-8"))
            else:
                parsed = urllib.parse.parse_qs(raw.decode("utf-8"))
                params = {k: v[0] for k, v in parsed.items()}
        except Exception as exc:
            _json_response(self, {"error": f"Bad request body: {exc}"}, status=400)
            return

        try:
            result = run_pipeline(
                project_root=PROJECT_ROOT,
                commodity=str(params.get("commodity", "brent")),
                category=str(params.get("category", "oil")),
                model_name=str(params.get("model_name", "ridge")),
                horizon_days=int(params.get("horizon_days", 5)),
                features=str(params.get("features", "base,seasonality")),
                allow_demo_fallback=bool(params.get("allow_demo_fallback", False)),
            )
            _json_response(self, result)
        except Exception as exc:
            _json_response(self, {"error": str(exc)}, status=500)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OilEnergy web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    args = parser.parse_args(argv)

    server = HTTPServer((args.host, args.port), ForecastHandler)
    print(f"OilEnergy web server running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()

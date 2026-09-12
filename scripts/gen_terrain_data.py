"""Gera terrain.js: grade de elevacao real e curvas de nivel para desenho.

Fonte: Open-Elevation (SRTM). Area = cercamento eletronico padrao do ResgatAr.
A rota de varredura NAO vem daqui: ela e gerada em app.js a partir desta grade,
porque depende do espacamento entre faixas (funcao da altura de voo e da camera).
"""
import json
import time

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAT_MIN, LAT_MAX = -29.3450, -29.3300
LNG_MIN, LNG_MAX = -52.0100, -51.9900
NX, NY = 24, 20

OUT = r"C:\Users\jefer\ResgatAr\terrain.js"


def fetch_grid():
    lats = np.linspace(LAT_MAX, LAT_MIN, NY)  # linha 0 = norte
    lngs = np.linspace(LNG_MIN, LNG_MAX, NX)
    locations = [
        {"latitude": round(float(la), 5), "longitude": round(float(lo), 5)}
        for la in lats
        for lo in lngs
    ]
    url = "https://api.open-elevation.com/api/v1/lookup"
    elevations = []
    for i in range(0, len(locations), 200):
        batch = locations[i : i + 200]
        for _ in range(4):
            try:
                r = requests.post(url, json={"locations": batch}, timeout=60)
                r.raise_for_status()
                elevations.extend([p["elevation"] for p in r.json()["results"]])
                break
            except Exception as e:
                print("retry", e)
                time.sleep(3)
        else:
            raise RuntimeError("falha no lote " + str(i))
    return np.array(elevations, dtype=float).reshape(NY, NX)


def smooth(g, iters=2):
    for _ in range(iters):
        pad = np.pad(g, 1, mode="edge")
        g = (
            pad[1:-1, 1:-1] * 4
            + pad[:-2, 1:-1]
            + pad[2:, 1:-1]
            + pad[1:-1, :-2]
            + pad[1:-1, 2:]
        ) / 8.0
    return g


def px_to_latlng(px, py):
    lng = LNG_MIN + (px / (NX - 1)) * (LNG_MAX - LNG_MIN)
    lat = LAT_MAX - (py / (NY - 1)) * (LAT_MAX - LAT_MIN)
    return round(lat, 6), round(lng, 6)


def seg_length_px(seg):
    d = np.diff(seg, axis=0)
    return float(np.sum(np.sqrt((d ** 2).sum(axis=1))))


grid = fetch_grid()
grid_s = smooth(grid, iters=3)
vmin, vmax = float(grid_s.min()), float(grid_s.max())
print("elevacao", round(vmin), "a", round(vmax))

X, Y = np.meshgrid(np.arange(NX), np.arange(NY))

# --- curvas de nivel para desenho (mais niveis, visual) ---
draw_levels = np.linspace(vmin + (vmax - vmin) * 0.05, vmax - (vmax - vmin) * 0.05, 11)
cs_draw = plt.contour(X, Y, grid_s, levels=draw_levels)
contours = []
for level, segs in zip(draw_levels, cs_draw.allsegs):
    for seg in segs:
        if len(seg) < 3:
            continue
        pts = [px_to_latlng(px, py) for px, py in seg[::2]]
        if len(pts) < 3:
            continue
        contours.append({"elev": round(float(level)), "pts": pts})
print("curvas para desenho:", len(contours))

terrain = {
    "latMin": LAT_MIN,
    "latMax": LAT_MAX,
    "lngMin": LNG_MIN,
    "lngMax": LNG_MAX,
    "nx": NX,
    "ny": NY,
    "elevMin": round(vmin),
    "elevMax": round(vmax),
    "grid": [[round(float(v), 1) for v in row] for row in grid_s],
}

header = (
    "// Dados de terreno reais da area de busca do ResgatAr.\n"
    "// Fonte: Open-Elevation (SRTM), grade %dx%d sobre o cercamento eletronico padrao.\n"
    "// Gerado por scripts/gen_terrain_data.py. Nao editar a mao.\n\n" % (NX, NY)
)

body = (
    "const MISSION_TERRAIN = " + json.dumps(terrain, separators=(",", ":")) + ";\n\n"
    "const TERRAIN_CONTOURS = " + json.dumps(contours, separators=(",", ":")) + ";\n"
)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(header + body)

print("gravado", OUT)

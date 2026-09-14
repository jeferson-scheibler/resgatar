"""Prepara dados de treino para o estagio de gesto ("socorro" x "sem_socorro") do ResgatAr.

Fonte: voos proprios com um DJI Flip sobre o campus da Univates (Lajeado, RS) em 14/09/2026,
gimbal a -90 graus (e -61 graus no voo obliquo), entre 13 e 37 m acima do solo, com pessoas
fazendo o gesto escolhido (bracos abertos na horizontal, o "T") e as posturas que NAO sao
socorro: em pe, caminhando, sentado, deitado, um braco so erguido, acenando.

Os videos (4K, HEVC, 3 GB) ficam em videos-drone/ e nao sao versionados. Os ZIPs em
dataset-gesto/ sao o produto final e sao suficientes para treinar.

Tres decisoes que definem a qualidade do resultado:

1. O recorte tem TAMANHO FIXO NO SOLO (cerca de 6 m de lado), nao relativo ao tamanho da
   pessoa. Se o recorte acompanhasse o vulto, uma pessoa de bracos abertos (1,7 m) receberia
   um recorte tres vezes maior que uma de bracos caidos (0,5 m), e o modelo aprenderia escala
   do piso em vez de postura. A altura de cada quadro vem da telemetria que o DJI grava dentro
   do MP4 (trilha "djmd", protobuf), o que permite converter metros em pixels quadro a quadro.
2. A pessoa e localizada por um detector generico (YOLO11, classe "person") em janela local
   ao redor da ultima posicao, com varredura completa periodica. Quadros sem deteccao sao
   descartados, nunca interpolados.
3. Os negativos vem das MESMAS pessoas, no MESMO piso e nas MESMAS alturas dos positivos, e
   incluem gestos parecidos com o T (um braco erguido, aceno). Sem isso, o modelo separaria
   "tem pessoa" de "nao tem" e qualquer caminhante viraria alerta. Uma parcela dos negativos
   sai dos proprios quadros dos positivos, em regiao sem a pessoa, para a cena (faixa de
   pedestres, luz) nao virar atalho: medido com um classificador linear so de cor, o atalho
   caiu de 87,5% para o valor registrado em dataset-gesto/LEIA-ME.md.

Uso:
    python scripts/build_gesture_dataset.py            # tudo (extrai, rastreia, recorta)
    python scripts/build_gesture_dataset.py recortar   # so recorta a partir dos rastros em cache

Dependencias: ffmpeg/ffprobe no PATH, opencv-python, ultralytics (baixa yolo11m.pt na 1a vez).
"""
import json
import math
import random
import shutil
import struct
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
VIDEOS = RAIZ / "videos-drone"
BRUTOS = RAIZ / "dados-brutos"
QUADROS = BRUTOS / "quadros-gesto"
RASTROS = BRUTOS / "rastros-gesto"
OUT = RAIZ / "dataset-gesto"

FPS_EXTRACAO = 2          # quadros por segundo extraidos do video
HFOV_GRAUS = 74.4         # campo de visao horizontal do DJI Flip em 16:9 (82,1 graus na diagonal)
TILE = 224
LADO_SOLO_M = (5.0, 7.5)  # lado do recorte no solo, sorteado nesse intervalo
JITTER_M = 0.8            # deslocamento maximo do centro, para o modelo nao exigir centralizacao
GANHO = (0.75, 1.25)      # ganho de brilho sorteado por recorte, nas duas classes (ver abaixo)
MAX_POR_CLASSE = 360
VAZIOS_POR_QUADRO = 1     # recortes de terreno vazio por quadro do voo sem pessoa
# Composicao dos negativos: mesma cena dos positivos (sem a pessoa), pessoas em outras posturas
# e terreno vazio. A parcela "mesma cena" existe para o modelo nao aprender a faixa de
# pedestres e a luz do voo dos positivos como se fossem o gesto.
MISTURA_NEGATIVOS = {"mesma_cena": 0.35, "pessoa": 0.55, "vazio": 0.10}

random.seed(23)

# Rotulos por video: (classe, t_inicio, t_fim, ids_das_trilhas ou None para todas).
# Definidos por inspecao visual das folhas de contato dos recortes rastreados.
ROTULOS = {
    "0002": [("socorro", 0, 999, [1])],                  # T em orbita, 35 m descendo ate 13 m
    "0003": [("sem_socorro", 0, 999, None)],             # sentado, caminhando, em pe; transeuntes
    "0004": [("sem_socorro", 0, 999, [1, 2, 3, 4, 5, 6, 8, 10]),   # caminhando, em pe, acenando
             ("sem_socorro", 0, 999, [7]),               # caminhando, um braco, agachado, deitado
             ("sem_socorro", 0, 70, [9]),                # caminhando, em pe
             # 70 a 76 s: a pessoa de camisa cinza abriu os dois bracos por acaso, sem intencao
             # de sinalizar. Visualmente e um T, entao esse trecho fica fora das duas classes.
             ("sem_socorro", 76, 999, [9])],             # braco so, diagonal, aceno
    "0005": [("vazio", 0, 999, None)],                   # terreno sem pessoa
    "0006": [("sem_socorro", 42, 43.5, [2]),             # gimbal a -61 graus: caminhando
             ("socorro", 44.5, 48, [2]),                 # T
             ("sem_socorro", 49, 51, [2]),               # caminhando
             ("socorro", 52.5, 69.5, [2]),               # T
             ("sem_socorro", 70.5, 999, [2])],           # bracos caidos
}


# ---------- telemetria DJI (trilha djmd) ----------
def _varint(b, i):
    r, sh = 0, 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << sh
        sh += 7
        if not c & 0x80:
            return r, i


def _valido(b):
    i = 0
    try:
        while i < len(b):
            k, i = _varint(b, i)
            wt = k & 7
            if wt == 0:
                _, i = _varint(b, i)
            elif wt == 1:
                i += 8
            elif wt == 5:
                i += 4
            elif wt == 2:
                n, i = _varint(b, i)
                i += n
            else:
                return False
            if i > len(b):
                return False
    except IndexError:
        return False
    return i == len(b)


def _decode(b, path=""):
    """Decodifica protobuf sem esquema: devolve {caminho.do.campo: valor}."""
    i, res = 0, {}
    while i < len(b):
        k, i = _varint(b, i)
        f, wt = k >> 3, k & 7
        if wt == 0:
            res[path + str(f)], i = _varint(b, i)
        elif wt == 1:
            res[path + str(f)] = struct.unpack("<d", b[i:i + 8])[0]
            i += 8
        elif wt == 5:
            res[path + str(f)] = struct.unpack("<f", b[i:i + 4])[0]
            i += 4
        elif wt == 2:
            n, i = _varint(b, i)
            sub = b[i:i + n]
            i += n
            if sub and _valido(sub):
                res.update(_decode(sub, path + str(f) + "."))
            else:
                res[path + str(f)] = sub
        else:
            break
    return res


def altitudes(mp4):
    """Lista de (tempo_s, altura_acima_da_decolagem_m) lida da trilha djmd.

    O campo 3.3.5.1 e a altura relativa em milimetros; confere com a escala aparente da
    pessoa no quadro e com o perfil de descida visivel no video."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-map", "0:1", "-c", "copy",
                          "-f", "data", "-"], capture_output=True, check=True).stdout
    meta = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "1", "-show_entries",
                           "packet=pts_time,size", "-of", "csv=p=0", str(mp4)],
                          capture_output=True, text=True, check=True).stdout.split()
    out, o = [], 0
    for linha in meta:
        t, s = linha.split(",")
        s = int(s)
        d = _decode(raw[o:o + s])
        o += s
        if "3.3.5.1" in d:
            out.append((float(t), d["3.3.5.1"] / 1000.0))
    return out


def altura_em(alts, t):
    return min(alts, key=lambda p: abs(p[0] - t))[1]


# ---------- quadros ----------
def videos():
    vs = {}
    for p in sorted(VIDEOS.glob("*.MP4")):
        vid = p.stem.split("_")[2]
        vs[vid] = p
    return vs


def extrair(vid, mp4):
    pasta = QUADROS / vid
    if pasta.exists() and any(pasta.glob("*.jpg")):
        return pasta
    pasta.mkdir(parents=True, exist_ok=True)
    print(f"extraindo quadros de {mp4.name} a {FPS_EXTRACAO} fps (decodificar 4K HEVC leva minutos)")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"fps={FPS_EXTRACAO}",
                    "-q:v", "2", "-y", str(pasta / "f%04d.jpg")], check=True)
    return pasta


# ---------- rastreamento ----------
def rastrear(vid, pasta, alts):
    cache = RASTROS / f"{vid}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from ultralytics import YOLO
    modelo = YOLO("yolo11m.pt")

    def det(img, imgsz, conf, ox=0, oy=0):
        r = modelo.predict(img, imgsz=imgsz, conf=conf, classes=[0], verbose=False)[0]
        return [(int(b.xyxy[0][0]) + ox, int(b.xyxy[0][1]) + oy, int(b.xyxy[0][2]) + ox,
                 int(b.xyxy[0][3]) + oy, float(b.conf[0])) for b in r.boxes]

    def janela(im, cx, cy, lado=1280):
        h, w = im.shape[:2]
        x = int(min(max(cx - lado // 2, 0), w - lado))
        y = int(min(max(cy - lado // 2, 0), h - lado))
        return det(im[y:y + lado, x:x + lado], lado, 0.12, x, y)

    def ladrilhos(im, lado=1280, sobre=200):
        h, w = im.shape[:2]
        xs = list(range(0, w - lado + 1, lado - sobre))
        ys = list(range(0, h - lado + 1, lado - sobre))
        if xs[-1] + lado < w:
            xs.append(w - lado)
        if ys[-1] + lado < h:
            ys.append(h - lado)
        out = []
        for y in ys:
            for x in xs:
                out += det(im[y:y + lado, x:x + lado], lado, 0.12, x, y)
        return out

    def centro(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def nms(ds, thr=150):
        keep = []
        for d in sorted(ds, key=lambda d: -d[4]):
            if all(dist(centro(d), centro(k)) > thr for k in keep):
                keep.append(d)
        return keep

    quadros = sorted(pasta.glob("*.jpg"))
    trilhas, prox_id, saida = {}, 1, []
    for fi, f in enumerate(quadros):
        im = cv2.imread(str(f))
        t = fi / FPS_EXTRACAO
        ativas = [(tid, b) for tid, (b, ult) in trilhas.items() if fi - ult <= 6]
        ds = []
        for _, b in ativas:
            cx, cy = centro(b)
            ds += janela(im, cx, cy)
        if fi % 5 == 0 or not ativas:
            ds += det(im, 1920, 0.15)
        if not ds and not ativas:
            ds += ladrilhos(im)
        atribuidas, usadas = [], set()
        for d in nms(ds):
            c = centro(d)
            melhor = None
            for tid, b in ativas:
                if tid in usadas:
                    continue
                dd = dist(c, centro(b))
                if dd < 400 and (melhor is None or dd < melhor[0]):
                    melhor = (dd, tid)
            if melhor:
                tid = melhor[1]
                usadas.add(tid)
            else:
                tid = prox_id
                prox_id += 1
            trilhas[tid] = (d, fi)
            atribuidas.append(dict(id=tid, x1=d[0], y1=d[1], x2=d[2], y2=d[3], conf=round(d[4], 3)))
        saida.append(dict(frame=f.name, t=t, alt=round(altura_em(alts, t), 2), tracks=atribuidas))
        if fi % 20 == 0:
            print(f"  {vid}: quadro {fi}/{len(quadros)} t={t:.0f}s alt={saida[-1]['alt']:.0f} m "
                  f"pessoas={len(atribuidas)}", flush=True)
    RASTROS.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(saida))
    return saida


# ---------- recorte ----------
def gsd_m_por_px(alt_m, largura_px):
    return 2 * alt_m * math.tan(math.radians(HFOV_GRAUS) / 2) / largura_px


def recorte_solo(im, alt, cx, cy):
    """Recorte quadrado com lado sorteado em metros e centro deslocado por jitter."""
    h, w = im.shape[:2]
    gsd = gsd_m_por_px(alt, w)
    lado = int(random.uniform(*LADO_SOLO_M) / gsd)
    lado = min(lado, h)
    cx += random.uniform(-JITTER_M, JITTER_M) / gsd
    cy += random.uniform(-JITTER_M, JITTER_M) / gsd
    x = int(min(max(cx - lado / 2, 0), w - lado))
    y = int(min(max(cy - lado / 2, 0), h - lado))
    tile = cv2.resize(im[y:y + lado, x:x + lado], (TILE, TILE), interpolation=cv2.INTER_AREA)
    # O voo dos positivos saiu mais claro que os demais (exposicao automatica do drone), e um
    # classificador linear so de brilho ja separava as classes. O ganho aleatorio sobrepoe as
    # distribuicoes de brilho das duas classes, para que o modelo nao use exposicao como pista.
    ganho = random.uniform(*GANHO)
    return np.clip(tile.astype(np.float32) * ganho, 0, 255).astype(np.uint8)


def fracao_branca(im, cx, cy, lado):
    x0, y0 = int(max(cx - lado / 2, 0)), int(max(cy - lado / 2, 0))
    g = cv2.cvtColor(im[y0:y0 + lado:4, x0:x0 + lado:4], cv2.COLOR_BGR2GRAY)
    return float((g > 200).mean()) if g.size else 0.0


def classe_de(vid, t, tid):
    for classe, t0, t1, ids in ROTULOS.get(vid, []):
        if t0 <= t < t1 and (ids is None or tid in ids):
            return classe
    return None


def recortar(rastros_por_video):
    if OUT.exists():
        for sub in ("socorro", "sem_socorro"):
            shutil.rmtree(OUT / sub, ignore_errors=True)
    for sub in ("socorro", "sem_socorro"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    # candidato: (video, registro do quadro, trilha ou None, tipo)
    positivos, negativos = [], {"mesma_cena": [], "pessoa": [], "vazio": []}
    for vid, rastros in rastros_por_video.items():
        for rec in rastros:
            if classe_de(vid, rec["t"], None) == "vazio":
                for _ in range(VAZIOS_POR_QUADRO):
                    negativos["vazio"].append((vid, rec, None, "vazio"))
                continue
            for tr in rec["tracks"]:
                classe = classe_de(vid, rec["t"], tr["id"])
                if classe == "socorro":
                    positivos.append((vid, rec, tr, "pessoa"))
                    negativos["mesma_cena"].append((vid, rec, tr, "mesma_cena"))
                elif classe == "sem_socorro":
                    negativos["pessoa"].append((vid, rec, tr, "pessoa"))

    def uniforme(lista, n):
        if len(lista) <= n:
            return lista
        passo = len(lista) / n
        return [lista[int(i * passo)] for i in range(n)]

    selecao = {"socorro": uniforme(positivos, MAX_POR_CLASSE), "sem_socorro": []}
    for tipo, fracao in MISTURA_NEGATIVOS.items():
        selecao["sem_socorro"] += uniforme(negativos[tipo], int(MAX_POR_CLASSE * fracao))

    resumo = {}
    for classe, lista in selecao.items():
        origem = Counter()
        for vid, rec, tr, tipo in lista:
            im = cv2.imread(str(QUADROS / vid / rec["frame"]))
            h, w = im.shape[:2]
            gsd = gsd_m_por_px(rec["alt"], w)
            lado = int(max(LADO_SOLO_M) / gsd)
            if tipo == "vazio":
                cx = random.uniform(lado / 2, w - lado / 2)
                cy = random.uniform(lado / 2, h - lado / 2)
                tag = "vazio"
            elif tipo == "mesma_cena":
                # Mesmo quadro do positivo, deslocado o bastante para a pessoa ficar de fora.
                # Entre varios deslocamentos possiveis, fica o que mais se parece com o recorte
                # do positivo em fracao de pixels brancos (a faixa de pedestres): sem isso o
                # negativo cai no piso liso ao lado e a faixa vira atalho para "socorro".
                px = (tr["x1"] + tr["x2"]) / 2
                py = (tr["y1"] + tr["y2"]) / 2
                alvo = fracao_branca(im, px, py, lado)
                opcoes = []
                for _ in range(60):
                    ang = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(1.1, 2.0) * lado
                    ox, oy = px + dist * math.cos(ang), py + dist * math.sin(ang)
                    if lado / 2 <= ox <= w - lado / 2 and lado / 2 <= oy <= h - lado / 2:
                        opcoes.append((abs(fracao_branca(im, ox, oy, lado) - alvo), ox, oy))
                if not opcoes:
                    continue
                _, cx, cy = min(opcoes)
                tag = f"cena_t{tr['id']}"
            else:
                cx = (tr["x1"] + tr["x2"]) / 2
                cy = (tr["y1"] + tr["y2"]) / 2
                tag = f"t{tr['id']}"
            tile = recorte_solo(im, rec["alt"], cx, cy)
            nome = f"{classe}_{vid}_{rec['t']:05.1f}s_{tag}_{int(rec['alt']):02d}m.jpg"
            cv2.imwrite(str(OUT / classe / nome), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
            origem[f"{vid}/{tipo}"] += 1
        resumo[classe] = (sum(origem.values()), origem)
        print(f"{classe}: {sum(origem.values())} recortes  {dict(origem)}")

    for classe in selecao:
        zp = OUT / f"{classe}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted((OUT / classe).glob("*.jpg")):
                z.write(p, p.name)
        print(f"{zp.name}: {zp.stat().st_size / 1e6:.1f} MB")
    return resumo


def main():
    etapas = sys.argv[1:] or ["extrair", "rastrear", "recortar"]
    vs = videos()
    if not vs:
        sys.exit(f"nenhum video em {VIDEOS}")
    rastros = {}
    for vid, mp4 in vs.items():
        if vid not in ROTULOS:
            print(f"{mp4.name}: sem rotulo, ignorado")
            continue
        alts = altitudes(mp4)
        print(f"{mp4.name}: {len(alts)} amostras de telemetria, altura "
              f"{min(a for _, a in alts):.0f} a {max(a for _, a in alts):.0f} m")
        pasta = extrair(vid, mp4) if "extrair" in etapas else QUADROS / vid
        if "rastrear" in etapas or "recortar" in etapas:
            if ROTULOS[vid][0][0] == "vazio":
                quadros = sorted(pasta.glob("*.jpg"))
                rastros[vid] = [dict(frame=f.name, t=i / FPS_EXTRACAO,
                                     alt=round(altura_em(alts, i / FPS_EXTRACAO), 2), tracks=[])
                                for i, f in enumerate(quadros)]
            else:
                rastros[vid] = rastrear(vid, pasta, alts)
    if "recortar" in etapas:
        recortar(rastros)


if __name__ == "__main__":
    main()

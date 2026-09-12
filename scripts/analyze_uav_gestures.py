"""Escolhe o gesto de socorro do ResgatAr medindo dados aereos reais.

Fonte: UAV-GESTURE (Perera, Law, Chahl; ECCVW 2018), anotacoes de articulacoes estimadas
sobre filmagem de drone pairando. 13 gestos, 119 clipes, 37.151 quadros, formato OpenPose
COCO com 18 articulacoes (x, y, confianca). Uso academico.

O gesto de socorro precisa de quatro propriedades, todas mensuraveis aqui:

1. Extensao: quanto maior a silhueta, mais pixels no sensor, e maior a altura em que ainda
   da para classificar. Liga direto a conta de resolucao do prototipo.
2. Distancia do repouso: o falso positivo mais provavel e uma pessoa simplesmente em pe.
3. Estabilidade temporal: a janela de confirmacao sustentada classifica quadro a quadro,
   entao um gesto que mantem a forma e melhor que um movimento rapido.
4. Simetria: o drone chega por um azimute qualquer. Gesto simetrico nos dois bracos depende
   menos da direcao de aproximacao.
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRATCH = Path(r"C:\Users\jefer\AppData\Local\Temp\claude\C--Users-jefer-ResgatAr\f00b77f9-d6a7-4987-b3cf-804cf0e37af2\scratchpad")
ZIP_PATH = SCRATCH / "uavg_json.zip"

# OpenPose COCO-18
NOSE, NECK = 0, 1
RSHO, RELB, RWRI = 2, 3, 4
LSHO, LELB, LWRI = 5, 6, 7
RHIP, RKNE, RANK = 8, 9, 10
LHIP, LKNE, LANK = 11, 12, 13
MIN_CONF = 0.2


def load_frames():
    """Devolve {gesto: {clipe: [pose 18x3, ...]}}."""
    z = zipfile.ZipFile(ZIP_PATH)
    data = defaultdict(lambda: defaultdict(list))
    for name in z.namelist():
        parts = name.split("/")
        if len(parts) < 4 or not parts[3].endswith(".json"):
            continue
        d = json.loads(z.read(name))
        people = d.get("people", [])
        if not people:
            continue
        kp = np.array(people[0]["pose_keypoints_2d"], dtype=float).reshape(-1, 3)
        data[parts[1]][parts[2]].append(kp)
    return data


def normalize(kp):
    """Centra no pescoco e escala pelo tronco. Devolve (18x2, ok)."""
    conf = kp[:, 2]
    if conf[NECK] < MIN_CONF:
        return None, False
    hips = [i for i in (RHIP, LHIP) if conf[i] >= MIN_CONF]
    if not hips:
        return None, False
    quadril = kp[hips, :2].mean(axis=0)
    tronco = np.linalg.norm(kp[NECK, :2] - quadril)
    if tronco < 1e-3:
        return None, False
    pts = (kp[:, :2] - kp[NECK, :2]) / tronco
    pts[conf < MIN_CONF] = np.nan
    return pts, True


def features(pts):
    """Vetor de forma: articulacoes dos bracos e pernas em relacao ao tronco."""
    idx = [RSHO, RELB, RWRI, LSHO, LELB, LWRI, RHIP, RKNE, RANK, LHIP, LKNE, LANK, NOSE]
    return pts[idx].reshape(-1)


def extent(pts):
    """Maior distancia entre articulacoes validas, em unidades de tronco."""
    v = pts[~np.isnan(pts).any(axis=1)]
    if len(v) < 2:
        return np.nan
    d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
    return d.max()


def arm_spread(pts):
    """Largura horizontal entre os punhos: e o que a camera de cima realmente enxerga."""
    if np.isnan(pts[[RWRI, LWRI]]).any():
        return np.nan
    return abs(pts[RWRI, 0] - pts[LWRI, 0])


def wrist_height(pts):
    """Altura media dos punhos acima do pescoco (y cresce para baixo na imagem)."""
    w = pts[[RWRI, LWRI], 1]
    if np.isnan(w).any():
        return np.nan
    return -w.mean()


def asymmetry(pts):
    """Diferenca entre os dois bracos: 0 = perfeitamente simetrico."""
    r, l = pts[RWRI], pts[LWRI]
    if np.isnan(r).any() or np.isnan(l).any():
        return np.nan
    espelhado = np.array([-l[0], l[1]])
    return np.linalg.norm(r - espelhado)


def main():
    data = load_frames()

    per_gesture = {}
    for gesto, clipes in data.items():
        feats, exts, spreads, heights, asyms, clip_var = [], [], [], [], [], []
        for clipe, frames in clipes.items():
            cf = []
            for kp in frames:
                pts, ok = normalize(kp)
                if not ok:
                    continue
                f = features(pts)
                if np.isnan(f).mean() > 0.3:
                    continue
                cf.append(np.nan_to_num(f))
                exts.append(extent(pts))
                spreads.append(arm_spread(pts))
                heights.append(wrist_height(pts))
                asyms.append(asymmetry(pts))
            if len(cf) > 3:
                cf = np.array(cf)
                feats.append(cf)
                clip_var.append(cf.var(axis=0).mean())
        if not feats:
            continue
        todos = np.vstack(feats)
        per_gesture[gesto] = {
            "n": len(todos),
            "centro": todos.mean(axis=0),
            "espalhamento": todos.std(axis=0).mean(),
            "extensao": np.nanmean(exts),
            "abertura": np.nanmean(spreads),
            "altura_punhos": np.nanmean(heights),
            "assimetria": np.nanmean(asyms),
            "variacao_temporal": float(np.mean(clip_var)),
        }

    nomes = sorted(per_gesture)
    centros = np.array([per_gesture[g]["centro"] for g in nomes])

    # separabilidade: distancia ao centro mais proximo, dividida pela dispersao interna
    for i, g in enumerate(nomes):
        d = np.linalg.norm(centros - centros[i], axis=1)
        d[i] = np.inf
        per_gesture[g]["dist_vizinho"] = float(d.min())
        per_gesture[g]["vizinho"] = nomes[int(d.argmin())]
        per_gesture[g]["separabilidade"] = float(d.min() / (per_gesture[g]["espalhamento"] + 1e-6))

    # repouso de referencia: gesto com punhos mais baixos (bracos caidos)
    repouso = min(nomes, key=lambda g: per_gesture[g]["altura_punhos"])
    c_rep = per_gesture[repouso]["centro"]
    for g in nomes:
        per_gesture[g]["dist_repouso"] = float(np.linalg.norm(per_gesture[g]["centro"] - c_rep))

    def z(vals):
        v = np.array(vals, dtype=float)
        return (v - v.mean()) / (v.std() + 1e-9)

    ext = z([per_gesture[g]["extensao"] for g in nomes])
    abr = z([per_gesture[g]["abertura"] for g in nomes])
    rep = z([per_gesture[g]["dist_repouso"] for g in nomes])
    sep = z([per_gesture[g]["separabilidade"] for g in nomes])
    est = -z([per_gesture[g]["variacao_temporal"] for g in nomes])
    sim = -z([per_gesture[g]["assimetria"] for g in nomes])

    score = 0.25 * abr + 0.15 * ext + 0.20 * rep + 0.20 * sep + 0.10 * est + 0.10 * sim
    ordem = np.argsort(-score)

    print(f"Referencia de repouso (punhos mais baixos): {repouso}\n")
    print(f"{'gesto':20} {'nota':>6} {'abert':>6} {'exten':>6} {'repou':>6} {'separ':>6} "
          f"{'estab':>6} {'simet':>6}  confusao mais provavel")
    print("-" * 104)
    for i in ordem:
        g = nomes[i]
        p = per_gesture[g]
        print(f"{g:20} {score[i]:6.2f} {p['abertura']:6.2f} {p['extensao']:6.2f} "
              f"{p['dist_repouso']:6.2f} {p['separabilidade']:6.2f} "
              f"{p['variacao_temporal']:6.3f} {p['assimetria']:6.2f}  {p['vizinho']}")

    print("\nabert = abertura horizontal entre punhos (unidades de tronco), o que a camera de cima ve")
    print("exten = maior distancia entre articulacoes | repou = distancia da postura de repouso")
    print("separ = distancia ao gesto vizinho dividida pela dispersao interna")
    print("estab = variacao dentro do clipe (menor e melhor) | simet = assimetria entre bracos (menor e melhor)")


if __name__ == "__main__":
    main()

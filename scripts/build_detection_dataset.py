"""Prepara dados de treino para o estagio de deteccao ("pessoa" x "sem pessoa") do ResgatAr.

Fonte: "Search-and-rescue from drones with computer vision" (Castellano, Carone, Scigliuto,
Vessio; Universidade de Bari), Zenodo 3924925, cenario mountains: 200 quadros de drone,
100 com pessoas anotadas e 100 sem pessoa.

As anotacoes seguem o formato YOLO: <categoria> <x_centro> <y_centro> <largura> <altura>,
normalizados pelas dimensoes do quadro. Categoria 1 = pessoa.

Recortar e essencial: numa imagem de 1920 px, uma pessoa vista de cima ocupa poucas dezenas
de pixels e desapareceria ao reduzir o quadro inteiro para os 224x224 do Teachable Machine.
Os recortes negativos usam a mesma distribuicao de tamanhos dos positivos, senao o modelo
aprenderia escala em vez de conteudo.
"""
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageStat

RAIZ = Path(__file__).resolve().parent.parent
BRUTOS = RAIZ / "dados-brutos"
ZIP_PATH = BRUTOS / "uav-search-and-rescue-v1.0.zip"
WORK = BRUTOS / "sar_mountains"
OUT = RAIZ / "dataset-deteccao"

FONTE_URL = "https://zenodo.org/api/records/3924925/files/gvessio/uav-search-and-rescue-v1.0.zip/content"

INNER = "gvessio-uav-search-and-rescue-8961f73/dataset/mountains"
TILE = 224
CONTEXT = 2.6      # quantas vezes o lado da caixa vira o lado do recorte
MIN_CROP_PX = 48   # evita recortes minusculos que viram borrao ao ampliar

random.seed(11)


def baixar_fonte():
    """Baixa o arquivo original do Zenodo, se ainda nao estiver em dados-brutos/."""
    if ZIP_PATH.exists():
        print(f"fonte ja presente: {ZIP_PATH.name}")
        return
    import urllib.request
    BRUTOS.mkdir(parents=True, exist_ok=True)
    print(f"baixando {ZIP_PATH.name} do Zenodo (139 MB)...")
    urllib.request.urlretrieve(FONTE_URL, ZIP_PATH)
    print("download concluido")


def extract():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        members = [n for n in z.namelist() if n.startswith(INNER) and not n.endswith("/")]
        for n in members:
            target = WORK / Path(n).relative_to(INNER)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"extraidos {len(members)} arquivos de mountains")


def read_boxes(txt_path, w, h):
    """Devolve caixas em pixels (x0, y0, x1, y1)."""
    boxes = []
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        return boxes
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = parts
        xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
        boxes.append((xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
    return boxes


def square_crop(img, cx, cy, side):
    w, h = img.size
    side = min(side, w, h)
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    side = int(round(side))
    return img.crop((x0, y0, x0 + side, y0 + side)).resize((TILE, TILE), Image.LANCZOS)


def is_letterbox(tile):
    """Descarta recortes tocados pelas barras pretas dos videos originais.

    Barras so aparecem em negativos (positivos sao centrados na pessoa), entao qualquer
    fracao preta relevante vira atalho para o classificador e precisa sair.
    """
    gray = tile.convert("L")
    stat = ImageStat.Stat(gray)
    if stat.mean[0] < 14 or stat.stddev[0] < 4:
        return True
    hist = gray.histogram()
    dark = sum(hist[:18]) / (gray.size[0] * gray.size[1])
    return dark > 0.15


def overlaps_person(x0, y0, side, boxes, margin=1.4):
    """Verdadeiro se o recorte encosta em alguma caixa de pessoa (com folga)."""
    cx, cy = x0 + side / 2, y0 + side / 2
    for bx0, by0, bx1, by1 in boxes:
        pcx, pcy = (bx0 + bx1) / 2, (by0 + by1) / 2
        half = (side + max(bx1 - bx0, by1 - by0) * margin) / 2
        if abs(cx - pcx) < half and abs(cy - pcy) < half:
            return True
    return False


def tile_stats(tile):
    st = ImageStat.Stat(tile.convert("L"))
    return st.mean[0], st.stddev[0]


def sample_negative(img, boxes, sides, target, tries=34):
    """Recorte sem pessoa, do mesmo quadro, pareado aos positivos em tamanho, brilho e contraste.

    Sortear negativo puramente ao acaso cai quase sempre em ceu, agua ou grama lisa, e o
    classificador passa a separar "liso e claro" de "texturizado e escuro" em vez de detectar
    a pessoa. Cada negativo e escolhido entre varios candidatos, ficando o mais proximo de um
    par (brilho, contraste) sorteado da distribuicao dos positivos, o que e mineracao de
    negativos dificeis. O pareamento nunca fica perfeito: a pessoa vista de cima ao entardecer
    e mesmo um vulto escuro, e parte dessa diferenca e sinal legitimo, nao atalho.
    """
    t_mean, t_std = target
    w, h = img.size
    best = None
    best_gap = None
    for _ in range(tries):
        side = min(random.choice(sides), w, h)
        x0 = random.uniform(0, w - side)
        y0 = random.uniform(0, h - side)
        if overlaps_person(x0, y0, side, boxes):
            continue
        tile = square_crop(img, x0 + side / 2, y0 + side / 2, side)
        if is_letterbox(tile):
            continue
        mean, std = tile_stats(tile)
        gap = abs(mean - t_mean) / 40 + abs(std - t_std) / 18
        if best_gap is None or gap < best_gap:
            best, best_gap = tile, gap
        if gap < 0.25:
            break
    return best


def build():
    frames = sorted((WORK / "frames").glob("*.jpg"))
    ann_dir = WORK / "annotations"

    pos_dir = OUT / "pessoa"
    neg_dir = OUT / "sem_pessoa"
    for d in (pos_dir, neg_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    crop_sides = []
    pos_contrasts = []   # pares (brilho, contraste) dos positivos
    n_pos = 0
    annotated = []   # (caminho, caixas)
    empty_frames = []

    for f in frames:
        img = Image.open(f).convert("RGB")
        boxes = read_boxes(ann_dir / (f.stem + ".txt"), *img.size)
        if not boxes:
            empty_frames.append(f)
            continue
        annotated.append((f, boxes))
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            side = max(max(x1 - x0, y1 - y0) * CONTEXT, MIN_CROP_PX)
            crop_sides.append(side)
            tile = square_crop(img, (x0 + x1) / 2, (y0 + y1) / 2, side)
            pos_contrasts.append(tile_stats(tile))
            tile.save(pos_dir / f"{f.stem}_{i:02d}.jpg", quality=88)
            n_pos += 1

    print(f"quadros com pessoa: {len(annotated)} | recortes positivos: {n_pos}")
    print(f"quadros sem pessoa: {len(empty_frames)}")
    if crop_sides:
        print(f"lado do recorte: min {int(min(crop_sides))} px, "
              f"mediana {int(sorted(crop_sides)[len(crop_sides)//2])} px, max {int(max(crop_sides))} px")

    # Negativos: a maior parte sai dos MESMOS quadros dos positivos, em regioes sem pessoa.
    # Sem isso, as classes diferem por cena (luz, terreno, video) e o modelo aprende o cenario
    # em vez da pessoa, com acuracia alta e inutil na pratica.
    n_neg = 0
    target_same = round(n_pos * 0.7)
    per_annotated = max(1, round(target_same / max(len(annotated), 1)))

    for f, boxes in annotated:
        img = Image.open(f).convert("RGB")
        for i in range(per_annotated):
            tile = sample_negative(img, boxes, crop_sides, random.choice(pos_contrasts))
            if tile is None:
                continue
            tile.save(neg_dir / f"mesmo_{f.stem}_{i:02d}.jpg", quality=88)
            n_neg += 1
    n_same = n_neg

    # complemento com quadros sem pessoa, para variedade de fundo
    faltam = max(n_pos - n_neg, 0)
    per_empty = max(1, round(faltam / max(len(empty_frames), 1)))
    for f in empty_frames:
        img = Image.open(f).convert("RGB")
        for i in range(per_empty):
            tile = sample_negative(img, [], crop_sides, random.choice(pos_contrasts))
            if tile is None:
                continue
            tile.save(neg_dir / f"vazio_{f.stem}_{i:02d}.jpg", quality=88)
            n_neg += 1

    print(f"recortes negativos: {n_neg} ({n_same} do mesmo quadro dos positivos, "
          f"{n_neg - n_same} de quadros sem pessoa)")

    for name, d in (("pessoa", pos_dir), ("sem_pessoa", neg_dir)):
        zip_path = OUT / f"{name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for img_path in sorted(d.glob("*.jpg")):
                z.write(img_path, img_path.name)
        print(f"{zip_path.name}: {len(list(d.glob('*.jpg')))} imagens, "
              f"{round(zip_path.stat().st_size / 1024 / 1024, 1)} MB")


if __name__ == "__main__":
    baixar_fonte()
    extract()
    build()

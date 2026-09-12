# Dados de treino: estágio de detecção (pessoa vista de cima)

Duas classes prontas para subir no Teachable Machine (projeto de imagem):

| Classe | Imagens | Arquivo |
|---|---|---|
| `pessoa` | 305 | `pessoa.zip` |
| `sem_pessoa` | 301 | `sem_pessoa.zip` |

Todas em 224x224, que é a entrada do Teachable Machine.

## Procedência

Base: **"Search-and-rescue from drones with computer vision"**, de Giovanna Castellano,
Davide Carone, Francesco Scigliuto e Gennaro Vessio (Universidade de Bari "Aldo Moro"),
publicada no Zenodo sob o registro 3924925, com acesso aberto.

Usamos apenas o cenário `mountains`: 200 quadros capturados por drone, sendo 101 com pessoas
anotadas e 99 sem pessoa. As anotações originais seguem o formato YOLO
(`categoria x_centro y_centro largura altura`, normalizados), com categoria 1 para pessoa.

Regeneração: `python scripts/build_detection_dataset.py`

## Como os recortes foram feitos

Os quadros têm cerca de 1920 px de largura e uma pessoa vista de cima ocupa poucas dezenas de
pixels. Reduzir o quadro inteiro para 224x224 apagaria a pessoa, então cada amostra é um
recorte quadrado centrado na anotação, com 2,6 vezes o lado da caixa para dar contexto.

Os negativos não são aleatórios. Foram tomadas duas decisões que mudam o resultado:

1. **Dois terços dos negativos saem dos mesmos quadros dos positivos**, em regiões sem pessoa.
   Se as classes viessem de vídeos diferentes, o modelo aprenderia a separar cenas (luz, terreno,
   câmera) em vez de detectar pessoas, e ainda assim mostraria acurácia alta.
2. **Cada negativo é pareado aos positivos em tamanho, brilho e contraste.** Sorteio puramente
   aleatório cai quase sempre em céu, água ou grama lisa, e o classificador passaria a separar
   "liso e claro" de "texturizado e escuro".

Recortes que tocam as barras pretas de letterbox dos vídeos originais são descartados, porque
essas barras só apareceriam nos negativos e viram atalho.

## Limitações (importantes para o relatório)

- **Diversidade de cena baixa.** Os 101 quadros anotados vêm de poucos vídeos, e quadros
  consecutivos são quase idênticos. Os 305 recortes representam bem menos que 305 situações
  distintas, então a acurácia relatada pelo Teachable Machine será otimista.
- **Ponto de vista.** São tomadas oblíquas de drone a baixa altura, não a vista de cima a 100 m
  que o protótipo simula.
- **Viés residual medido.** Um classificador linear usando somente estatísticas de cor da imagem
  atinge 73,4% de acerto nessas duas classes (era 75,9% antes do pareamento). Parte disso é sinal
  legítimo, já que uma pessoa vista de cima ao entardecer é mesmo um vulto mais escuro e
  texturizado que o terreno ao redor, mas significa que um modelo pode se apoiar parcialmente em
  cor e textura em vez de forma humana.
- **Não há rótulo de gesto.** Esta base treina apenas "há pessoa" contra "não há pessoa". O
  reconhecimento do gesto de socorro continua sendo um segundo estágio, com dados próprios.

## Citação

Castellano, G., Carone, D., Scigliuto, F., Vessio, G. *Search-and-rescue from drones with
computer vision*. Zenodo. https://doi.org/10.5281/zenodo.3924925

# Dados de treino: estágio de gesto (socorro visto de cima)

Duas classes prontas para subir no Teachable Machine (projeto de imagem):

| Classe | Imagens | Arquivo |
|---|---|---|
| `socorro` | 214 | `socorro.zip` |
| `sem_socorro` | 331 | `sem_socorro.zip` |

Todas em 224x224, que é a entrada do Teachable Machine.

## Procedência

Voos próprios com um DJI Flip sobre o campus da Univates (Lajeado, RS), em 14 de setembro de
2026, por volta das 13h40, céu encoberto. Cinco vídeos em 4K a 30 fps, somando 6 minutos:

| Vídeo | Gimbal | Altura | Conteúdo |
|---|---|---|---|
| 0002 | -90° | 37 m descendo a 13 m | pessoa em T, drone orbitando |
| 0003 | -90° | 17 a 24 m | mesma pessoa sentada, caminhando, em pé; transeuntes |
| 0004 | -90° | 15 a 25 m | duas pessoas: caminhando, um braço só, agachado, deitado, acenando |
| 0005 | -90° | 15 a 22 m | terreno sem ninguém |
| 0006 | -61° | 14 m | pessoa em T e caminhando, tomada oblíqua |

A altura de cada quadro foi lida da telemetria que o DJI grava dentro do próprio MP4 (trilha
de dados `djmd`, em protobuf; o campo de altura relativa vem em milímetros). Os vídeos não estão
no repositório (3 GB). As pessoas filmadas consentiram com o uso acadêmico das imagens.

Regeneração: `python scripts/build_gesture_dataset.py` (precisa de ffmpeg, OpenCV e
ultralytics; o rastreio da pessoa leva cerca de 40 minutos em CPU).

## Como os recortes foram feitos

1. **Quadros a 2 por segundo.** Com o drone orbitando, dois quadros seguidos já diferem em
   ângulo e escala; mais que isso seria repetição.
2. **Pessoa localizada por um detector genérico** (YOLO11, classe "person"), em janela ao redor
   da última posição com varredura completa periódica. Quadro sem detecção é descartado.
3. **Recorte com tamanho fixo no solo**, entre 5 e 7,5 m de lado, sorteado, com o centro
   deslocado até 0,8 m. É a decisão mais importante: se o recorte acompanhasse o tamanho do
   vulto, uma pessoa de braços abertos (1,7 m) ganharia um recorte três vezes maior que uma de
   braços caídos (0,5 m), e o modelo aprenderia a escala do piso, não a postura. O protótipo
   aplica a mesma regra: a janela de inspeção cobre 6 m no solo na altura de inspeção.
4. **Rótulo por trecho de vídeo e trilha**, conferido visualmente em folhas de contato dos
   recortes. No vídeo 0004 a pessoa de camisa cinza abriu os dois braços sem intenção de
   sinalizar (70 a 76 s); como visualmente é um T, o trecho ficou fora das duas classes.

## Composição dos negativos

| Origem | Quantidade | Para quê |
|---|---|---|
| Mesmas pessoas em outras posturas (0003, 0004, 0006) | 198 | o modelo separar postura, não presença de pessoa |
| Mesmos quadros dos positivos, sem a pessoa (0002, 0006) | 97 | a cena do voo dos positivos (faixa de pedestres, luz) não virar atalho |
| Terreno vazio (0005) | 36 | ausência de pessoa também é "não socorro" |

Os negativos de mesma cena são escolhidos, entre vários deslocamentos possíveis, pelo que mais
se parece com o recorte positivo em fração de pixels brancos: sem isso caem no piso liso ao lado
da faixa, e a faixa vira sinônimo de socorro.

## Viés medido

Um classificador linear que enxerga só estatísticas de cor e brilho (sem forma) atinge:

| Versão dos dados | Acerto só com cor |
|---|---|
| Recortes brutos | 90% |
| Com negativos de mesma cena pareados | 90% |
| Com ganho de brilho aleatório de 0,75 a 1,25 em todos os recortes | **75%** |

O salto de 90% para 75% mostra o que estava acontecendo: o voo dos positivos saiu mais claro
(exposição automática do drone sobre a faixa de pedestres), e brilho médio sozinho separava as
classes. O ganho aleatório sobrepõe as distribuições de brilho. Os 75% restantes vêm em parte
da faixa de pedestres, presente em 86% dos positivos e em 63% dos negativos, e em parte de
sinal legítimo: braços abertos cobrem mais piso que braços caídos.

## Limitações (importantes para o relatório)

- **Uma pessoa faz o gesto**, de roupa escura, num único piso. Os 214 positivos são 173 quadros
  de uma órbita e 41 da tomada oblíqua. A acurácia do Teachable Machine será otimista; o teste
  honesto é outra pessoa, com outra roupa, em outro terreno.
- **Faixa de pedestres nos positivos.** Um modelo pode se apoiar parcialmente nela. O pareamento
  reduz, não elimina.
- **Altura.** Os dados vão de 13 a 37 m; o protótipo inspeciona a 25 m. A escala aparente é
  normalizada pelo recorte de tamanho fixo no solo, mas a nitidez muda com a altura.
- **Detector genérico.** A localização usa um modelo COCO que não foi treinado para vista de
  cima; a 35 m ele perde quadros (13 dos 186 do vídeo 0002). Só afeta a cobertura, não o rótulo.

# Escolha do gesto de socorro, por medição em dados aéreos reais

O protótipo nunca definiu qual é o "gesto de socorro". Esta análise escolhe um, a partir de
dados reais de gestos filmados por drone, em vez de intuição.

**Resultado: braços estendidos horizontalmente para os lados (postura em T).**

## Dados

UAV-GESTURE (Perera, Law, Chahl; ECCVW 2018): 13 gestos de sinalização de aeronave,
119 clipes, 37.151 quadros, gravados por um drone pairando, com articulações estimadas por
OpenPose (18 pontos). Uso acadêmico. Os vídeos exigem pedido aos autores; as anotações de
articulações são de acesso aberto, e foram elas que usamos.

Reprodução: `python scripts/analyze_uav_gestures.py`

## Critérios, todos mensuráveis nos dados

1. **Abertura horizontal entre os punhos.** É o que a câmera de cima realmente enxerga, e
   define em que altura o gesto ainda é classificável.
2. **Distância da postura de repouso.** O falso positivo mais provável é uma pessoa apenas
   em pé, então o gesto precisa estar longe disso.
3. **Separabilidade dos demais gestos**, medida como distância ao vizinho mais próximo
   dividida pela dispersão interna da classe.
4. **Estabilidade temporal.** A confirmação sustentada do protótipo classifica quadro a
   quadro, então uma postura mantida vale mais que um movimento rápido.
5. **Simetria entre os braços.** O drone chega por um azimute qualquer; gesto simétrico
   depende menos da direção de aproximação.

## Ranking (os cinco primeiros de 13)

| Gesto | Nota | Abertura | Dist. repouso | Separabilidade | Estabilidade | Assimetria |
|---|---|---|---|---|---|---|
| **hover** | **0,91** | **2,53** | **2,36** | **10,94** | **0,001** | **0,12** |
| move_to_right | 0,66 | 2,02 | 2,30 | 6,04 | 0,011 | 0,68 |
| move_to_left | 0,59 | 2,00 | 2,28 | 6,15 | 0,011 | 0,76 |
| move_upward | 0,42 | 1,73 | 2,28 | 2,15 | 0,031 | 0,15 |
| wave_off | 0,07 | 1,37 | 2,11 | 2,03 | 0,040 | 0,15 |

Abertura e distância em unidades de tronco (pescoço a quadril). Estabilidade é variação
dentro do clipe, menor é melhor. Assimetria menor é melhor.

O gesto `hover` vence em **todos** os critérios simultaneamente, o que é raro e dispensa
discussão de pesos.

## Geometria medida do gesto vencedor

Posição média dos punhos, em unidades de tronco, relativa ao pescoço:

| Gesto | Punho direito | Punho esquerdo | Altura do punho vs ombro |
|---|---|---|---|
| **hover** | x = -1,28, y = -0,01 | x = +1,25, y = -0,03 | +0,01 |
| move_upward | x = -0,90, y = -0,33 | x = +0,82, y = -0,29 | +0,32 |
| wave_off | x = -0,71, y = -0,25 | x = +0,66, y = -0,30 | +0,25 |
| land (repouso) | x = -0,08, y = +0,89 | x = +0,07, y = +0,90 | -0,90 |

Ou seja, no `hover` os punhos ficam na altura exata do ombro e no extremo lateral: braços
estendidos para os lados, formando um T.

## Por que o T é o gesto certo visto de cima

Da vertical, um braço erguido para cima projeta-se sobre a própria cabeça e some. Um braço
estendido na horizontal projeta a envergadura inteira no plano do solo. Por isso o gesto
clássico de acenar com os braços acima da cabeça, que funciona bem visto de frente, é uma
escolha ruim para um drone diretamente acima, e o T é a melhor.

Os dados confirmam a geometria: `move_upward` e `wave_off`, que erguem os braços, têm pouco
mais da metade da abertura lateral do `hover`.

Em pixels, com a câmera do protótipo (84 graus, 1920 px): a envergadura do T fica em torno de
50 a 73 px na altura de inspeção de 25 m, contra 13 a 18 px na altura de varredura de 100 m.
É a mesma conta que motivou o estágio de descida.

## Limites desta conclusão

- A separabilidade foi medida **contra os outros 12 gestos de sinalização**, não contra
  atividade humana comum. Numa busca real, o falso positivo vem de alguém caminhando,
  sentado ou deitado, não de outro sinal de marshalling. O número é evidência de postura
  distinguível, não validação de taxa de falso positivo em campo.
- A filmagem é de drone pairando em ângulo oblíquo, não na vertical a 25 m. A conclusão do T
  se mantém, e na verdade se reforça na vertical, mas os valores numéricos de abertura
  mudariam.
- Articulações vêm de estimativa do OpenPose sobre vídeo aéreo, com o erro próprio disso.

## Citação

Perera, A. G., Law, Y. W., Chahl, J. *UAV-GESTURE: A Dataset for UAV Control and Gesture
Recognition*. ECCV Workshops, 2018.

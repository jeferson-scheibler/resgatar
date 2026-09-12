# ResgatAr

Protótipo de sistema interativo de busca e resgate: um drone varre uma área de mata,
reconhece o gesto de socorro de quem está perdido e alerta a equipe, sem depender de
alguém pilotando e assistindo vídeo o tempo todo.

Trabalho da disciplina de Sistemas Interativos e de Visualização, Univates.

## Como rodar

O protótipo é uma página estática, sem build. Na raiz do repositório:

```bash
python -m http.server 8731
```

Depois abra `http://localhost:8731`. É preciso servir por HTTP: abrir o `index.html`
direto do disco bloqueia o acesso à câmera.

Para percorrer o fluxo sem treinar nada, use os botões **Simular candidato** e
**Simular gesto**, que disparam os mesmos eventos dos modelos reais.

## O fluxo

```
Usuário/Ambiente  →  Sensor  →  Processamento  →  Atuador  →  Feedback
   pessoa faz       câmera de     modelo TinyML    drone paira    alerta na
   o gesto          bordo         classifica       e registra     base
```

## Operação em dois estágios

A 100 m de altura, uma pessoa vista de cima ocupa 4,8 px de ombro a ombro, e os braços
abertos somam 18 px com 1,1 px de espessura. **O gesto é fisicamente irresolvível nessa
altura**, por mais dados de treino que existam: a informação não chega ao sensor.

Por isso a operação tem dois estágios, como em Liu e Szirányi (2021):

| Estágio | Altura | Modelo | Pergunta |
|---|---|---|---|
| 1. Varredura | 100 m | detecção de pessoa | Há alguém aqui embaixo? |
| 2. Inspeção | 25 m | gesto de socorro | Essa pessoa está pedindo ajuda? |

Confirmado um candidato, o drone desce. Os braços passam de 18 px para 73 px, e só então
o gesto pode ser classificado. Se o gesto não se confirma em 20 s, o candidato é
descartado e o drone sobe: a descida custou tempo e bateria, o que é o preço real de um
falso positivo no estágio 1.

## O que é dado real e o que é simulado

Real:

- **Relevo** da área de busca, de elevação SRTM obtida pela API Open-Elevation, usado
  para desenhar as curvas de nível, manter o voo com seguimento de terreno e priorizar
  as faixas de varredura.
- **Conjunto de treino do estágio 1**, derivado de filmagem de drone em cenário de
  montanha.
- **Escolha do gesto de socorro**, medida sobre 13 gestos aéreos reais.
- **Física do sensor**: faixa da câmera, distância de amostragem no solo e resolução do
  gesto saem do campo de visão e da altura, não de estimativa.

Simulado:

- O voo, a varredura e o cercamento eletrônico, em canvas 2D.
- A câmera de bordo, substituída pela webcam do operador.

## Varredura

Padrão paralelo (parallel track, o padrão IAMSAR para cobertura de área), com espaçamento
derivado da faixa que a câmera enxerga no solo: 180 m de faixa a 100 m de altura, 20% de
sobreposição, 144 m entre faixas, 12 faixas para a área.

O relevo define **a ordem** das faixas. Cada uma recebe um escore por elevação e
declividade, seguindo o perfil de comportamento de pessoa perdida discutido em Ewers et
al. (2023): vales e encostas suaves primeiro, encostas íngremes por último.

A cobertura exibida é **área efetivamente vista pela câmera**, calculada numa grade de
25 m sobre o cercamento, e não progresso ao longo da rota.

Um resultado que aparece sozinho: a rota dá 27,2 km e 38 min de voo, contra 35 min de
autonomia. Uma bateria cobre 93% da varredura, então são duas saídas.

## O gesto de socorro

**Braços estendidos na horizontal, formando um T.**

Escolhido por medição sobre a base UAV-GESTURE, e não por intuição. Entre 13 gestos
aéreos reais, o T venceu simultaneamente nos cinco critérios avaliados: maior abertura
horizontal vista de cima, maior distância da postura de repouso, melhor separabilidade,
melhor estabilidade temporal e melhor simetria entre os braços.

A razão é geométrica: visto da vertical, um braço erguido projeta-se sobre a própria
cabeça e some, enquanto um braço na horizontal projeta a envergadura inteira no solo.
O aceno clássico acima da cabeça, que funciona visto de frente, é uma escolha ruim para
um drone acima.

Método e limites em [ESCOLHA-DO-GESTO.md](ESCOLHA-DO-GESTO.md).

## Treinar os modelos

Os dois estágios usam modelos de imagem do [Teachable Machine](https://teachablemachine.withgoogle.com/train/image),
carregados na interface por link compartilhável ou pelos arquivos exportados
(`model.json`, `weights.bin`, `metadata.json`).

**Estágio 1, detecção de pessoa.** Suba as duas classes prontas de
`dataset-deteccao/`: `pessoa.zip` (305 imagens) e `sem_pessoa.zip` (301 imagens),
derivadas de filmagem real de drone. Procedência e limitações em
[dataset-deteccao/LEIA-ME.md](dataset-deteccao/LEIA-ME.md).

**Estágio 2, gesto.** Ainda precisa ser capturado. O ponto crítico: os dados devem vir de
uma câmera elevada apontada para baixo, aproximando a geometria de 25 m de altura. Uma
webcam frontal ao nível dos olhos serve para demonstrar a interface, nunca como modelo de
produção.

## Estrutura

```
index.html, app.js, style.css   protótipo
terrain.js                      elevação SRTM e curvas de nível da área (gerado)
dataset-deteccao/               classes prontas para o estágio 1
scripts/
  gen_terrain_data.py           busca a elevação e gera terrain.js
  build_detection_dataset.py    monta as classes do estágio 1
  analyze_uav_gestures.py       mede os 13 gestos e escolhe o de socorro
ESCOLHA-DO-GESTO.md             método e resultado da escolha do gesto
Relatorio_Tecnico_ResgatAr.docx relatório técnico
```

Os três scripts reproduzem os dados a partir das fontes originais, então a procedência é
auditável.

## Limitações

- A webcam vê de frente ao nível dos olhos; o drone vê de cima. O protótipo assume essa
  substituição de forma explícita, e ela não se sustenta num modelo de produção.
- O conjunto do estágio 1 vem de poucos vídeos, com quadros consecutivos quase idênticos,
  então a acurácia relatada pelo Teachable Machine é otimista.
- A separabilidade do gesto foi medida contra outros sinais aeronáuticos, não contra
  atividade humana comum, que é a fonte real de falso positivo numa busca.
- O voo é simulado em canvas 2D: não há controle de voo, telemetria de rádio nem desvio
  de obstáculo.

## Fontes

- Castellano, G., Carone, D., Scigliuto, F., Vessio, G. *Search-and-Rescue From Drones
  With Computer Vision*. Zenodo. https://doi.org/10.5281/zenodo.3924925
- Perera, A. G., Law, Y. W., Chahl, J. *UAV-GESTURE: A Dataset for UAV Control and
  Gesture Recognition*. ECCV Workshops, 2018.
- Liu, C., Szirányi, T. *Real-Time Human Detection and Gesture Recognition for On-Board
  UAV Rescue*. Sensors, 21(6), 2180, 2021.
- Ewers, J.-H., Anderson, D., Thomson, D. *Optimal path planning using psychological
  profiling in drone-assisted missing person search*. Advanced Control for Applications,
  5(4), e167, 2023.
- Schedl, D. C., Kurmi, I., Bimber, O. *An autonomous drone for search and rescue in
  forests using airborne optical sectioning*. Science Robotics, 6(55), 2021.
- Espressif Systems. *ESP-Drone*. Documentação oficial.

## Licença

Código e documentação sob licença MIT. Dados de terceiros incluídos ou derivados mantêm
os termos de origem, detalhados em [LICENSE](LICENSE).

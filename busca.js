// Tela reduzida do ResgatAr: prova que o modelo de gesto treinado funciona sobre um voo gravado.
// Sem mapa nem simulação de voo. O classificador recebe uma janela do quadro, a janela varre o
// vídeo em grade, trava onde encontra o gesto, acompanha a pessoa e exige confirmação sustentada.

const LIMIAR = 0.85;          // confiança mínima para contar como gesto
const SUSTENTAR_MS = 1500;    // tempo com confiança acima do limiar para declarar localizado
const ALFA = 0.35;            // suavização exponencial da confiança
const TRAVAR_EM = 0.5;        // confiança que interrompe a varredura e trava na posição
const INTERVALO_MS = 60;      // ritmo do laço de inferência

const els = {};
["video", "janela", "aviso", "registro", "estado", "estadoDetalhe", "link", "btnCarregar",
 "statusModelo", "arquivo", "statusVideo", "tamanho", "statusTamanho", "btnIniciar", "btnParar",
 "barraSocorro", "statusProb", "resultado", "resultadoImg", "resConf", "resTempo", "resPos",
 "resInf", "recorte"].forEach((id) => { els[id] = document.getElementById(id); });

const estado = {
  modelo: null,
  alvo: 0,
  rotulos: [],
  arquivoUrl: null,
  rodando: false,
  modo: "aguardando", // aguardando | procurando | confirmando | localizado
  janela: { cx: 0.5, cy: 0.5, frac: 0.13 },
  grade: [],
  gradeIdx: 0,
  suavizada: 0,
  sustentandoDesde: null,
  sonda: 0,
  latenciaMs: 0,
  timer: null,
};

// ---------- registro ----------
function registrar(msg, destaque = false) {
  const hora = new Date().toLocaleTimeString("pt-BR");
  const linha = document.createElement("div");
  linha.append(`${hora} `);
  const texto = destaque ? document.createElement("b") : document.createTextNode("");
  texto.textContent = msg;
  linha.append(texto);
  els.registro.prepend(linha);
}

function setEstado(modo, detalhe) {
  estado.modo = modo;
  const titulos = { aguardando: "Aguardando", procurando: "Procurando", confirmando: "Confirmando", localizado: "Localizado" };
  els.estado.dataset.modo = modo;
  els.estado.childNodes[0].nodeValue = titulos[modo];
  els.estadoDetalhe.textContent = detalhe || "";
}

// ---------- geometria da janela ----------
function retanguloRecorte(dx = 0, dy = 0) {
  const vw = els.video.videoWidth;
  const vh = els.video.videoHeight;
  const j = estado.janela;
  const lado = Math.min(vh, Math.round(vw * j.frac));
  const x = Math.round(Math.min(Math.max((j.cx + dx) * vw - lado / 2, 0), vw - lado));
  const y = Math.round(Math.min(Math.max((j.cy + dy) * vh - lado / 2, 0), vh - lado));
  return { x, y, lado };
}

function retanguloExibicao() {
  const caixa = els.video.getBoundingClientRect();
  const vw = els.video.videoWidth || 16;
  const vh = els.video.videoHeight || 9;
  const escala = Math.min(caixa.width / vw, caixa.height / vh);
  const w = vw * escala;
  const h = vh * escala;
  return { left: (caixa.width - w) / 2, top: (caixa.height - h) / 2, width: w, height: h };
}

function desenharJanela() {
  if (!els.video.videoWidth || estado.modo === "aguardando") { els.janela.hidden = true; return; }
  const r = retanguloRecorte();
  const d = retanguloExibicao();
  const k = d.width / els.video.videoWidth;
  els.janela.hidden = false;
  els.janela.style.left = `${d.left + r.x * k}px`;
  els.janela.style.top = `${d.top + r.y * k}px`;
  els.janela.style.width = `${r.lado * k}px`;
  els.janela.style.height = `${r.lado * k}px`;
  els.janela.classList.toggle("travada", estado.modo !== "procurando");
}

function recorte(dx = 0, dy = 0) {
  const r = retanguloRecorte(dx, dy);
  const c = els.recorte;
  c.width = 224;
  c.height = 224;
  c.getContext("2d").drawImage(els.video, r.x, r.y, r.lado, r.lado, 0, 0, 224, 224);
  return c;
}

// Posições da varredura: passo de meia janela, cobrindo o quadro inteiro, em zigue-zague.
function montarGrade() {
  const j = estado.janela;
  const aspecto = els.video.videoWidth / els.video.videoHeight;
  const passoX = j.frac / 2;
  const passoY = (j.frac * aspecto) / 2;
  const meiaX = j.frac / 2;
  const meiaY = (j.frac * aspecto) / 2;
  const pontos = [];
  let linha = 0;
  for (let cy = meiaY; cy <= 1 - meiaY + 1e-6; cy += passoY, linha++) {
    const xs = [];
    for (let cx = meiaX; cx <= 1 - meiaX + 1e-6; cx += passoX) xs.push(cx);
    if (linha % 2) xs.reverse();
    xs.forEach((cx) => pontos.push([cx, cy]));
  }
  estado.grade = pontos;
  estado.gradeIdx = 0;
}

// ---------- laço ----------
async function prob(dx = 0, dy = 0) {
  const t0 = performance.now();
  const p = await estado.modelo.predict(recorte(dx, dy));
  const lat = performance.now() - t0;
  estado.latenciaMs = estado.latenciaMs ? estado.latenciaMs * 0.8 + lat * 0.2 : lat;
  return p[estado.alvo].probability;
}

let ocupado = false;
async function passo() {
  if (!estado.rodando || ocupado || els.video.paused || !els.video.videoWidth) return;
  ocupado = true;
  try {
    const j = estado.janela;
    if (estado.modo === "procurando") {
      const [cx, cy] = estado.grade[estado.gradeIdx];
      j.cx = cx;
      j.cy = cy;
      estado.gradeIdx = (estado.gradeIdx + 1) % estado.grade.length;
      const p = await prob();
      mostrarProb(p);
      if (p >= TRAVAR_EM) {
        estado.suavizada = p;
        estado.sustentandoDesde = null;
        setEstado("confirmando", `candidato em (${cx.toFixed(2)}, ${cy.toFixed(2)})`);
        registrar(`candidato com ${Math.round(p * 100)}% em (${cx.toFixed(2)}, ${cy.toFixed(2)}); travando`);
      }
    } else if (estado.modo === "confirmando") {
      const p = await prob();
      estado.suavizada = estado.suavizada * (1 - ALFA) + p * ALFA;
      mostrarProb(estado.suavizada);

      // acompanhamento: uma sonda vizinha por ciclo, passo só se for claramente melhor
      const aspecto = els.video.videoWidth / els.video.videoHeight;
      const dirs = [[0.5, 0], [0, 0.5 * aspecto], [-0.5, 0], [0, -0.5 * aspecto],
                    [1, 0], [0, aspecto], [-1, 0], [0, -aspecto]];
      const [dx, dy] = dirs[estado.sonda++ % dirs.length];
      const la = await prob(dx * j.frac, dy * j.frac);
      if (la > p + 0.15 && la > 0.5) {
        j.cx = Math.min(1, Math.max(0, j.cx + dx * j.frac));
        j.cy = Math.min(1, Math.max(0, j.cy + dy * j.frac));
      }

      const agora = performance.now();
      if (estado.suavizada >= LIMIAR) {
        if (estado.sustentandoDesde === null) estado.sustentandoDesde = agora;
        const dur = agora - estado.sustentandoDesde;
        els.estadoDetalhe.textContent = `${(dur / 1000).toFixed(1)} s de ${SUSTENTAR_MS / 1000} s acima de ${Math.round(LIMIAR * 100)}%`;
        if (dur >= SUSTENTAR_MS) localizado();
      } else if (estado.suavizada < TRAVAR_EM * 0.5) {
        registrar(`confiança caiu para ${Math.round(estado.suavizada * 100)}%; volta a procurar`);
        estado.sustentandoDesde = null;
        setEstado("procurando", "varrendo o quadro");
      } else {
        if (estado.sustentandoDesde !== null) registrar("confirmação interrompida");
        estado.sustentandoDesde = null;
        els.estadoDetalhe.textContent = "acompanhando o candidato";
      }
    }
    desenharJanela();
  } finally {
    ocupado = false;
  }
}

function mostrarProb(p) {
  els.barraSocorro.style.width = `${Math.round(p * 100)}%`;
  els.statusProb.textContent = `${estado.rotulos[estado.alvo] || "socorro"}: ${Math.round(p * 100)}%  (${Math.round(estado.latenciaMs)} ms por inferência)`;
}

function localizado() {
  estado.rodando = false;
  clearInterval(estado.timer);
  els.video.pause();
  const j = estado.janela;
  setEstado("localizado", "pessoa pedindo socorro; missão encerrada");
  registrar(`LOCALIZADO com ${Math.round(estado.suavizada * 100)}% aos ${els.video.currentTime.toFixed(1)} s do vídeo`, true);
  els.resultadoImg.src = recorte().toDataURL("image/jpeg", 0.9);
  els.resConf.textContent = `${Math.round(estado.suavizada * 100)}% sustentado por ${SUSTENTAR_MS / 1000} s`;
  els.resTempo.textContent = `${els.video.currentTime.toFixed(1)} s do vídeo`;
  els.resPos.textContent = `x ${Math.round(j.cx * 100)}%, y ${Math.round(j.cy * 100)}%, janela ${Math.round(j.frac * 100)}%`;
  els.resInf.textContent = `${Math.round(estado.latenciaMs)} ms por quadro, ${tf.getBackend()}`;
  els.resultado.classList.add("ativo");
  els.btnIniciar.disabled = false;
  els.btnIniciar.textContent = "Nova busca";
  els.btnParar.disabled = true;
  desenharJanela();
}

function iniciar() {
  if (!estado.modelo || !els.video.videoWidth) return;
  els.resultado.classList.remove("ativo");
  estado.rodando = true;
  estado.suavizada = 0;
  estado.sustentandoDesde = null;
  montarGrade();
  setEstado("procurando", "varrendo o quadro");
  registrar(`missão iniciada: grade de ${estado.grade.length} posições, janela de ${Math.round(estado.janela.frac * 100)}% da largura`);
  els.video.currentTime = 0;
  els.video.play().catch(() => {
    els.aviso.textContent = "Clique aqui para reproduzir o vídeo";
    els.aviso.hidden = false;
  });
  els.btnIniciar.disabled = true;
  els.btnParar.disabled = false;
  clearInterval(estado.timer);
  estado.timer = setInterval(passo, INTERVALO_MS);
}

function parar(motivo) {
  estado.rodando = false;
  clearInterval(estado.timer);
  els.video.pause();
  setEstado("aguardando", motivo || "busca interrompida");
  registrar(motivo || "busca interrompida");
  els.btnIniciar.disabled = false;
  els.btnIniciar.textContent = "Iniciar missão";
  els.btnParar.disabled = true;
  desenharJanela();
}

// ---------- entradas ----------
els.btnCarregar.addEventListener("click", async () => {
  let raiz = els.link.value.trim();
  if (!raiz) return;
  if (!raiz.endsWith("/")) raiz += "/";
  els.statusModelo.textContent = "Carregando...";
  try {
    estado.modelo = await tmImage.load(raiz + "model.json", raiz + "metadata.json");
    estado.rotulos = estado.modelo.getClassLabels();
    const negacao = /^(sem|nao|não|no|not)[\s_-]/i;
    const dica = /socorro|ajuda|help|sos|emerg/i;
    let idx = estado.rotulos.findIndex((l) => dica.test(l) && !negacao.test(l));
    if (idx < 0) idx = 0;
    estado.alvo = idx;
    els.statusModelo.textContent = `Carregado: ${estado.rotulos.join(", ")}. Alvo: ${estado.rotulos[idx]}.`;
    registrar(`modelo carregado (${estado.rotulos.length} classes), alvo "${estado.rotulos[idx]}"`);
  } catch (e) {
    estado.modelo = null;
    els.statusModelo.textContent = "Falha ao carregar. Confira o link (termina em /).";
  }
  atualizarBotoes();
});

els.arquivo.addEventListener("change", () => {
  const f = els.arquivo.files[0];
  if (!f) return;
  if (estado.arquivoUrl) URL.revokeObjectURL(estado.arquivoUrl);
  estado.arquivoUrl = URL.createObjectURL(f);
  els.video.src = estado.arquivoUrl;
  els.video.load();
  els.statusVideo.textContent = f.name;
  els.aviso.hidden = true;
  parar("vídeo carregado");
  registrar(`vídeo: ${f.name}`);
});

els.video.addEventListener("loadedmetadata", () => {
  els.aviso.hidden = true;
  els.statusVideo.textContent += ` (${els.video.videoWidth}x${els.video.videoHeight}, ${els.video.duration.toFixed(0)} s)`;
  els.video.play().then(() => els.video.pause()).catch(() => {});
  atualizarBotoes();
});

els.aviso.addEventListener("click", () => {
  els.video.play().then(() => { els.aviso.hidden = true; }).catch(() => {});
});

els.tamanho.addEventListener("input", () => {
  estado.janela.frac = Number(els.tamanho.value) / 100;
  els.statusTamanho.textContent = `${els.tamanho.value}% da largura`;
  if (estado.modo === "procurando") montarGrade();
  desenharJanela();
});

els.btnIniciar.addEventListener("click", iniciar);
els.btnParar.addEventListener("click", () => parar("busca interrompida pelo operador"));
window.addEventListener("resize", desenharJanela);

function atualizarBotoes() {
  const pronto = !!estado.modelo && !!els.video.videoWidth;
  els.btnIniciar.disabled = !pronto || estado.rodando;
  if (pronto && estado.modo === "aguardando") els.estadoDetalhe.textContent = "pronto para iniciar";
}

// ResgatAr: protótipo de drone de busca com reconhecimento de gesto de socorro.
// Sensor: webcam (simula câmera de bordo) | Processamento: modelo Teachable Machine (TinyML)
// Atuador: simulação de voo sobre terreno real | Feedback: painel de alerta da base de resgate
// Dados de terreno e rota: terrain.js (elevação SRTM real da área de busca)

const DEFAULT_GEOFENCE = {
  nw: { lat: -29.3300, lng: -52.0100 },
  se: { lat: -29.3450, lng: -51.9900 },
};

// Autonomia de 35 min corresponde a uma plataforma de mapeamento (classe DJI Neo ou superior),
// que é a evolução prevista no relatório. O ESP-Drone da versão caseira voa cerca de 5 min.
//
// O voo usa seguimento de terreno: a altura é mantida constante em relação ao solo usando o
// modelo de elevação embarcado (MISSION_TERRAIN). Sem isso, uma altitude barométrica fixa sobre
// um relevo de 155 m a 472 m deixaria o drone a mais de 400 m do solo nos vales, acima do teto
// legal de 120 m (RBAC-E 94 da ANAC e ICA 100-40 do DECEA).
// Operação em dois estágios. A 100 m, uma pessoa vista de cima tem 4,8 px de ombro a ombro e
// os braços abertos somam 18 px, com 1,1 px de espessura: o gesto é fisicamente irresolvível
// nessa altura, por mais dados de treino que existam. Por isso o drone detecta o candidato na
// altura de varredura e desce para inspecionar, como faz Liu e Szirányi (Sensors, 2021).
const DRONE = {
  cruiseSpeedMps: 12,
  enduranceMin: 35,
  searchAglM: 100,
  inspectAglM: 25,
  descentRateMps: 3,
  climbRateMps: 2.5,
  legalCeilingM: 120,
  cameraFovDeg: 84,
  cameraPixels: 1920,
  sidelap: 0.20,
  inspectTimeoutMs: 20000,
  // Lado, no solo, da janela que o classificador de gesto recebe. O mesmo valor usado ao
  // recortar os dados de treino (scripts/build_gesture_dataset.py): a pessoa ocupa cerca de
  // um quarto do recorte, o suficiente para os braços terem espessura no 224x224 do modelo.
  inspectFootprintM: 6,
};

const ARM_SPAN_M = 1.7;      // envergadura de braços abertos
const ARM_THICKNESS_M = 0.10;

// Largura da faixa vista pela câmera no solo, na altura informada.
function swathWidthM(aglM) {
  const agl = aglM == null ? DRONE.searchAglM : aglM;
  return 2 * agl * Math.tan((DRONE.cameraFovDeg * Math.PI) / 360);
}
function trackSpacingM() {
  return swathWidthM(DRONE.searchAglM) * (1 - DRONE.sidelap);
}
function groundSampleDistanceM(aglM) {
  return swathWidthM(aglM) / DRONE.cameraPixels;
}
// Quantos pixels os braços abertos ocupam na altura atual: é este número que decide
// se o gesto pode ou não ser classificado.
function gesturePixels(aglM) {
  const gsd = groundSampleDistanceM(aglM);
  return { span: ARM_SPAN_M / gsd, thickness: ARM_THICKNESS_M / gsd };
}
function gestureResolvable(aglM) {
  return gesturePixels(aglM).thickness >= 3;
}

const DETECTION = {
  emaAlpha: 0.35,
  rearmMs: 5000,
  logLimit: 14,
};

const EARTH_R = 6371000;

// Um modelo por estágio: "deteccao" roda na altura de varredura e responde "há pessoa aqui?";
// "gesto" só roda depois da descida, onde os braços têm pixels suficientes para classificar.
const state = {
  models: {
    deteccao: { model: null, labels: [], count: 0, targetIndex: 0 },
    gesto: { model: null, labels: [], count: 0, targetIndex: 0 },
  },
  webcamStream: null,
  streamPc: null,

  simSpeed: 10, // multiplica o tempo da simulação de voo, nunca o da detecção

  threshold: 0.85,
  sustainMs: 1500,
  sustainStartedAt: null,
  sustainPeak: 0,
  smoothedProb: 0,
  rearmUntil: 0,

  inference: { fps: 0, latencyMs: 0, frames: 0, windowStart: 0 },

  // Janela de inspeção: centro em fração do quadro (arrastável) e lado em fração da largura.
  // O lado é recalculado pela altura ao iniciar a inspeção e pode ser ajustado à mão.
  inspectWindow: { cx: 0.5, cy: 0.5, frac: 0.13 },
  fileUrl: null,

  mission: {
    running: false,
    // idle | searching | descending | inspecting | climbing | alert | swept | depleted
    droneState: "idle",
    aglM: DRONE.searchAglM,
    candidatePos: null,
    inspectStartedAt: null,
    elapsedMs: 0,
    flightMs: 0,
    geofence: { nw: { ...DEFAULT_GEOFENCE.nw }, se: { ...DEFAULT_GEOFENCE.se } },
    path: [],
    pathIndex: 0,
    segmentT: 0,
    dronePos: null,
    headingRad: 0,
    hasTerrain: true,
  },

  alerts: [],
  detections: [],
};

const els = {};
[
  "webcam", "videoOverlay", "btnCamera", "btnSimulateCandidate", "btnSimulateGesture",
  "cameraSource", "cameraSourceRow", "streamUrl", "btnStream", "streamStatus",
  "urlDeteccao", "btnUrlDeteccao", "filesDeteccao", "btnFilesDeteccao", "statusDeteccao",
  "urlGesto", "btnUrlGesto", "filesGesto", "btnFilesGesto", "statusGesto",
  "predictions", "predictionsStage",
  "threshold", "thresholdValue", "sustain", "sustainValue",
  "sustainRing", "pipelineState", "infFps", "infLatency", "detectionLog",
  "mapCanvas", "droneState", "dronePos", "missionTime",
  "droneGround", "droneAgl", "droneGesturePx", "droneBattery", "droneCoverage", "terrainNote",
  "cornerNW", "cornerSE", "btnApplyGeofence", "routeInfo", "simSpeed",
  "btnStartMission", "btnPauseMission", "btnResumeSearch", "btnExportReport",
  "alertLog", "flowStrip", "snapshotCanvas", "inferCanvas",
  "inspectWindow", "inspectRow", "inspectFrac", "inspectInfo", "fileVideo",
  "shotModal", "shotModalImg", "shotModalMeta", "shotModalClose",
].forEach((id) => { els[id] = document.getElementById(id); });

const ctx = els.mapCanvas.getContext("2d");

// ---------- Geo helpers ----------
function metersBetween(a, b) {
  const midLat = ((a.lat + b.lat) / 2) * Math.PI / 180;
  const dy = (b.lat - a.lat) * Math.PI / 180 * EARTH_R;
  const dx = (b.lng - a.lng) * Math.PI / 180 * EARTH_R * Math.cos(midLat);
  return Math.hypot(dx, dy);
}

function parseLatLng(text) {
  const [lat, lng] = text.split(",").map((s) => parseFloat(s.trim()));
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) throw new Error("coordenada inválida");
  return { lat, lng };
}

function latLngToXY(lat, lng) {
  const { nw, se } = state.mission.geofence;
  const pad = 34;
  const fx = (lng - nw.lng) / (se.lng - nw.lng);
  const fy = (lat - nw.lat) / (se.lat - nw.lat);
  return {
    x: pad + fx * (els.mapCanvas.width - pad * 2),
    y: pad + fy * (els.mapCanvas.height - pad * 2),
  };
}

// Elevação do solo por interpolação bilinear na grade SRTM real.
function groundElevationAt(lat, lng) {
  const t = MISSION_TERRAIN;
  const fx = ((lng - t.lngMin) / (t.lngMax - t.lngMin)) * (t.nx - 1);
  const fy = ((t.latMax - lat) / (t.latMax - t.latMin)) * (t.ny - 1);
  const cx = Math.min(Math.max(fx, 0), t.nx - 1);
  const cy = Math.min(Math.max(fy, 0), t.ny - 1);
  const x0 = Math.floor(cx), y0 = Math.floor(cy);
  const x1 = Math.min(x0 + 1, t.nx - 1), y1 = Math.min(y0 + 1, t.ny - 1);
  const tx = cx - x0, ty = cy - y0;
  const top = t.grid[y0][x0] * (1 - tx) + t.grid[y0][x1] * tx;
  const bot = t.grid[y1][x0] * (1 - tx) + t.grid[y1][x1] * tx;
  return top * (1 - ty) + bot * ty;
}

// ---------- Rotas ----------
function isDefaultGeofence() {
  const { nw, se } = state.mission.geofence;
  return (
    Math.abs(nw.lat - DEFAULT_GEOFENCE.nw.lat) < 1e-6 &&
    Math.abs(nw.lng - DEFAULT_GEOFENCE.nw.lng) < 1e-6 &&
    Math.abs(se.lat - DEFAULT_GEOFENCE.se.lat) < 1e-6 &&
    Math.abs(se.lng - DEFAULT_GEOFENCE.se.lng) < 1e-6
  );
}

// Declividade local do terreno em graus, por diferenças centrais na grade SRTM.
function slopeDegAt(lat, lng) {
  const t = MISSION_TERRAIN;
  const dLat = (t.latMax - t.latMin) / (t.ny - 1);
  const dLng = (t.lngMax - t.lngMin) / (t.nx - 1);
  const stepY = metersBetween({ lat, lng }, { lat: lat + dLat, lng });
  const stepX = metersBetween({ lat, lng }, { lat, lng: lng + dLng });
  const dzY = groundElevationAt(lat + dLat, lng) - groundElevationAt(lat - dLat, lng);
  const dzX = groundElevationAt(lat, lng + dLng) - groundElevationAt(lat, lng - dLng);
  const grad = Math.hypot(dzY / (2 * Math.max(stepY, 1)), dzX / (2 * Math.max(stepX, 1)));
  return (Math.atan(grad) * 180) / Math.PI;
}

// Probabilidade relativa de encontrar a pessoa numa faixa, a partir do relevo.
// Perfil de comportamento de pessoa perdida (Ewers et al., 2023): tende a descer para
// drenagens e vales e a evitar encostas íngremes, onde caminhar e descansar é inviável.
function trackPriorityScore(fixed, fromVar, toVar, orientation) {
  const t = MISSION_TERRAIN;
  const elevRange = Math.max(t.elevMax - t.elevMin, 1);
  const samples = 12;
  let elevSum = 0;
  let slopeSum = 0;
  for (let i = 0; i <= samples; i++) {
    const v = fromVar + ((toVar - fromVar) * i) / samples;
    const lat = orientation === "ew" ? fixed : v;
    const lng = orientation === "ew" ? v : fixed;
    elevSum += (groundElevationAt(lat, lng) - t.elevMin) / elevRange;
    slopeSum += Math.min(slopeDegAt(lat, lng) / 35, 1);
  }
  const normElev = elevSum / (samples + 1);
  const normSlope = slopeSum / (samples + 1);
  return 0.6 * (1 - normElev) + 0.4 * (1 - normSlope);
}

// Varredura paralela (parallel track, padrão IAMSAR para cobertura de área).
// O espaçamento vem da faixa da câmera, e as faixas são agrupadas em zonas de
// prioridade pelo relevo: a zona mais provável é varrida primeiro.
function buildTracks(orientation) {
  const { nw, se } = state.mission.geofence;
  const acrossM = orientation === "ew"
    ? metersBetween({ lat: nw.lat, lng: nw.lng }, { lat: se.lat, lng: nw.lng })
    : metersBetween({ lat: nw.lat, lng: nw.lng }, { lat: nw.lat, lng: se.lng });
  const count = Math.max(2, Math.round(acrossM / trackSpacingM()));

  const tracks = [];
  for (let i = 0; i < count; i++) {
    const f = (i + 0.5) / count;
    if (orientation === "ew") {
      const lat = nw.lat + (se.lat - nw.lat) * f;
      tracks.push({
        a: { lat, lng: nw.lng },
        b: { lat, lng: se.lng },
        score: state.mission.hasTerrain ? trackPriorityScore(lat, nw.lng, se.lng, "ew") : 0.5,
      });
    } else {
      const lng = nw.lng + (se.lng - nw.lng) * f;
      tracks.push({
        a: { lat: nw.lat, lng },
        b: { lat: se.lat, lng },
        score: state.mission.hasTerrain ? trackPriorityScore(lng, nw.lat, se.lat, "ns") : 0.5,
      });
    }
  }
  return tracks;
}

function scoreSpread(tracks) {
  const mean = tracks.reduce((s, t) => s + t.score, 0) / tracks.length;
  return Math.sqrt(tracks.reduce((s, t) => s + (t.score - mean) ** 2, 0) / tracks.length);
}

function generateParallelPath() {
  // Faixas paralelas à direção de menor variação do relevo perdem poder de discriminação:
  // cada faixa acaba com a mesma média. Escolhemos a orientação que melhor separa as zonas.
  const candidates = ["ew", "ns"].map((o) => {
    const tracks = buildTracks(o);
    return { orientation: o, tracks, spread: scoreSpread(tracks) };
  });
  const best = state.mission.hasTerrain
    ? candidates.reduce((a, b) => (b.spread > a.spread ? b : a))
    : candidates[0];

  state.mission.orientation = best.orientation;
  const tracks = best.tracks;
  const count = tracks.length;

  // zonas de prioridade por tercil do escore, preservando a ordem geográfica dentro da zona
  const sorted = [...tracks].sort((a, b) => b.score - a.score);
  const cut1 = sorted[Math.floor(count / 3)] ? sorted[Math.floor(count / 3)].score : 0;
  const cut2 = sorted[Math.floor((2 * count) / 3)] ? sorted[Math.floor((2 * count) / 3)].score : 0;
  tracks.forEach((t) => {
    t.tier = t.score >= cut1 ? 1 : t.score >= cut2 ? 2 : 3;
  });

  const path = [];
  let flip = false;
  [1, 2, 3].forEach((tier) => {
    tracks.filter((t) => t.tier === tier).forEach((t) => {
      const ends = flip ? [t.b, t.a] : [t.a, t.b];
      path.push({ ...ends[0], tier }, { ...ends[1], tier });
      flip = !flip;
    });
  });

  return path;
}

function generateSearchPath() {
  state.mission.hasTerrain = isDefaultGeofence();
  return generateParallelPath();
}

// ---------- Cobertura real de área ----------
// Grade sobre o cercamento marcando o que a câmera efetivamente enxergou, e não
// apenas o progresso ao longo da rota.
const coverage = {
  cellM: 25,
  gx: 0,
  gy: 0,
  cells: null,
  covered: 0,
  canvas: document.createElement("canvas"),
};

function resetCoverage() {
  const { nw, se } = state.mission.geofence;
  const widthM = metersBetween({ lat: nw.lat, lng: nw.lng }, { lat: nw.lat, lng: se.lng });
  const heightM = metersBetween({ lat: nw.lat, lng: nw.lng }, { lat: se.lat, lng: nw.lng });
  coverage.gx = Math.max(1, Math.ceil(widthM / coverage.cellM));
  coverage.gy = Math.max(1, Math.ceil(heightM / coverage.cellM));
  coverage.cells = new Uint8Array(coverage.gx * coverage.gy);
  coverage.covered = 0;
  coverage.canvas.width = els.mapCanvas.width;
  coverage.canvas.height = els.mapCanvas.height;
  coverage.canvas.getContext("2d").clearRect(0, 0, coverage.canvas.width, coverage.canvas.height);
}

function stampCoverageAt(pos) {
  const { nw, se } = state.mission.geofence;
  const radiusM = swathWidthM(state.mission.aglM) / 2;
  const cctx = coverage.canvas.getContext("2d");
  const reach = Math.ceil(radiusM / coverage.cellM);

  const fx = ((pos.lng - nw.lng) / (se.lng - nw.lng)) * coverage.gx;
  const fy = ((pos.lat - nw.lat) / (se.lat - nw.lat)) * coverage.gy;
  const cx = Math.floor(fx);
  const cy = Math.floor(fy);

  for (let y = cy - reach; y <= cy + reach; y++) {
    if (y < 0 || y >= coverage.gy) continue;
    for (let x = cx - reach; x <= cx + reach; x++) {
      if (x < 0 || x >= coverage.gx) continue;
      const idx = y * coverage.gx + x;
      if (coverage.cells[idx]) continue;
      if (Math.hypot((x + 0.5 - fx) * coverage.cellM, (y + 0.5 - fy) * coverage.cellM) > radiusM) continue;

      coverage.cells[idx] = 1;
      coverage.covered++;

      const latA = nw.lat + ((se.lat - nw.lat) * y) / coverage.gy;
      const latB = nw.lat + ((se.lat - nw.lat) * (y + 1)) / coverage.gy;
      const lngA = nw.lng + ((se.lng - nw.lng) * x) / coverage.gx;
      const lngB = nw.lng + ((se.lng - nw.lng) * (x + 1)) / coverage.gx;
      const p0 = latLngToXY(latA, lngA);
      const p1 = latLngToXY(latB, lngB);
      cctx.fillStyle = "rgba(255, 106, 26, 0.22)";
      cctx.fillRect(p0.x, p0.y, p1.x - p0.x + 0.6, p1.y - p0.y + 0.6);
    }
  }
}

function stampCoverageSegment(from, to) {
  const dist = metersBetween(from, to);
  const steps = Math.max(1, Math.ceil(dist / (swathWidthM(state.mission.aglM) / 3)));
  for (let i = 1; i <= steps; i++) {
    stampCoverageAt({
      lat: from.lat + (to.lat - from.lat) * (i / steps),
      lng: from.lng + (to.lng - from.lng) * (i / steps),
    });
  }
}

function coveragePercent() {
  if (!coverage.cells || !coverage.cells.length) return 0;
  return (coverage.covered / coverage.cells.length) * 100;
}

function routeStats() {
  const p = state.mission.path;
  let meters = 0;
  // rota aberta: o drone não retorna ao ponto inicial ao terminar a varredura
  for (let i = 0; i < p.length - 1; i++) meters += metersBetween(p[i], p[i + 1]);
  const minutes = meters / DRONE.cruiseSpeedMps / 60;
  return {
    km: meters / 1000,
    minutes,
    tracks: p.length / 2,
    sorties: Math.ceil(minutes / DRONE.enduranceMin),
  };
}

function resetRoute() {
  const m = state.mission;
  m.path = generateSearchPath();
  m.pathIndex = 0;
  m.segmentT = 0;
  m.dronePos = { ...m.path[0] };
  els.terrainNote.hidden = m.hasTerrain;
  resetCoverage();

  const r = routeStats();
  els.routeInfo.textContent =
    `${r.tracks} faixas a cada ${Math.round(trackSpacingM())} m ` +
    `(câmera enxerga ${Math.round(swathWidthM(DRONE.searchAglM))} m a ${DRONE.searchAglM} m do solo). ` +
    `${r.km.toFixed(1)} km, ${Math.round(r.minutes)} min a ${DRONE.cruiseSpeedMps} m/s, ` +
    `${r.sorties} ${r.sorties > 1 ? "saídas" : "saída"} de ${DRONE.enduranceMin} min.`;
}

function applyGeofence() {
  try {
    const nw = parseLatLng(els.cornerNW.value);
    const se = parseLatLng(els.cornerSE.value);
    state.mission.geofence = { nw, se };
    resetRoute();
    drawMap();
    updateTelemetry();
  } catch (e) {
    els.terrainNote.hidden = false;
    els.terrainNote.textContent = "Coordenadas inválidas. Use o formato: lat, lng";
  }
}

// ---------- Voo ----------
// Seguimento de terreno: o controlador mantém a altura alvo sobre o solo consultando o
// modelo de elevação embarcado, então a altitude barométrica acompanha o relevo.
function currentGroundElev() {
  const m = state.mission;
  if (!m.dronePos) return null;
  return groundElevationAt(m.dronePos.lat, m.dronePos.lng);
}

function currentFlightAltitude() {
  const g = currentGroundElev();
  return g == null ? null : g + state.mission.aglM;
}

function targetAglForState() {
  const s = state.mission.droneState;
  if (s === "descending" || s === "inspecting") return DRONE.inspectAglM;
  return DRONE.searchAglM;
}

// Movimento vertical. O drone só avança pela rota em varredura: durante descida, inspeção
// e subida ele mantém a posição horizontal sobre o candidato.
function stepAltitude(dtMs) {
  const m = state.mission;
  const target = targetAglForState();
  const diff = target - m.aglM;
  if (Math.abs(diff) < 0.4) {
    m.aglM = target;
    return true;
  }
  const rate = diff < 0 ? DRONE.descentRateMps : DRONE.climbRateMps;
  const step = (rate * dtMs) / 1000;
  m.aglM += Math.sign(diff) * Math.min(step, Math.abs(diff));
  return false;
}

function batteryLevel() {
  return Math.max(0, 1 - state.mission.flightMs / (DRONE.enduranceMin * 60000));
}

function stepDrone(dtMs) {
  const m = state.mission;
  if (m.droneState !== "searching" || m.path.length < 2) return;

  let remaining = (DRONE.cruiseSpeedMps * dtMs) / 1000;
  let guard = 0;
  const origin = { ...m.dronePos };

  while (remaining > 0 && guard++ < 128) {
    if (m.pathIndex >= m.path.length - 1) {
      completeSweep();
      break;
    }

    const a = m.path[m.pathIndex];
    const b = m.path[m.pathIndex + 1];
    const segLen = Math.max(metersBetween(a, b), 0.5);
    const stepFrac = remaining / segLen;

    if (m.segmentT + stepFrac < 1) {
      m.segmentT += stepFrac;
      remaining = 0;
    } else {
      remaining -= (1 - m.segmentT) * segLen;
      m.segmentT = 0;
      m.pathIndex += 1;
    }

    const p0 = m.path[Math.min(m.pathIndex, m.path.length - 2)];
    const p1 = m.path[Math.min(m.pathIndex + 1, m.path.length - 1)];
    m.dronePos = {
      lat: p0.lat + (p1.lat - p0.lat) * m.segmentT,
      lng: p0.lng + (p1.lng - p0.lng) * m.segmentT,
    };
    m.headingRad = Math.atan2(p1.lat - p0.lat, p1.lng - p0.lng);
  }

  stampCoverageSegment(origin, m.dronePos);
}

// ---------- Estágio 1 para estágio 2: descida de inspeção ----------
function beginDescent(confidence) {
  const m = state.mission;
  if (m.droneState !== "searching") return;
  m.droneState = "descending";
  m.candidatePos = { ...m.dronePos };
  state.sustainStartedAt = null;
  setSustainProgress(0);
  logDetection(
    "start",
    `candidato detectado, descendo de ${Math.round(m.aglM)} m para ${DRONE.inspectAglM} m`,
    confidence
  );
  syncButtons();
  syncFlowToMission();
}

function beginInspection() {
  const m = state.mission;
  m.droneState = "inspecting";
  // Tempo real, nunca simulado: quem faz o gesto é uma pessoa, e o relógio dela não acelera.
  m.inspectStartedAt = performance.now();
  state.sustainStartedAt = null;
  state.smoothedProb = 0;
  state.inspectWindow.frac = inspectFracFor(m.aglM);
  els.inspectFrac.value = Math.round(state.inspectWindow.frac * 100);
  const px = gesturePixels(m.aglM);
  logDetection(
    "info",
    `inspecionando a ${Math.round(m.aglM)} m: braços com ${px.span.toFixed(0)} px de envergadura, ` +
    `janela de ${DRONE.inspectFootprintM} m no solo`
  );
  syncButtons();
  syncFlowToMission();
  updatePipelineState();
}

// Sem confirmação do gesto dentro da janela, o candidato é descartado e o drone sobe.
// A descida custou tempo e bateria: é o preço de um falso positivo no estágio 1.
function checkInspectTimeout() {
  const m = state.mission;
  if (m.inspectStartedAt == null) return;
  if (performance.now() - m.inspectStartedAt < DRONE.inspectTimeoutMs) return;
  logDetection("reject", "gesto não confirmado na inspeção, candidato descartado");
  beginClimb();
}

function beginClimb() {
  const m = state.mission;
  m.droneState = "climbing";
  m.inspectStartedAt = null;
  m.candidatePos = null;
  state.sustainStartedAt = null;
  setSustainProgress(0);
  state.smoothedProb = 0;
  syncButtons();
  syncFlowToMission();
}

function resumeSweep() {
  const m = state.mission;
  m.droneState = "searching";
  state.rearmUntil = performance.now() + DETECTION.rearmMs;
  logDetection("rearm", `de volta a ${Math.round(m.aglM)} m, varredura retomada`);
  syncButtons();
  syncFlowToMission();
}

function completeSweep() {
  const m = state.mission;
  m.running = false;
  m.droneState = "swept";
  logDetection("info", `varredura concluída, ${Math.round(coveragePercent())}% da área coberta`);
  syncButtons();
  updateTelemetry(true);
}

// ---------- Desenho ----------
function drawMap() {
  const m = state.mission;
  const w = els.mapCanvas.width;
  const h = els.mapCanvas.height;
  // fundo opaco: o mapa é exportado como PNG no relatório de missão
  ctx.fillStyle = "#0A0A08";
  ctx.fillRect(0, 0, w, h);

  if (m.hasTerrain) drawTerrain();
  ctx.drawImage(coverage.canvas, 0, 0);
  drawGeofence();
  drawPath();
  drawAlertMarkers();
  drawDrone();
}

function drawTerrain() {
  const range = Math.max(MISSION_TERRAIN.elevMax - MISSION_TERRAIN.elevMin, 1);
  TERRAIN_CONTOURS.forEach((c) => {
    const rel = (c.elev - MISSION_TERRAIN.elevMin) / range;
    ctx.strokeStyle = `rgba(185, 178, 156, ${0.07 + rel * 0.13})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    c.pts.forEach(([lat, lng], i) => {
      const xy = latLngToXY(lat, lng);
      if (i === 0) ctx.moveTo(xy.x, xy.y);
      else ctx.lineTo(xy.x, xy.y);
    });
    ctx.stroke();
  });
}

function drawGeofence() {
  const { nw, se } = state.mission.geofence;
  const a = latLngToXY(nw.lat, nw.lng);
  const b = latLngToXY(se.lat, se.lng);
  ctx.strokeStyle = "#B9B29C";
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 2;
  ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
  ctx.setLineDash([]);

  ctx.fillStyle = "#8A8C82";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillText("CERCAMENTO ELETRÔNICO", a.x + 6, a.y - 10);
}

// As faixas são coloridas pela zona de prioridade calculada a partir do relevo:
// quanto mais claro, maior a probabilidade estimada de encontrar a pessoa.
const TIER_STROKE = {
  1: "rgba(237, 233, 221, 0.72)",
  2: "rgba(237, 233, 221, 0.40)",
  3: "rgba(237, 233, 221, 0.20)",
};

function drawPath() {
  const m = state.mission;
  if (m.path.length < 2) return;

  for (let i = 0; i < m.path.length - 1; i++) {
    const p0 = m.path[i];
    const p1 = m.path[i + 1];
    const xy0 = latLngToXY(p0.lat, p0.lng);
    const xy1 = latLngToXY(p1.lat, p1.lng);
    const isTrack = p0.tier != null && p0.tier === p1.tier;

    ctx.beginPath();
    ctx.moveTo(xy0.x, xy0.y);
    ctx.lineTo(xy1.x, xy1.y);

    if (i < m.pathIndex) {
      ctx.strokeStyle = "rgba(255, 106, 26, 0.85)";
      ctx.lineWidth = 2.2;
      ctx.setLineDash([]);
    } else if (isTrack) {
      ctx.strokeStyle = TIER_STROKE[p0.tier] || TIER_STROKE[3];
      ctx.lineWidth = 1.8;
      ctx.setLineDash([]);
    } else {
      ctx.strokeStyle = "rgba(138, 140, 130, 0.30)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawAlertMarkers() {
  state.alerts.forEach((a) => {
    const xy = latLngToXY(a.lat, a.lng);
    ctx.strokeStyle = a.resolved ? "#8A9A6E" : "#E8491D";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(xy.x, xy.y, 11, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = a.resolved ? "#8A9A6E" : "#E8491D";
    ctx.beginPath();
    ctx.arc(xy.x, xy.y, 5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawDrone() {
  const m = state.mission;
  if (!m.dronePos) return;
  const xy = latLngToXY(m.dronePos.lat, m.dronePos.lng);
  const color =
    m.droneState === "alert" ? "#E8491D" :
    m.droneState === "descending" || m.droneState === "inspecting" || m.droneState === "climbing" ? "#FF6A1A" :
    m.droneState === "depleted" ? "#8A8C82" : "#EDE9DD";

  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.3;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(xy.x, xy.y, 16, 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalAlpha = 1;

  ctx.save();
  ctx.translate(xy.x, xy.y);
  ctx.rotate(m.headingRad);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(9, 0);
  ctx.lineTo(-6, 5.5);
  ctx.lineTo(-3, 0);
  ctx.lineTo(-6, -5.5);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// ---------- Telemetria ----------
const STATE_LABELS = {
  idle: "parado",
  searching: "varrendo área",
  descending: "descendo para inspecionar",
  inspecting: "inspecionando candidato",
  climbing: "subindo para retomar",
  alert: "alerta enviado",
  depleted: "bateria esgotada",
  swept: "varredura concluída",
};

let lastTelemetryAt = 0;
function updateTelemetry(force = false) {
  const now = performance.now();
  if (!force && now - lastTelemetryAt < 180) return;
  lastTelemetryAt = now;

  const m = state.mission;
  els.droneState.textContent = STATE_LABELS[m.droneState];

  if (m.dronePos) {
    els.dronePos.textContent = `${m.dronePos.lat.toFixed(5)}, ${m.dronePos.lng.toFixed(5)}`;
  }

  const mm = String(Math.floor(m.elapsedMs / 60000)).padStart(2, "0");
  const ss = String(Math.floor((m.elapsedMs % 60000) / 1000)).padStart(2, "0");
  els.missionTime.textContent = `${mm}:${ss}`;

  const ground = currentGroundElev();
  els.droneGround.textContent = m.hasTerrain && ground != null ? `${Math.round(ground)} m` : "N/D";
  els.droneAgl.textContent = `${Math.round(m.aglM)} m`;

  const px = gesturePixels(m.aglM);
  els.droneGesturePx.textContent = `${px.span.toFixed(0)} px`;
  els.droneGesturePx.classList.toggle("insufficient", !gestureResolvable(m.aglM));

  els.droneBattery.textContent = `${Math.round(batteryLevel() * 100)}%`;
  els.droneCoverage.textContent = `${Math.round(coveragePercent())}%`;
  updatePipelineState();
}

function setActiveFlow(stage) {
  els.flowStrip.querySelectorAll(".flow-step").forEach((el) => {
    el.classList.toggle("active", el.dataset.stage === stage);
  });
}

function syncFlowToMission() {
  const s = state.mission.droneState;
  if (s === "searching") setActiveFlow("atuador");
  else if (s === "descending" || s === "inspecting" || s === "climbing") setActiveFlow("processamento");
  else if (s === "alert") setActiveFlow("feedback");
}

// ---------- Laço principal ----------
let lastFrameTime = performance.now();
function animate(now) {
  const dt = Math.min(now - lastFrameTime, 100); // evita salto ao voltar de aba inativa
  lastFrameTime = now;

  const m = state.mission;
  if (m.running) {
    // A aceleração vale só para o voo. Durante a inspeção há uma pessoa fazendo o gesto,
    // então o tempo volta a correr 1x para que a janela de confirmação seja realista.
    const speed = m.droneState === "inspecting" ? 1 : state.simSpeed;
    const simDt = dt * speed;
    m.elapsedMs += simDt;
    const flying = ["searching", "descending", "inspecting", "climbing"].includes(m.droneState);
    if (flying) {
      m.flightMs += simDt;
      if (m.droneState === "searching") {
        stepAltitude(simDt);
        stepDrone(simDt);
      } else if (m.droneState === "descending") {
        if (stepAltitude(simDt)) beginInspection();
      } else if (m.droneState === "climbing") {
        if (stepAltitude(simDt)) resumeSweep();
      } else if (m.droneState === "inspecting") {
        checkInspectTimeout();
      }
      if (batteryLevel() <= 0) endMissionByBattery();
    }
  }

  drawMap();
  updateTelemetry();
  requestAnimationFrame(animate);
}

// ---------- Controles de missão ----------
function startMission() {
  const m = state.mission;
  if (m.droneState === "depleted" || m.droneState === "swept") {
    m.flightMs = 0;
    m.elapsedMs = 0;
    m.aglM = DRONE.searchAglM;
    resetRoute();
  }
  if (!m.path.length) resetRoute();

  m.running = true;
  // parado no chão ou após alerta, a missão recomeça subindo para a altura de varredura
  m.droneState = m.aglM < DRONE.searchAglM - 1 ? "climbing" : "searching";
  state.rearmUntil = performance.now() + 1200;
  syncButtons();
  syncFlowToMission();
  updateTelemetry(true);
}

function pauseMission() {
  state.mission.running = false;
  syncButtons();
  updatePipelineState();
}

// Depois do alerta, o drone sobe de volta à altura de varredura antes de continuar.
function resumeSearch() {
  state.mission.running = true;
  beginClimb();
  logDetection("info", `subindo de ${Math.round(state.mission.aglM)} m para ${DRONE.searchAglM} m`);
}

function endMissionByBattery() {
  const m = state.mission;
  m.running = false;
  m.droneState = "depleted";
  logDetection("info", "autonomia esgotada, missão encerrada");
  syncButtons();
  updateTelemetry(true);
}

function syncButtons() {
  const m = state.mission;
  const holding = m.droneState === "alert";
  const paused = !m.running && ["searching", "descending", "inspecting", "climbing"].includes(m.droneState);

  els.btnStartMission.disabled = m.running || holding;
  els.btnStartMission.textContent = paused ? "Continuar missão" : "Iniciar missão";
  els.btnPauseMission.disabled = !m.running;
  els.btnResumeSearch.disabled = !holding;
  els.btnSimulateCandidate.disabled = m.droneState !== "searching";
  els.btnSimulateGesture.disabled = m.droneState !== "inspecting";
  els.btnExportReport.disabled = state.alerts.length === 0 && m.elapsedMs === 0;
}

// ---------- Detecção ----------
// Qual estágio está ativo agora: na varredura roda o detector de pessoa, na inspeção roda
// o classificador de gesto. Em nenhum outro estado há inferência com efeito.
function activeStage() {
  const s = state.mission.droneState;
  if (s === "searching") return "deteccao";
  if (s === "inspecting") return "gesto";
  return null;
}

function detectionArmed() {
  return activeStage() !== null && performance.now() >= state.rearmUntil;
}

function logDetection(kind, message, prob) {
  state.detections.unshift({
    time: new Date(),
    kind,
    message,
    prob: prob == null ? null : Math.round(prob * 100),
  });
  if (state.detections.length > DETECTION.logLimit) state.detections.pop();
  renderDetectionLog();
}

function renderDetectionLog() {
  if (!state.detections.length) {
    els.detectionLog.innerHTML = '<li class="empty-state">Sem eventos de detecção.</li>';
    return;
  }
  els.detectionLog.innerHTML = "";
  state.detections.forEach((d) => {
    const li = document.createElement("li");
    li.className = `det-item det-${d.kind}`;

    const time = document.createElement("span");
    time.className = "det-time";
    time.textContent = d.time.toLocaleTimeString("pt-BR");

    const msg = document.createElement("span");
    msg.className = "det-msg";
    msg.textContent = d.message;

    li.append(time, msg);
    if (d.prob != null) {
      const prob = document.createElement("span");
      prob.className = "det-prob";
      prob.textContent = `${d.prob}%`;
      li.append(prob);
    }
    els.detectionLog.appendChild(li);
  });
}

function setSustainProgress(fraction) {
  const ring = els.sustainRing;
  const circumference = 2 * Math.PI * 26;
  ring.style.strokeDasharray = `${circumference}`;
  ring.style.strokeDashoffset = `${circumference * (1 - fraction)}`;
  els.sustainRing.parentElement.classList.toggle("active", fraction > 0);
}

function updatePipelineState() {
  const s = state.mission.droneState;
  const stage = activeStage();
  renderPredictionRows();
  let label;

  if (s === "alert") label = "alerta enviado";
  else if (s === "descending") label = "descendo para inspecionar";
  else if (s === "climbing") label = "subindo, inferência suspensa";
  else if (!stage) label = "inferência inativa (drone parado)";
  else if (!state.models[stage].model) {
    label = stage === "deteccao"
      ? "estágio 1 sem modelo de detecção"
      : "estágio 2 sem modelo de gesto";
  } else if (!els.webcam.videoWidth) label = "aguardando câmera";
  else if (!detectionArmed()) label = "detecção rearmando";
  else if (state.sustainStartedAt !== null) {
    label = stage === "deteccao" ? "confirmando pessoa" : "confirmando gesto";
  } else {
    label = stage === "deteccao" ? "procurando pessoa" : "classificando gesto";
  }
  els.pipelineState.textContent = label;
  renderInspectWindow();
}

function checkSustainedGesture(rawProb) {
  state.smoothedProb = DETECTION.emaAlpha * rawProb + (1 - DETECTION.emaAlpha) * state.smoothedProb;
  const prob = state.smoothedProb;
  const now = performance.now();

  if (!detectionArmed()) {
    if (state.sustainStartedAt !== null) {
      state.sustainStartedAt = null;
      setSustainProgress(0);
    }
    updatePipelineState();
    return;
  }

  const stage = activeStage();
  const alvo = stage === "deteccao" ? "pessoa" : "gesto";

  if (prob >= state.threshold) {
    if (state.sustainStartedAt === null) {
      state.sustainStartedAt = now;
      state.sustainPeak = prob;
      logDetection("start", `${alvo} detectado, iniciando confirmação`, prob);
    }
    state.sustainPeak = Math.max(state.sustainPeak, prob);
    const progress = Math.min((now - state.sustainStartedAt) / state.sustainMs, 1);
    setSustainProgress(progress);
    if (progress >= 1) {
      const peak = state.sustainPeak;
      state.sustainStartedAt = null;
      setSustainProgress(0);
      if (stage === "deteccao") beginDescent(peak);
      else triggerAlert(peak);
    }
  } else {
    if (state.sustainStartedAt !== null) {
      const held = ((now - state.sustainStartedAt) / 1000).toFixed(1);
      logDetection("reject", `confirmação de ${alvo} interrompida após ${held}s`, prob);
      state.sustainStartedAt = null;
      setSustainProgress(0);
    }
  }
  updatePipelineState();
}

// ---------- Alerta ----------
function captureSnapshot() {
  if (!els.webcam.videoWidth) return null;
  const sc = els.snapshotCanvas;
  sc.width = 320;
  sc.height = 240;
  const ctx = sc.getContext("2d");
  ctx.drawImage(els.webcam, 0, 0, sc.width, sc.height);
  if (state.mission.droneState === "inspecting") {
    // a janela de inspeção em destaque: é este recorte que o classificador recebeu
    const r = inspectCropRect();
    const kx = sc.width / els.webcam.videoWidth;
    const ky = sc.height / els.webcam.videoHeight;
    ctx.strokeStyle = "#FF6A1A";
    ctx.lineWidth = 2;
    ctx.strokeRect(r.x * kx, r.y * ky, r.side * kx, r.side * ky);
  }
  return sc.toDataURL("image/jpeg", 0.72);
}

function triggerAlert(confidence) {
  const m = state.mission;
  if (m.droneState !== "inspecting") {
    logDetection("info", "gesto ignorado: o drone só classifica gesto durante a inspeção");
    return;
  }

  m.droneState = "alert";
  m.running = false;
  m.inspectStartedAt = null;
  state.sustainStartedAt = null;
  setSustainProgress(0);

  const alertEntry = {
    id: Date.now(),
    lat: m.dronePos.lat,
    lng: m.dronePos.lng,
    groundElev: Math.round(groundElevationAt(m.dronePos.lat, m.dronePos.lng)),
    agl: Math.round(m.aglM),
    gesturePx: Math.round(gesturePixels(m.aglM).span),
    battery: Math.round(batteryLevel() * 100),
    coverage: Math.round(coveragePercent()),
    confidence: confidence == null ? null : Math.round(confidence * 100),
    time: new Date(),
    snapshot: captureSnapshot(),
    resolved: false,
  };
  state.alerts.push(alertEntry);
  logDetection("alert", "gesto confirmado na inspeção, alerta enviado à base", confidence);
  renderAlerts();

  syncButtons();
  syncFlowToMission();
  updateTelemetry(true);
}

function renderAlerts() {
  if (!state.alerts.length) {
    els.alertLog.innerHTML = '<li class="empty-state">Nenhum alerta recebido ainda.</li>';
    syncButtons();
    return;
  }
  els.alertLog.innerHTML = "";

  [...state.alerts].reverse().forEach((a) => {
    const li = document.createElement("li");
    li.className = "alert-item" + (a.resolved ? " resolved" : "");

    const thumb = document.createElement(a.snapshot ? "img" : "div");
    thumb.className = "alert-thumb";
    if (a.snapshot) {
      thumb.src = a.snapshot;
      thumb.alt = "Quadro capturado no momento da detecção";
      thumb.addEventListener("click", () => openSnapshot(a));
    } else {
      thumb.classList.add("no-image");
      thumb.textContent = "sem imagem";
    }

    const info = document.createElement("div");
    info.className = "alert-info";

    const time = document.createElement("div");
    time.className = "alert-time";
    time.textContent = a.time.toLocaleTimeString("pt-BR");

    const coords = document.createElement("div");
    coords.className = "alert-coords";
    coords.textContent = `${a.lat.toFixed(5)}, ${a.lng.toFixed(5)}`;

    const meta = document.createElement("div");
    meta.className = "alert-meta";
    const bits = [`solo ${a.groundElev} m`];
    if (a.agl != null) bits.push(`AGL ${a.agl} m`);
    if (a.gesturePx != null) bits.push(`gesto ${a.gesturePx} px`);
    if (a.confidence != null) bits.push(`conf. ${a.confidence}%`);
    meta.textContent = bits.join(" · ");

    const status = document.createElement("div");
    status.className = "alert-state";
    const dot = document.createElement("span");
    dot.className = "alert-dot" + (a.resolved ? " resolved" : "");
    status.append(dot, document.createTextNode(
      a.resolved ? "Resgate confirmado" : "Pessoa localizada, aguardando equipe"
    ));

    info.append(time, coords, meta, status);

    if (!a.resolved) {
      const btn = document.createElement("button");
      btn.textContent = "Confirmar resgate";
      btn.addEventListener("click", () => {
        a.resolved = true;
        renderAlerts();
        drawMap();
      });
      info.append(btn);
    }

    li.append(thumb, info);
    els.alertLog.appendChild(li);
  });

  syncButtons();
}

function openSnapshot(a) {
  els.shotModalImg.src = a.snapshot;
  els.shotModalMeta.textContent =
    `${a.time.toLocaleTimeString("pt-BR")} · ${a.lat.toFixed(5)}, ${a.lng.toFixed(5)}` +
    (a.confidence != null ? ` · confiança ${a.confidence}%` : "");
  els.shotModal.hidden = false;
}

// ---------- Janela de inspeção ----------
// Fração da largura do quadro que cobre inspectFootprintM metros na altura informada.
function inspectFracFor(aglM) {
  const frac = DRONE.inspectFootprintM / swathWidthM(aglM);
  return Math.min(1, Math.max(0.05, frac));
}

// Retângulo do recorte em pixels do vídeo (resolução nativa, não a exibida).
function inspectCropRect() {
  const vw = els.webcam.videoWidth;
  const vh = els.webcam.videoHeight;
  const w = state.inspectWindow;
  const side = Math.min(vh, Math.round(vw * w.frac));
  const x = Math.round(Math.min(Math.max(w.cx * vw - side / 2, 0), vw - side));
  const y = Math.round(Math.min(Math.max(w.cy * vh - side / 2, 0), vh - side));
  return { x, y, side };
}

// Onde o vídeo aparece dentro da caixa (object-fit: contain deixa barras quando as
// proporções diferem), para posicionar o quadro da janela e converter cliques.
function videoDisplayRect() {
  const box = els.webcam.getBoundingClientRect();
  const vw = els.webcam.videoWidth || 4;
  const vh = els.webcam.videoHeight || 3;
  const scale = Math.min(box.width / vw, box.height / vh);
  const w = vw * scale;
  const h = vh * scale;
  return { left: (box.width - w) / 2, top: (box.height - h) / 2, width: w, height: h };
}

function renderInspectWindow() {
  const show = activeStage() === "gesto" && els.webcam.videoWidth > 0;
  els.inspectWindow.hidden = !show;
  els.inspectRow.hidden = !show;
  if (!show) return;
  const r = inspectCropRect();
  const d = videoDisplayRect();
  const k = d.width / els.webcam.videoWidth;
  els.inspectWindow.style.left = `${d.left + r.x * k}px`;
  els.inspectWindow.style.top = `${d.top + r.y * k}px`;
  els.inspectWindow.style.width = `${r.side * k}px`;
  els.inspectWindow.style.height = `${r.side * k}px`;
  const metros = state.inspectWindow.frac * swathWidthM(state.mission.aglM);
  els.inspectInfo.textContent = `${metros.toFixed(1)} m no solo a ${Math.round(state.mission.aglM)} m`;
}

// Copia o recorte para o canvas de inferência, que é o que o classificador de gesto recebe.
function drawInspectCrop() {
  const r = inspectCropRect();
  const c = els.inferCanvas;
  c.width = 224;
  c.height = 224;
  c.getContext("2d").drawImage(els.webcam, r.x, r.y, r.side, r.side, 0, 0, 224, 224);
  return c;
}

function setInspectCenterFromPointer(ev) {
  const box = els.webcam.getBoundingClientRect();
  const d = videoDisplayRect();
  const fx = (ev.clientX - box.left - d.left) / d.width;
  const fy = (ev.clientY - box.top - d.top) / d.height;
  state.inspectWindow.cx = Math.min(1, Math.max(0, fx));
  state.inspectWindow.cy = Math.min(1, Math.max(0, fy));
  renderInspectWindow();
}

// ---------- Câmera e modelo ----------
// A fonte pode ser a webcam interna ou a imagem do drone entrando por um capturador
// HDMI, que o navegador enxerga como mais um dispositivo de vídeo.
async function startStream(deviceId) {
  stopSources();
  const video = deviceId
    ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
    : { width: { ideal: 640 }, height: { ideal: 480 } };
  state.webcamStream = await navigator.mediaDevices.getUserMedia({ video });
  els.webcam.srcObject = state.webcamStream;
}

async function listCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter((d) => d.kind === "videoinput");
  if (cams.length < 1) return;

  const atual = state.webcamStream && state.webcamStream.getVideoTracks()[0];
  const atualId = atual && atual.getSettings ? atual.getSettings().deviceId : null;

  els.cameraSource.innerHTML = "";
  cams.forEach((c, i) => {
    const opt = document.createElement("option");
    opt.value = c.deviceId;
    // o rótulo só vem preenchido depois que a permissão é concedida
    opt.textContent = c.label || `Dispositivo de vídeo ${i + 1}`;
    els.cameraSource.appendChild(opt);
  });
  if (atualId) els.cameraSource.value = atualId;
  els.cameraSourceRow.hidden = cams.length < 2;
}

async function enableCamera() {
  try {
    await startStream(null);
    await listCameras();
    els.videoOverlay.hidden = true;
    els.btnCamera.textContent = "Câmera ligada";
    els.btnCamera.disabled = true;
    setActiveFlow("sensor");
    startPredictionLoop();
    updatePipelineState();
  } catch (e) {
    els.videoOverlay.textContent = "Câmera indisponível (permissão negada ou sem dispositivo)";
  }
}

// Recebe a imagem de bordo por WebRTC, no padrão WHEP: envia uma oferta SDP ao servidor
// local que está recebendo o RTMP do DJI Fly e recebe a resposta. O elemento de vídeo passa
// a exibir a transmissão, e a inferência segue igual, porque o modelo classifica o elemento.
async function connectStream(url) {
  els.streamStatus.textContent = "Conectando...";
  stopSources();

  const pc = new RTCPeerConnection({ iceServers: [] });
  state.streamPc = pc;
  pc.addTransceiver("video", { direction: "recvonly" });

  pc.addEventListener("track", (ev) => {
    els.webcam.srcObject = ev.streams[0];
    els.videoOverlay.hidden = true;
  });

  pc.addEventListener("connectionstatechange", () => {
    if (pc !== state.streamPc) return;
    if (pc.connectionState === "connected") {
      els.streamStatus.textContent = "Transmissão do drone conectada.";
      logDetection("info", "fonte de vídeo: transmissão do drone");
    } else if (pc.connectionState === "failed" || pc.connectionState === "disconnected") {
      els.streamStatus.textContent = "Conexão perdida.";
    }
  });

  await pc.setLocalDescription(await pc.createOffer());
  // sem trickle: espera reunir os candidatos antes de enviar a oferta
  await new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") return resolve();
    const t = setTimeout(resolve, 2500);
    pc.addEventListener("icegatheringstatechange", () => {
      if (pc.iceGatheringState === "complete") { clearTimeout(t); resolve(); }
    });
  });

  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/sdp" },
    body: pc.localDescription.sdp,
  });
  if (!resp.ok) throw new Error(`servidor respondeu ${resp.status}`);
  await pc.setRemoteDescription({ type: "answer", sdp: await resp.text() });

  els.btnCamera.disabled = true;
  setActiveFlow("sensor");
  startPredictionLoop();
  updatePipelineState();
}

// Um voo gravado entra pelo mesmo elemento de vídeo: a inferência não distingue a origem.
function playVideoFile(file) {
  stopSources();
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.fileUrl = URL.createObjectURL(file);
  els.webcam.srcObject = null;
  els.webcam.src = state.fileUrl;
  els.webcam.loop = true;
  els.webcam.play();
  els.videoOverlay.hidden = true;
  els.btnCamera.disabled = true;
  els.streamStatus.textContent = `Reproduzindo ${file.name}.`;
  logDetection("info", `fonte de vídeo: arquivo ${file.name}`);
  setActiveFlow("sensor");
  startPredictionLoop();
  updatePipelineState();
}

function stopSources() {
  if (els.webcam.src && !els.webcam.srcObject) {
    els.webcam.pause();
    els.webcam.removeAttribute("src");
    els.webcam.load();
  }
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach((t) => t.stop());
    state.webcamStream = null;
  }
  if (state.streamPc) {
    state.streamPc.close();
    state.streamPc = null;
  }
}

async function switchCamera(deviceId) {
  try {
    await startStream(deviceId);
    const nome = els.cameraSource.selectedOptions[0];
    logDetection("info", `fonte de vídeo alterada para ${nome ? nome.textContent : "outro dispositivo"}`);
  } catch (e) {
    logDetection("reject", "não foi possível abrir essa fonte de vídeo");
  }
}

const STAGE_HINT = {
  deteccao: /pessoa|person|human|alvo/i,
  gesto: /socorro|ajuda|help|sos|emerg/i,
};

async function loadModel(stage, loader) {
  const statusEl = stage === "deteccao" ? els.statusDeteccao : els.statusGesto;
  statusEl.textContent = "Carregando modelo...";
  try {
    const model = await loader();
    const slot = state.models[stage];
    slot.model = model;
    slot.count = model.getTotalClasses();
    slot.labels = model.getClassLabels ? model.getClassLabels() : [];
    const idx = slot.labels.findIndex((l) => STAGE_HINT[stage].test(l));
    slot.targetIndex = idx >= 0 ? idx : 0;
    statusEl.textContent =
      `Carregado: ${slot.count} classes, alvo "${slot.labels[slot.targetIndex] || slot.targetIndex}".`;
    logDetection("info", `modelo de ${stage} carregado (${slot.count} classes)`);
    renderPredictionRows();
    updatePipelineState();
  } catch (e) {
    statusEl.textContent = "Falha ao carregar. Confira o link ou os 3 arquivos exportados.";
  }
}

// As barras mostram sempre o modelo do estágio ativo; fora de varredura e inspeção,
// mostram o que estiver disponível apenas como referência.
let shownStage = null;
function renderPredictionRows(force = false) {
  const stage = activeStage() || (state.models.deteccao.model ? "deteccao" : "gesto");
  if (!force && stage === shownStage) return;
  shownStage = stage;
  const slot = state.models[stage];

  els.predictions.innerHTML = "";
  els.predictionsStage.textContent = stage === "deteccao"
    ? "estágio 1: detecção de pessoa"
    : "estágio 2: gesto de socorro";

  if (!slot.model) {
    const p = document.createElement("p");
    p.className = "empty-state";
    p.textContent = "Modelo deste estágio não carregado.";
    els.predictions.appendChild(p);
    return;
  }

  for (let i = 0; i < slot.count; i++) {
    const row = document.createElement("div");
    row.className = "pred-row";

    const name = document.createElement("span");
    name.className = "pred-name";
    name.textContent = slot.labels[i] || "Classe " + i;

    const track = document.createElement("span");
    track.className = "pred-bar-track";
    const fill = document.createElement("span");
    fill.className = "pred-bar-fill";
    fill.id = `predBar${i}`;
    track.appendChild(fill);

    const pct = document.createElement("span");
    pct.className = "pred-pct";
    pct.id = `predPct${i}`;
    pct.textContent = "0%";

    row.append(name, track, pct);
    els.predictions.appendChild(row);
  }
}

let predictionInterval = null;
function startPredictionLoop() {
  if (predictionInterval) clearInterval(predictionInterval);
  state.inference.windowStart = performance.now();
  state.inference.frames = 0;

  predictionInterval = setInterval(async () => {
    renderPredictionRows();
    const stage = activeStage() || shownStage;
    const slot = state.models[stage];
    if (!slot || !slot.model || !els.webcam.videoWidth) return;

    const t0 = performance.now();
    const input = stage === "gesto" ? drawInspectCrop() : els.webcam;
    const predictions = await slot.model.predict(input);
    const latency = performance.now() - t0;

    state.inference.latencyMs = state.inference.latencyMs
      ? state.inference.latencyMs * 0.8 + latency * 0.2
      : latency;
    state.inference.frames += 1;
    const windowMs = performance.now() - state.inference.windowStart;
    if (windowMs >= 1000) {
      state.inference.fps = (state.inference.frames * 1000) / windowMs;
      state.inference.frames = 0;
      state.inference.windowStart = performance.now();
      els.infFps.textContent = state.inference.fps.toFixed(1);
      els.infLatency.textContent = `${Math.round(state.inference.latencyMs)} ms`;
    }

    predictions.forEach((p, i) => {
      const bar = document.getElementById(`predBar${i}`);
      const pct = document.getElementById(`predPct${i}`);
      if (bar) {
        bar.style.width = `${Math.round(p.probability * 100)}%`;
        bar.classList.toggle("alert-class", i === slot.targetIndex);
      }
      if (pct) pct.textContent = `${Math.round(p.probability * 100)}%`;
    });

    if (!activeStage()) return;
    const raw = predictions[slot.targetIndex] ? predictions[slot.targetIndex].probability : 0;
    checkSustainedGesture(raw);
  }, 150);
}

// ---------- Relatório da missão ----------
function buildMissionReport() {
  const m = state.mission;
  const mapPng = els.mapCanvas.toDataURL("image/png");
  const now = new Date();
  const durationMin = (m.elapsedMs / 60000).toFixed(1);
  const coveragePct = Math.round(coveragePercent());
  const route = routeStats();

  const alertsHtml = state.alerts.length
    ? state.alerts.map((a, i) => `
      <div class="alert">
        ${a.snapshot ? `<img src="${a.snapshot}" alt="Quadro da detecção ${i + 1}">` : '<div class="noimg">sem imagem</div>'}
        <div>
          <strong>Alerta ${String(i + 1).padStart(2, "0")}</strong><br>
          Horário: ${a.time.toLocaleString("pt-BR")}<br>
          Coordenadas: ${a.lat.toFixed(5)}, ${a.lng.toFixed(5)}<br>
          Elevação do solo: ${a.groundElev} m${a.agl != null ? ` · altura de inspeção: ${a.agl} m` : ""}${a.gesturePx != null ? ` · gesto com ${a.gesturePx} px de envergadura na imagem` : ""}<br>
          ${a.confidence != null ? `Confiança na confirmação: ${a.confidence}%<br>` : ""}
          Bateria no momento: ${a.battery}% · Cobertura: ${a.coverage}%<br>
          Situação: ${a.resolved ? "resgate confirmado" : "aguardando equipe"}
        </div>
      </div>`).join("")
    : "<p>Nenhum alerta registrado nesta missão.</p>";

  const detectionsHtml = state.detections.length
    ? `<ul>${state.detections.map((d) =>
        `<li>${d.time.toLocaleTimeString("pt-BR")} · ${d.message}${d.prob != null ? ` (${d.prob}%)` : ""}</li>`
      ).join("")}</ul>`
    : "<p>Sem eventos de detecção registrados.</p>";

  return `<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<title>ResgatAr: relatório de missão ${now.toLocaleDateString("pt-BR")}</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 32px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.5; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .05em; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
  .sub { color: #666; font-size: 13px; margin-top: 0; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  td { padding: 5px 0; border-bottom: 1px solid #eee; }
  td:first-child { color: #666; width: 45%; }
  img.map { width: 100%; display: block; margin-top: 10px; background: #0A0A08; padding: 8px; border: 1px solid #ccc; }
  .alert { display: flex; gap: 14px; align-items: flex-start; padding: 12px 0; border-bottom: 1px solid #eee; font-size: 14px; }
  .alert img { width: 160px; border: 1px solid #ddd; }
  .noimg { width: 160px; height: 120px; background: #f2f2f2; color: #999; display: flex; align-items: center; justify-content: center; font-size: 12px; }
  ul { font-size: 13px; padding-left: 18px; }
  footer { margin-top: 32px; font-size: 12px; color: #777; border-top: 1px solid #ddd; padding-top: 10px; }
</style></head><body>
<h1>ResgatAr: relatório de missão</h1>
<p class="sub">Gerado em ${now.toLocaleString("pt-BR")}</p>

<h2>Resumo</h2>
<table>
  <tr><td>Duração da missão (tempo simulado)</td><td>${durationMin} min</td></tr>
  <tr><td>Padrão de varredura</td><td>paralela (parallel track), ${route.tracks} faixas${m.hasTerrain ? ", ordenadas por probabilidade estimada pelo relevo" : ""}</td></tr>
  <tr><td>Extensão da rota</td><td>${route.km.toFixed(1)} km, ${Math.round(route.minutes)} min de voo a ${DRONE.cruiseSpeedMps} m/s</td></tr>
  <tr><td>Saídas necessárias</td><td>${route.sorties} (autonomia ${DRONE.enduranceMin} min por bateria)</td></tr>
  <tr><td>Cobertura de área pela câmera</td><td>${coveragePct}%</td></tr>
  <tr><td>Bateria restante</td><td>${Math.round(batteryLevel() * 100)}%</td></tr>
  <tr><td>Cercamento eletrônico</td><td>NO ${m.geofence.nw.lat.toFixed(4)}, ${m.geofence.nw.lng.toFixed(4)} / SE ${m.geofence.se.lat.toFixed(4)}, ${m.geofence.se.lng.toFixed(4)}</td></tr>
  <tr><td>Voo</td><td>seguimento de terreno a ${DRONE.searchAglM} m do solo na varredura (teto legal ${DRONE.legalCeilingM} m)</td></tr>
  <tr><td>Faixa da câmera</td><td>${Math.round(swathWidthM(DRONE.searchAglM))} m no solo, espaçamento ${Math.round(trackSpacingM())} m (sobreposição ${Math.round(DRONE.sidelap * 100)}%)</td></tr>
  <tr><td>Inspeção do candidato</td><td>descida para ${DRONE.inspectAglM} m, onde os braços abertos passam de ${gesturePixels(DRONE.searchAglM).span.toFixed(0)} px para ${gesturePixels(DRONE.inspectAglM).span.toFixed(0)} px</td></tr>
  <tr><td>Velocidade de cruzeiro</td><td>${DRONE.cruiseSpeedMps} m/s</td></tr>
  <tr><td>Limiar de confiança</td><td>${Math.round(state.threshold * 100)}% por ${(state.sustainMs / 1000).toFixed(1)}s contínuos</td></tr>
  <tr><td>Alertas emitidos</td><td>${state.alerts.length}</td></tr>
</table>

<h2>Mapa da missão</h2>
<img class="map" src="${mapPng}" alt="Mapa da área de busca com rota e alertas">

<h2>Alertas</h2>
${alertsHtml}

<h2>Eventos de detecção</h2>
${detectionsHtml}

<footer>
Relevo: dados de elevação SRTM obtidos via Open-Elevation para a área de busca
(${MISSION_TERRAIN.elevMin} m a ${MISSION_TERRAIN.elevMax} m).
Reconhecimento de gesto: modelo de imagem do Teachable Machine executado localmente em TensorFlow.js.
</footer>
</body></html>`;
}

function exportMissionReport() {
  const blob = new Blob([buildMissionReport()], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
  const a = document.createElement("a");
  a.href = url;
  a.download = `missao-resgatar-${stamp}.html`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ---------- Eventos ----------
els.btnCamera.addEventListener("click", enableCamera);

els.cameraSource.addEventListener("change", () => switchCamera(els.cameraSource.value));

els.btnStream.addEventListener("click", async () => {
  try {
    await connectStream(els.streamUrl.value.trim());
  } catch (e) {
    els.streamStatus.textContent = `Falha ao conectar: ${e.message}. Confira se o servidor local está no ar e recebendo o RTMP.`;
  }
});

els.fileVideo.addEventListener("change", () => {
  const f = els.fileVideo.files[0];
  if (f) playVideoFile(f);
});

// a janela de inspeção segue o dedo ou o mouse sobre o vídeo
els.webcam.parentElement.addEventListener("pointerdown", (ev) => {
  if (activeStage() !== "gesto") return;
  ev.preventDefault();
  setInspectCenterFromPointer(ev);
  const move = (e) => setInspectCenterFromPointer(e);
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
});
els.inspectFrac.addEventListener("input", () => {
  state.inspectWindow.frac = Number(els.inspectFrac.value) / 100;
  renderInspectWindow();
});
els.webcam.addEventListener("loadedmetadata", renderInspectWindow);
window.addEventListener("resize", renderInspectWindow);

// um capturador pode ser conectado depois que a câmera já está ligada
navigator.mediaDevices.addEventListener("devicechange", () => {
  if (state.webcamStream) listCameras();
});

els.btnSimulateGesture.addEventListener("click", () => {
  if (state.mission.droneState !== "inspecting") {
    logDetection("info", "o gesto só é classificado durante a inspeção, após a descida");
    return;
  }
  triggerAlert(state.smoothedProb || null);
});

function bindModelLoaders(stage, urlEl, btnUrl, filesEl, btnFiles, statusEl) {
  btnUrl.addEventListener("click", () => {
    const v = urlEl.value.trim();
    if (!v) return;
    const root = v.endsWith("/") ? v : v + "/";
    loadModel(stage, () => tmImage.load(root + "model.json", root + "metadata.json"));
  });

  btnFiles.addEventListener("click", () => {
    const files = [...filesEl.files];
    const pick = (re) => files.find((f) => re.test(f.name));
    const model = pick(/^model\.json$/i);
    const weights = pick(/weights.*\.bin$/i);
    const meta = pick(/^metadata\.json$/i);
    if (model && weights && meta) {
      loadModel(stage, () => tmImage.loadFromFiles(model, weights, meta));
    } else {
      statusEl.textContent = "Selecione os 3 arquivos juntos: model.json, weights.bin e metadata.json.";
    }
  });
}

bindModelLoaders("deteccao", els.urlDeteccao, els.btnUrlDeteccao, els.filesDeteccao, els.btnFilesDeteccao, els.statusDeteccao);
bindModelLoaders("gesto", els.urlGesto, els.btnUrlGesto, els.filesGesto, els.btnFilesGesto, els.statusGesto);

els.btnSimulateCandidate.addEventListener("click", () => {
  if (state.mission.droneState !== "searching") return;
  beginDescent(null);
});

els.threshold.addEventListener("input", () => {
  state.threshold = Number(els.threshold.value) / 100;
  els.thresholdValue.textContent = `${els.threshold.value}%`;
});

els.sustain.addEventListener("input", () => {
  state.sustainMs = Number(els.sustain.value) * 100;
  els.sustainValue.textContent = `${(state.sustainMs / 1000).toFixed(1)}s`;
});

els.simSpeed.addEventListener("change", () => {
  state.simSpeed = Number(els.simSpeed.value);
});

els.btnApplyGeofence.addEventListener("click", applyGeofence);
els.btnStartMission.addEventListener("click", startMission);
els.btnPauseMission.addEventListener("click", pauseMission);
els.btnResumeSearch.addEventListener("click", resumeSearch);
els.btnExportReport.addEventListener("click", exportMissionReport);

els.shotModalClose.addEventListener("click", () => { els.shotModal.hidden = true; });
els.shotModal.addEventListener("click", (e) => {
  if (e.target === els.shotModal) els.shotModal.hidden = true;
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") els.shotModal.hidden = true;
});

// O navegador congela requestAnimationFrame em abas ocultas, então a simulação de voo
// para junto. Sem isso registrado, o operador volta e vê o relógio parado sem explicação.
document.addEventListener("visibilitychange", () => {
  if (state.mission.droneState !== "searching" || !state.mission.running) return;
  if (document.visibilityState === "hidden") {
    logDetection("info", "aba em segundo plano: simulação de voo suspensa");
  } else {
    lastFrameTime = performance.now();
    logDetection("info", "aba em primeiro plano: simulação de voo retomada");
  }
});

// ---------- Init ----------
resetRoute();
setSustainProgress(0);
renderDetectionLog();
renderAlerts();
updatePipelineState();
updateTelemetry(true);
syncButtons();
requestAnimationFrame(animate);

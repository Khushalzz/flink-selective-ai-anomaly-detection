const previewSamples = {
  "1h": [280, 345, 310, 420, 390, 475, 430, 520, 465, 555, 510, 610],
  "6h": [330, 370, 345, 410, 395, 460, 435, 520, 480, 560, 525, 605],
  "24h": [290, 350, 320, 430, 385, 470, 440, 535, 490, 570, 530, 625],
};
const previewEvents = [
  { event_id: "EV-2418", moteid: 27, anomaly_reasons: "Voltage dip", timestamp: "2004-03-01 02:18:00", is_anomaly: 1, escalated: 1, engine: "xgboost", inference_status: "OK" },
  { event_id: "EV-2417", moteid: 12, anomaly_reasons: "Temperature drift", timestamp: "2004-03-01 02:14:00", is_anomaly: 0, escalated: 1, engine: "xgboost", inference_status: "OK" },
  { event_id: "EV-2416", moteid: 41, anomaly_reasons: "Sensor flatline", timestamp: "2004-03-01 02:09:00", is_anomaly: 1, escalated: 0, engine: "fast-path", inference_status: "OK" },
  { event_id: "EV-2415", moteid: 8, anomaly_reasons: "Light spike", timestamp: "2004-03-01 02:02:00", is_anomaly: 0, escalated: 1, engine: "xgboost", inference_status: "OK" },
  { event_id: "EV-2414", moteid: 33, anomaly_reasons: "Humidity shift", timestamp: "2004-03-01 01:56:00", is_anomaly: 0, escalated: 0, engine: "fast-path", inference_status: "OK" },
];

let mode = "live";
let selectedRange = "24h";
let selectedFilter = "all";
let selectedSystem = "bdt";
let currentEvents = [];
let pollHandle = null;

function byId(id) { return document.getElementById(id); }
function setText(id, value) { const node = byId(id); if (node) node.textContent = value; }
function pct(value) { return `${(Number(value || 0) * 100).toFixed(1)}%`; }
function safeNumber(value, digits = 0) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—"; }
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
}

function setConnection(state, label) {
  const pill = byId("connection-pill");
  if (!pill) return;
  pill.classList.remove("online", "offline", "preview");
  pill.classList.add(state);
  setText("connection-label", label);
  setText("sidebar-mode-title", state === "online" ? "Live stream" : state === "preview" ? "Preview mode" : "Disconnected");
  setText("sidebar-mode-caption", state === "online" ? "ClickHouse + Redis" : state === "preview" ? "Sample data" : "Waiting for services");
}

function renderChart(values, label = "LIVE RUN") {
  const svg = byId("volume-chart");
  if (!svg) return;
  if (!values || !values.length) {
    svg.innerHTML = '<text x="380" y="125" text-anchor="middle" fill="#94a0ae" font-size="13">Waiting for stream data…</text>';
    setText("chart-series-label", label);
    return;
  }
  const width = 760, height = 250, left = 43, right = 10, top = 10, bottom = 23;
  const innerWidth = width - left - right, innerHeight = height - top - bottom;
  const maxValue = Math.max(10, Math.ceil(Math.max(...values) / 100) * 100);
  const points = values.map((value, index) => ({
    x: left + (index / Math.max(1, values.length - 1)) * innerWidth,
    y: top + innerHeight - (Number(value) / maxValue) * innerHeight,
  }));
  const line = points.map((point, index) => `${index ? "L" : "M"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
  const area = `${line} L${points.at(-1).x.toFixed(1)},${(top + innerHeight).toFixed(1)} L${left},${(top + innerHeight).toFixed(1)} Z`;
  const grid = [0, 1, 2, 3].map((row) => {
    const y = top + row * (innerHeight / 3);
    const tick = Math.round(maxValue * (1 - row / 3));
    return `<line class="grid-line" x1="${left}" y1="${y}" x2="${width - right}" y2="${y}"/><text class="axis-label" x="2" y="${y + 3}">${tick}</text>`;
  }).join("");
  const dots = points.filter((_, index) => index % Math.max(1, Math.floor(points.length / 6)) === 0)
    .map((point) => `<circle class="chart-point" cx="${point.x}" cy="${point.y}" r="3.5"/>`).join("");
  svg.innerHTML = `<defs><linearGradient id="areaGradient" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#5e8ce7" stop-opacity=".22"/><stop offset="100%" stop-color="#5e8ce7" stop-opacity=".015"/></linearGradient></defs>${grid}<path class="area-fill" d="${area}"/><path class="chart-line" d="${line}"/>${dots}`;
  setText("chart-series-label", label);
}

function setMetrics(summary, health) {
  setText("metric-total", safeNumber(summary.total));
  setText("metric-throughput", safeNumber(summary.throughput_eps));
  setText("metric-f1", summary.f1 == null ? "—" : safeNumber(summary.f1, 3));
  const anomalyRate = summary.total ? 100 * summary.anomalies / summary.total : 0;
  setText("metric-anomaly-rate", summary.total ? safeNumber(anomalyRate, 2) : "0.00");
  setText("metric-truth-count", summary.ground_truth_count ? `${safeNumber(summary.ground_truth_count)} labeled windows` : `${safeNumber(summary.anomalies)} predicted`);
  setText("metric-truth-tag", summary.ground_truth_count ? "LABELED RUN" : "NO LABELS");
  setText("metric-escalation-tag", "FLINK OUTPUT");
  setText("metric-escalated-count", `${safeNumber(summary.escalated)} escalated · p95 ${safeNumber(summary.p95_ms, 1)} ms`);
  setText("filter-all-count", safeNumber(summary.total));
  setText("filter-high-count", safeNumber(summary.anomalies));
  setText("filter-review-count", safeNumber(summary.escalated));
  setText("metric-source", summary.run_id ? `Run ${summary.run_id.slice(0, 8)} · ${summary.status}` : "Waiting for producer");
  setText("metric-model-caption", selectedSystem === "laya" ? "Laya · async comparison" : selectedSystem === "fast" ? "IF + AE · no fallback" : "IF + AE · XGBoost fallback");
  setText("route-fallback-label", selectedSystem === "laya" ? "Laya comparison" : selectedSystem === "fast" ? "No fallback" : "XGBoost fallback");
  setText("routing-engine", selectedSystem === "laya" ? "Laya sidecar" : selectedSystem === "fast" ? "Fast path only" : "XGBoost · Flink JVM");
  setText("routing-subtitle", selectedSystem === "laya" ? "Optional Async I/O comparison branch" : selectedSystem === "fast" ? "Baseline · no fallback" : "BDT · JVM fallback route");
  setText("routing-eyebrow", "LIVE ROUTING");
  setText("events-eyebrow", "LIVE RUN DATA");
  setText("routing-rate", pct(summary.escalation_rate));
  setText("route-fast-percent", pct(1 - Number(summary.escalation_rate || 0)));
  setText("route-fallback-percent", pct(summary.escalation_rate));
  const rate = Math.max(0, Math.min(100, Number(summary.escalation_rate || 0) * 100));
  byId("routing-donut").style.background = `conic-gradient(#7566d7 0 ${rate}%, #edf0f6 ${rate}% 100%)`;
  setText("event-count", safeNumber(summary.anomalies));
  if (health && health.status === "ok") {
    setConnection("online", summary.run_id ? "LIVE DATA" : "READY · IDLE");
  } else {
    setConnection("offline", "DEGRADED");
  }
  setText("mode-title", summary.run_id ? `Live run · ${summary.status}` : "No active replay");
  setText("mode-message", summary.run_id
    ? "Metrics below come from Flink outputs in ClickHouse and alerts in Redis."
    : "Start the demo replay to see actual stream events. The page is connected, but no run is active.");
  setText("banner-tag", summary.run_id ? "LIVE RUN" : "READY");
}

function eventView(event) {
  const mote = String(event.moteid ?? "?").padStart(2, "0");
  const priority = Number(event.is_anomaly) === 1 ? "high" : Number(event.escalated) === 1 ? "review" : "normal";
  const type = event.anomaly_reasons && event.anomaly_reasons !== "NORMAL" ? event.anomaly_reasons.replaceAll("_", " ") : event.prediction;
  const timestamp = event.timestamp ? new Date(String(event.timestamp).replace(" ", "T") + (String(event.timestamp).includes("Z") ? "" : "Z")) : null;
  const time = timestamp && !Number.isNaN(timestamp.getTime()) ? timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—";
  return { id: event.event_id, mote, type: type || "Event", time, priority, state: event.inference_status === "OK" ? (priority === "high" ? "Anomaly" : priority === "review" ? "Escalated" : "Normal") : "Unavailable" };
}

function renderEvents(rows) {
  const query = byId("event-search").value.trim().toLowerCase().replace(/^mote\s*#?/, "");
  const visible = rows.map(eventView).filter((event) => {
    const matchesFilter = selectedFilter === "all" || event.priority === selectedFilter;
    const matchesSearch = !query || event.mote.includes(query) || event.type.toLowerCase().includes(query) || event.id.toLowerCase().includes(query);
    return matchesFilter && matchesSearch;
  });
  const tbody = byId("event-rows");
  if (!visible.length) {
    tbody.innerHTML = '<tr><td class="empty-state" colspan="5">No events match this run and filter.</td></tr>';
    return;
  }
  tbody.innerHTML = visible.map((event) => {
    const review = event.priority === "review";
    return `<tr><td><span class="event-name"><i class="event-indicator ${review ? "review" : ""}"></i>${escapeHtml(event.id)}</span></td><td>Mote ${escapeHtml(event.mote)}</td><td>${escapeHtml(event.type)}</td><td>${escapeHtml(event.time)}</td><td><span class="event-state ${review ? "review" : ""}">${escapeHtml(event.state)}</span></td></tr>`;
  }).join("");
}

function renderAttention(rows) {
  const top = rows.map(eventView).filter((event) => event.priority !== "normal").slice(0, 3);
  byId("map-list-title").textContent = selectedSystem === "laya" ? "Laya comparison events" : "Latest alerts";
  byId("map-list-tag").textContent = mode === "preview" ? "DEMO" : "LIVE";
  byId("map-aside-foot").textContent = mode === "preview" ? "Illustrative events only." : "Latest ClickHouse rows for this run.";
  byId("map-attention-list").innerHTML = top.length ? top.map((event) => {
    const severity = event.priority === "high" ? "high" : "medium";
    const label = event.priority === "high" ? "HIGH" : event.priority === "review" ? "REVIEW" : "NORMAL";
    return `<div class="sensor-row"><span class="sensor-number">${escapeHtml(event.mote)}</span><span><strong>Mote ${escapeHtml(event.mote)}</strong><small>${escapeHtml(event.type)}</small></span><span class="severity ${severity}">${label}</span></div>`;
  }).join("") : '<div class="map-aside-foot">No alerts or escalations in this run.</div>';
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function renderLive() {
  try {
    const health = await fetchJson("/api/health");
    const system = selectedSystem;
    const [summary, events, series] = await Promise.all([
      fetchJson(`/api/summary?system=${encodeURIComponent(system)}`),
      fetchJson(`/api/events?limit=100&system=${encodeURIComponent(system)}&priority=${encodeURIComponent(selectedFilter)}`),
      fetchJson(`/api/series?window=${encodeURIComponent(selectedRange)}&system=${encodeURIComponent(system)}`),
    ]);
    setMetrics(summary, health);
    currentEvents = events.events || [];
    renderEvents(currentEvents);
    renderAttention(currentEvents);
    renderChart((series.series || []).map((row) => Number(row.readings)), "CLICKHOUSE · LIVE");
    setText("chart-eyebrow", "LIVE STREAM");
    setText("chart-subtitle", `${series.series?.length || 0} time buckets · ClickHouse output`);
    setText("footer-mode", `Run ${summary.run_id ? summary.run_id.slice(0, 8) : "idle"} · live API connected`);
  } catch (error) {
    setConnection("offline", "API OFFLINE");
    setText("mode-title", "Live data unavailable");
    setText("mode-message", "Start the Compose stack and replay script, or switch to Preview for labeled sample data.");
    setText("banner-tag", "DISCONNECTED");
    renderChart([], "NO CONNECTION");
    renderEvents([]);
    renderAttention([]);
    setText("footer-mode", "Dashboard API is unavailable");
    console.warn("BDT dashboard API unavailable:", error);
  }
}

function renderPreview() {
  setConnection("preview", "SAMPLE PREVIEW");
  setText("mode-title", "Sample benchmark snapshot");
  setText("mode-message", "Illustrative offline results only; this mode does not read the live stream.");
  setText("banner-tag", "DEMO DATA");
  setText("metric-total", "40,000");
  setText("metric-throughput", "—");
  setText("metric-f1", "0.824");
  setText("metric-anomaly-rate", "2.85");
  setText("metric-truth-count", "1,138 injected test windows");
  setText("metric-truth-tag", "SYNTHETIC TEST");
  setText("metric-escalation-tag", "OFFLINE GATE");
  setText("metric-escalated-count", "20.5% escalated");
  setText("metric-source", "Included held-out split");
  setText("metric-model-caption", "IF + AE · XGBoost fallback");
  setText("event-count", "1,138");
  setText("filter-all-count", "40,000");
  setText("filter-high-count", "1,138");
  setText("filter-review-count", "8,200");
  setText("routing-rate", "20.5%");
  setText("route-fast-percent", "79.5%");
  setText("route-fallback-percent", "20.5%");
  setText("route-fallback-label", "XGBoost fallback");
  setText("routing-engine", "XGBoost · offline");
  setText("routing-subtitle", "Historical offline benchmark · synthetic labels");
  setText("routing-eyebrow", "OFFLINE BENCHMARK");
  setText("events-eyebrow", "SYNTHETIC TEST DATA");
  byId("routing-donut").style.background = "conic-gradient(#7566d7 0 20.5%, #edf0f6 20.5% 100%)";
  currentEvents = previewEvents;
  renderEvents(currentEvents);
  renderAttention(currentEvents);
  renderChart(previewSamples[selectedRange], "SAMPLE SERIES");
  setText("chart-eyebrow", "SAMPLE REPLAY");
  setText("chart-subtitle", "Illustrative incoming readings per interval");
  setText("footer-mode", "Preview only · no live inference");
}

function refresh() {
  if (mode === "preview") {
    renderPreview();
  } else {
    renderLive();
  }
}

function setMode(nextMode) {
  mode = nextMode;
  document.querySelectorAll("[data-mode]").forEach((button) => button.classList.toggle("selected", button.dataset.mode === mode));
  refresh();
}

document.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
document.querySelectorAll("[data-range]").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll("[data-range]").forEach((item) => item.classList.remove("selected"));
  button.classList.add("selected");
  selectedRange = button.dataset.range;
  refresh();
}));
document.querySelectorAll("[data-filter]").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll("[data-filter]").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  selectedFilter = button.dataset.filter;
  refresh();
}));
byId("event-search").addEventListener("input", () => renderEvents(currentEvents));
byId("system-select").value = selectedSystem;
byId("system-select").addEventListener("change", (event) => { selectedSystem = event.target.value; refresh(); });
refresh();
pollHandle = window.setInterval(() => { if (mode === "live") refresh(); }, 2500);

const samples = {
  "1h": [280, 345, 310, 420, 390, 475, 430, 520, 465, 555, 510, 610],
  "6h": [330, 370, 345, 410, 395, 460, 435, 520, 480, 560, 525, 605],
  "24h": [290, 350, 320, 430, 385, 470, 440, 535, 490, 570, 530, 625],
};
const events = [
  { id: "EV-2418", mote: 27, type: "Voltage dip", time: "2 min ago", priority: "high", state: "Escalated" },
  { id: "EV-2417", mote: 12, type: "Temperature drift", time: "6 min ago", priority: "review", state: "Review" },
  { id: "EV-2416", mote: 41, type: "Sensor flatline", time: "11 min ago", priority: "high", state: "Escalated" },
  { id: "EV-2415", mote: 8, type: "Light spike", time: "18 min ago", priority: "review", state: "Review" },
  { id: "EV-2414", mote: 33, type: "Humidity shift", time: "24 min ago", priority: "review", state: "Review" },
];
let selectedRange = "24h";
let selectedFilter = "all";

function renderChart() {
  const svg = document.getElementById("volume-chart");
  if (!svg) return;
  const values = samples[selectedRange];
  const width = 760;
  const height = 250;
  const left = 43;
  const right = 10;
  const top = 10;
  const bottom = 23;
  const innerWidth = width - left - right;
  const innerHeight = height - top - bottom;
  const maxValue = 800;
  const points = values.map((value, index) => ({
    x: left + (index / (values.length - 1)) * innerWidth,
    y: top + innerHeight - (value / maxValue) * innerHeight,
  }));
  const line = points.map((point, index) => `${index ? "L" : "M"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
  const area = `${line} L${points.at(-1).x.toFixed(1)},${(top + innerHeight).toFixed(1)} L${left},${(top + innerHeight).toFixed(1)} Z`;
  const grid = [0, 1, 2, 3].map((row) => {
    const y = top + row * (innerHeight / 3);
    const label = `${800 - row * 250}`;
    return `<line class="grid-line" x1="${left}" y1="${y}" x2="${width - right}" y2="${y}"/><text class="axis-label" x="2" y="${y + 3}">${label}</text>`;
  }).join("");
  const dots = points.filter((_, index) => index % 2 === 1).map((point) => `<circle class="chart-point" cx="${point.x}" cy="${point.y}" r="3.5"/>`).join("");
  svg.innerHTML = `<defs><linearGradient id="areaGradient" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#5e8ce7" stop-opacity=".22"/><stop offset="100%" stop-color="#5e8ce7" stop-opacity=".015"/></linearGradient></defs>${grid}<path class="area-fill" d="${area}"/><path class="chart-line" d="${line}"/>${dots}`;
}

function renderEvents() {
  const tbody = document.getElementById("event-rows");
  const query = document.getElementById("event-search").value.trim().toLowerCase();
  const visible = events.filter((event) => {
    const matchesFilter = selectedFilter === "all" || event.priority === selectedFilter;
    const matchesSearch = !query || String(event.mote).padStart(2, "0").includes(query.replace(/^mote\s*#?/, "")) || event.type.toLowerCase().includes(query);
    return matchesFilter && matchesSearch;
  });
  if (!visible.length) {
    tbody.innerHTML = '<tr><td class="empty-state" colspan="5">No sample events match this filter.</td></tr>';
    return;
  }
  tbody.innerHTML = visible.map((event) => {
    const review = event.priority === "review";
    return `<tr><td><span class="event-name"><i class="event-indicator ${review ? "review" : ""}"></i>${event.id}</span></td><td>Mote ${String(event.mote).padStart(2, "0")}</td><td>${event.type}</td><td>${event.time}</td><td><span class="event-state ${review ? "review" : ""}">${event.state}</span></td></tr>`;
  }).join("");
}

document.querySelectorAll("[data-range]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-range]").forEach((item) => item.classList.remove("selected"));
    button.classList.add("selected");
    selectedRange = button.dataset.range;
    renderChart();
  });
});
document.querySelectorAll("[data-filter]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-filter]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    selectedFilter = button.dataset.filter;
    renderEvents();
  });
});
document.getElementById("event-search").addEventListener("input", renderEvents);
renderChart();
renderEvents();

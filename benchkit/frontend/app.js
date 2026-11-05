"use strict";
// P100 Bench Kit UI. Plain JS, no build step. All server text goes through
// esc() or textContent; the API is same-origin with a session cookie.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => (typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(d)) : "–");
const dur = (s) => { s = Math.round(s || 0); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return h ? `${h} h ${m} min` : `${m} min`; };
const when = (t) => (t ? new Date(t * 1000).toLocaleString() : "–");
const gb = (b) => (b / 1e9).toFixed(1) + " GB";

const state = { system: null, models: null, stream: null, live: null, charts: [], layout: null, gen: 0 };
// Every navigation bumps state.gen; a view that awaited past a navigation stops instead of
// writing into the page that replaced it.
class Stale extends Error {}
const fresh = (gen) => { if (gen !== state.gen) throw new Stale(); };

function toast(msg, ms = 4000) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), ms);
}

async function api(path, opts = {}) {
  const init = { credentials: "same-origin", headers: {}, ...opts };
  if (opts.json !== undefined) { init.method = init.method || "POST"; init.body = JSON.stringify(opts.json); init.headers["Content-Type"] = "application/json"; }
  const r = await fetch("/api" + path, init);
  if (r.status === 401 && path !== "/login") { showLogin(); throw new Error("login required"); }
  const ct = r.headers.get("content-type") || "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((data && data.detail) || r.statusText);
  return data;
}

// ---------------- login ----------------
function showLogin() { $("#login").classList.remove("hidden"); $("#login-token").focus(); }
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/login", { json: { token: $("#login-token").value.trim() } });
    $("#login").classList.add("hidden"); $("#login-token").value = ""; $("#login-err").textContent = "";
    await boot();
  } catch (err) { $("#login-err").textContent = err.message; }
});

// ---------------- routing ----------------
const routes = { run: viewRun, live: viewLive, history: viewHistory, detail: viewDetail, compare: viewCompare, models: viewModels, system: viewSystem };
function route() {
  const [name, arg] = (location.hash.replace(/^#\//, "") || "run").split("/");
  $$("nav a").forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#/" + name));
  destroyCharts();
  state.live = null;
  const gen = ++state.gen;
  const fn = routes[name] || viewRun;
  fn(decodeURIComponent(arg || ""), gen).catch((e) => {
    if (e instanceof Stale || gen !== state.gen || e.message === "login required") return;
    $("#view").innerHTML = `<div class="card err">${esc(e.message)}</div>`;
  });
}
window.addEventListener("hashchange", route);

async function boot() {
  state.system = await api("/system");
  updateActive(state.system.active_run);
  openStream();
  route();
}

function updateActive(runId) {
  const el = $("#active-run");
  if (runId) { el.innerHTML = `<a href="#/live/${esc(runId)}" style="color:inherit">running</a>`; el.className = "pill running"; }
  else el.className = "pill hidden";
  if (state.system) state.system.active_run = runId || null;
}

// ---------------- live event stream ----------------
function openStream() {
  if (state.stream) return;
  const es = new EventSource("/api/stream");
  state.stream = es;
  es.onmessage = (m) => {
    let ev; try { ev = JSON.parse(m.data); } catch { return; }
    if (ev.type === "status") { updateActive(ev.status === "running" ? ev.run_id : null); }
    if (state.live && state.live.id === ev.run_id) state.live.onEvent(ev);
  };
  es.onerror = () => { /* EventSource reconnects by itself */ };
}

// ---------------- charts ----------------
function destroyCharts() { state.charts.forEach((c) => c.plot && c.plot.destroy()); state.charts = []; }
const PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d"];

function metricSeries(metric, samples, gpus) {
  // -> {x: [], ys: [[]...], names: []}
  const x = samples.map((s) => s.t);
  if (metric.scope === "gpu") {
    return { x, names: gpus.map((g) => "GPU " + g), ys: gpus.map((g) => samples.map((s) => num(s.gpus && s.gpus[g] && s.gpus[g][metric.key]))) };
  }
  return { x, names: [metric.label], ys: [samples.map((s) => num(s[metric.scope] && s[metric.scope][metric.key]))] };
}
const num = (v) => (typeof v === "number" ? v : null);

// latest non-empty value of each series, shown in the chart header so no hover is needed
function latestHtml(data, unit) {
  const fmt = (v) => (Math.abs(v) >= 100 ? Math.round(v) : Math.round(v * 10) / 10).toLocaleString();
  return data.ys.map((ys, i) => {
    let v = null;
    for (let j = ys.length - 1; j >= 0 && v === null; j--) v = ys[j];
    if (v === null) return "";
    const name = data.names.length > 1 ? `<span class="muted">${esc(data.names[i].replace("GPU ", ""))}</span> ` : "";
    return `<span class="latest" style="--c:${PALETTE[i % PALETTE.length]}">${name}${fmt(v)}${unit ? " " + esc(unit) : ""}</span>`;
  }).join("");
}

const clock = (u, vals) => vals.map((v) => new Date(v * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
function makeChart(el, metric, data) {
  const w = Math.max(200, el.clientWidth || 320);
  const opts = {
    width: w, height: 180, legend: { show: data.names.length > 1 },
    scales: { x: { time: true } },
    axes: [{ stroke: "#888", grid: { stroke: "rgba(128,128,128,.15)" }, values: clock, space: 70 },
           { stroke: "#888", grid: { stroke: "rgba(128,128,128,.15)" }, size: 62 }],
    series: [{}, ...data.names.map((n, i) => ({ label: n, stroke: PALETTE[i % PALETTE.length], width: 1.5, spanGaps: true }))],
  };
  const plot = new uPlot(opts, [data.x, ...data.ys], el);
  // follow the card's width (grid columns settle after first paint, and on rotate/resize)
  const ro = new ResizeObserver(() => { const nw = Math.floor(el.clientWidth); if (nw > 0 && nw !== plot.width) plot.setSize({ width: nw, height: 180 }); });
  ro.observe(el);
  const destroy = plot.destroy.bind(plot);
  plot.destroy = () => { ro.disconnect(); destroy(); };
  return plot;
}

function renderCharts(container, metricIds, samples, gpus) {
  destroyCharts();
  container.innerHTML = "";
  const cat = Object.fromEntries(state.system.metrics.map((m) => [m.id, m]));
  for (const id of metricIds) {
    const m = cat[id]; if (!m) continue;
    const box = document.createElement("div"); box.className = "card chart";
    box.innerHTML = `<div class="title">${esc(m.label)} ${m.unit ? "(" + esc(m.unit) + ")" : ""}</div><div class="latest-row"></div><div class="plot"></div>`;
    container.appendChild(box);
    const data = metricSeries(m, samples, gpus);
    const latest = $(".latest-row", box);
    latest.innerHTML = latestHtml(data, m.unit);
    state.charts.push({ metric: m, gpus, latest, plot: makeChart($(".plot", box), m, data) });
  }
}

function pushChartSample(samples) {
  for (const c of state.charts) {
    const d = metricSeries(c.metric, samples, c.gpus);
    c.plot.setData([d.x, ...d.ys]);
    c.latest.innerHTML = latestHtml(d, c.metric.unit);
  }
}

async function metricPicker(container, current, onChange) {
  const groups = {};
  for (const m of state.system.metrics) (groups[m.scope] = groups[m.scope] || []).push(m);
  container.innerHTML = Object.entries(groups).map(([g, ms]) =>
    `<div><div class="small muted">${esc(g)}</div><div class="chips">${ms.map((m) =>
      `<label><input type="checkbox" value="${esc(m.id)}" ${current.includes(m.id) ? "checked" : ""}> ${esc(m.label)}</label>`).join("")}</div></div>`).join("");
  container.addEventListener("change", async () => {
    const ids = $$("input:checked", container).map((i) => i.value);
    const saved = await api("/layouts/default", { method: "PUT", json: { charts: ids } });
    state.layout = saved.charts; onChange(saved.charts);
  });
}

async function getLayout() {
  if (!state.layout) state.layout = (await api("/layouts/default")).charts;
  return state.layout;
}

// ---------------- New run ----------------
function paramInput(mod, key, spec, value) {
  const id = `p-${mod}-${key}`;
  const lab = esc(spec.label || key);
  if (spec.type === "int") return `<label class="field"><span>${lab}</span><input type="number" id="${id}" value="${esc(value)}" min="${spec.min ?? ""}" max="${spec.max ?? ""}"></label>`;
  if (spec.type === "bool") return `<label><input type="checkbox" id="${id}" ${value ? "checked" : ""}> ${lab}</label>`;
  if (spec.type === "choice") return `<label class="field"><span>${lab}</span><select id="${id}">${spec.choices.map((c) => `<option ${c === value ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></label>`;
  if (spec.type === "multi") return `<div class="field"><span>${lab}</span><div class="chips" id="${id}">${spec.choices.map((c) => `<label><input type="checkbox" value="${esc(c)}" ${value.includes(c) ? "checked" : ""}> ${esc(c)}</label>`).join("")}</div></div>`;
  if (spec.type === "ints") return `<label class="field"><span>${lab} (comma separated)</span><input type="text" id="${id}" value="${esc(value.join(","))}"></label>`;
  return "";
}

function readParam(mod, key, spec) {
  const el = $(`#p-${mod}-${key}`);
  if (spec.type === "int") return parseInt(el.value, 10);
  if (spec.type === "bool") return el.checked;
  if (spec.type === "choice") return el.value;
  if (spec.type === "multi") return $$("input:checked", el).map((i) => i.value);
  if (spec.type === "ints") return el.value.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
}

async function viewRun(_, gen = state.gen) {
  const sys = state.system;
  const models = state.models = await api("/models");
  const presets = await api("/presets");
  fresh(gen);
  const present = models.standard.filter((m) => m.present);
  const opts = [...present.map((m) => `<option value="${esc(m.id)}">${esc(m.label)}${m.verified ? "" : " (not verified)"}</option>`),
    ...models.custom.map((m) => `<option value="${esc(m.id)}">custom: ${esc(m.label)}</option>`)].join("");
  const gpus = sys.snapshot.gpus || [];
  const engines = sys.engines.filter((e) => e.available);
  $("#view").innerHTML = `
  <h1>New run</h1>
  ${sys.active_run ? `<div class="card warn">A run is active: <a href="#/live/${esc(sys.active_run)}">watch it</a>.</div>` : ""}
  <div class="card">
    <div class="row">
      <label class="field" style="flex:2"><span>Model</span><select id="r-model">${opts || "<option value=''>no models yet: see Models</option>"}</select></label>
      <label class="field"><span>Engine</span><select id="r-engine">${engines.map((e) => `<option value="${esc(e.name)}">${esc(e.label || e.name)}</option>`).join("")}</select></label>
      <label class="field"><span>Multi-GPU split</span><select id="r-split"><option value="">engine default</option><option>layer</option><option>tensor</option></select></label>
      <label class="field" style="flex:1"><span>Label (optional)</span><input type="text" id="r-label" maxlength="120"></label>
    </div>
    <h3>GPUs</h3>
    <div class="chips" id="r-gpus">${gpus.map((g) => `<label><input type="checkbox" value="${g.index}" checked> ${g.index}: ${esc(g.name)} (${Math.round((g.memory_total || 0) / 1024)} GB)</label>`).join("") || '<span class="err">No GPUs visible.</span>'}</div>
    <h3>Presets</h3>
    <div class="row"><select id="r-preset"><option value="">(none)</option>${presets.map((p) => `<option>${esc(p.name)}</option>`).join("")}</select>
      <button class="secondary" id="r-preset-load">Load</button>
      <input type="text" id="r-preset-name" placeholder="preset name"><button class="secondary" id="r-preset-save">Save current</button></div>
  </div>
  <h2>Modules</h2>
  <div class="grid" id="r-modules">${sys.modules.map((m) => `
    <div class="module card off" data-mod="${esc(m.name)}">
      <label><input type="checkbox" class="mod-on"> <b>${esc(m.label)}</b></label>
      <div class="small muted">${esc(m.description)}</div>
      <div class="params">${Object.entries(m.params).map(([k, s]) => paramInput(m.name, k, s, s.default)).join("")}</div>
    </div>`).join("")}</div>
  <div class="card">
    <h3>Guardrails</h3>
    <div class="small muted">Warnings always show. Tick to also abort the run when one fires (after ${esc(sys.guardrails.sustain_s)} s sustained).</div>
    <div class="chips"><label><input type="checkbox" id="g-low"> abort on low clock (&lt; ${esc(sys.guardrails.low_clock_mhz)} MHz while busy)</label>
      <label><input type="checkbox" id="g-thermal"> abort on temperature &gt; ${esc(sys.guardrails.thermal_warn_c)} °C</label></div>
    <div class="row" style="margin-top:12px"><button id="r-review">Review run</button></div>
  </div>
  <div id="r-confirm"></div>`;
  $$(".module").forEach((el) => $(".mod-on", el).addEventListener("change", (e) => el.classList.toggle("off", !e.target.checked)));
  $("#r-review").onclick = reviewRun;
  $("#r-preset-save").onclick = async () => {
    const name = $("#r-preset-name").value.trim(); if (!name) return toast("Name the preset first");
    await api("/presets/" + encodeURIComponent(name), { method: "PUT", json: { spec: collectSpec(true) } }); toast("Preset saved"); route();
  };
  $("#r-preset-load").onclick = () => { const p = presets.find((x) => x.name === $("#r-preset").value); if (p) applySpec(p.spec); };
}

function collectSpec(forPreset = false) {
  const modules = {};
  for (const el of $$(".module")) {
    if (!$(".mod-on", el).checked) continue;
    const name = el.dataset.mod, desc = state.system.modules.find((m) => m.name === name);
    modules[name] = Object.fromEntries(Object.entries(desc.params).map(([k, s]) => [k, readParam(name, k, s)]));
  }
  const abort_on = [$("#g-low").checked && "low_clock", $("#g-thermal").checked && "thermal"].filter(Boolean);
  return { model: $("#r-model").value, engine: $("#r-engine").value, split: $("#r-split").value, label: forPreset ? "" : $("#r-label").value,
    gpus: $$("#r-gpus input:checked").map((i) => +i.value), modules, abort_on };
}

function applySpec(spec) {
  if (spec.model) $("#r-model").value = spec.model;
  if (spec.engine) $("#r-engine").value = spec.engine;
  $("#r-split").value = spec.split || "";
  $$("#r-gpus input").forEach((i) => (i.checked = (spec.gpus || []).includes(+i.value)));
  for (const el of $$(".module")) {
    const name = el.dataset.mod, p = (spec.modules || {})[name];
    $(".mod-on", el).checked = !!p; el.classList.toggle("off", !p);
    if (!p) continue;
    const desc = state.system.modules.find((m) => m.name === name);
    for (const [k, s] of Object.entries(desc.params)) {
      if (!(k in p)) continue;
      const inp = $(`#p-${name}-${k}`);
      if (s.type === "bool") inp.checked = !!p[k];
      else if (s.type === "multi") $$("input", inp).forEach((i) => (i.checked = p[k].includes(i.value)));
      else if (s.type === "ints") inp.value = p[k].join(",");
      else inp.value = p[k];
    }
  }
  $("#g-low").checked = (spec.abort_on || []).includes("low_clock");
  $("#g-thermal").checked = (spec.abort_on || []).includes("thermal");
}

async function reviewRun() {
  const box = $("#r-confirm");
  let r;
  try { r = await api("/runs", { json: collectSpec() }); }
  catch (e) { box.innerHTML = `<div class="card err">${esc(e.message)}</div>`; return; }
  const pf = r.preflight;
  box.innerHTML = `<div class="card">
    <h2>Confirm</h2>
    <dl class="kv"><dt>Model</dt><dd>${esc(r.spec.model)}</dd><dt>Engine</dt><dd>${esc(r.spec.engine)}</dd>
      <dt>GPUs</dt><dd>${esc(r.spec.gpus.join(", "))}${r.spec.gpus.length > 1 ? ", " + esc(r.spec.split) + " split" : ""}</dd><dt>Modules</dt><dd>${esc(Object.keys(r.spec.modules).join(", "))}</dd>
      <dt>Estimated time</dt><dd>about ${esc(dur(r.estimate_s))} (rough)</dd></dl>
    ${pf.block.length ? `<h3 class="err">Blocked</h3><ul>${pf.block.map((b) => `<li class="err">${esc(b)}</li>`).join("")}</ul>` : ""}
    ${pf.warn.length ? `<h3 class="warn">Warnings</h3><ul>${pf.warn.map((b) => `<li class="warn">${esc(b)}</li>`).join("")}</ul>` : ""}
    <div class="row"><button id="c-start" ${pf.block.length ? "disabled" : ""}>Start run</button></div></div>`;
  $("#c-start").onclick = async () => {
    try { await api(`/runs/${r.id}/start`, { method: "POST" }); updateActive(r.id); location.hash = "#/live/" + r.id; }
    catch (e) { toast(e.message, 8000); }
  };
  box.scrollIntoView({ behavior: "smooth" });
}

// ---------------- Live ----------------
async function viewLive(id, gen = state.gen) {
  if (!id) id = state.system.active_run;
  if (!id) { $("#view").innerHTML = `<div class="card">No run is active. <a href="#/history">History</a></div>`; return; }
  const d = await api("/runs/" + id);
  if (!["created", "running"].includes(d.run.status)) { location.hash = "#/detail/" + id; return; }
  const gpus = d.run.spec.gpus.map(String);
  const samples = (await api(`/runs/${id}/telemetry?max_points=900`));
  const layout = await getLayout();
  fresh(gen);
  $("#view").innerHTML = `
  <h1>Live: ${esc(d.run.spec.label || id)} <span class="pill running" id="l-status">running</span></h1>
  <div class="card"><div class="row" style="justify-content:space-between"><div id="l-step" class="muted">starting…</div>
    <button class="danger" id="l-abort">Abort</button></div><div class="bar" style="margin-top:8px"><div id="l-bar"></div></div></div>
  <details class="card"><summary>Choose charts</summary><div id="l-picker" style="margin-top:8px"></div></details>
  <div class="charts" id="l-charts"></div>
  <div class="grid">
    <div class="card"><h2>Results so far</h2><div class="table-wrap" id="l-results"></div></div>
    <div class="card"><h2>Events</h2><ul class="events small" id="l-events"></ul></div>
  </div>
  <div class="card"><div class="row" style="justify-content:space-between"><h2>Log</h2>
    <select id="l-stream"><option value="">all</option><option>runner</option><option>server</option><option>bench</option><option>agent</option></select></div>
    <pre class="log" id="l-log"></pre></div>`;
  renderCharts($("#l-charts"), layout, samples, gpus);
  metricPicker($("#l-picker"), layout, (ids) => renderCharts($("#l-charts"), ids, samples, gpus));
  $("#l-abort").onclick = async () => { if (confirm("Abort this run?")) { try { await api(`/runs/${id}/abort`, { method: "POST" }); } catch (e) { toast(e.message); } } };
  let results = d.results.slice();
  const showResults = () => { $("#l-results").innerHTML = resultsTable(summarizeLocal(results)); };
  showResults();
  const evs = $("#l-events");
  const addEvent = (e) => { const li = document.createElement("li"); li.className = e.level === "info" ? "" : e.level === "warn" ? "warn" : "err"; li.textContent = `${new Date(e.t * 1000).toLocaleTimeString()} ${e.message}`; evs.prepend(li); };
  d.events.forEach(addEvent);
  const logEl = $("#l-log"); const logLines = [];
  const addLog = (stream, line) => {
    logLines.push([stream, line]); if (logLines.length > 600) logLines.splice(0, 200);
    const f = $("#l-stream").value; if (f && f !== stream) return;
    const atBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 20;
    logEl.textContent += `[${stream}] ${line}\n`;
    if (logEl.textContent.length > 120000) logEl.textContent = logEl.textContent.slice(-80000);
    if (atBottom) logEl.scrollTop = logEl.scrollHeight;
  };
  const prior = await api(`/runs/${id}/logs?tail=300`);
  fresh(gen);
  prior.forEach((l) => addLog(l.stream, l.line));
  $("#l-stream").onchange = () => { const f = $("#l-stream").value; logEl.textContent = logLines.filter(([s]) => !f || s === f).map(([s, l]) => `[${s}] ${l}`).join("\n") + "\n"; };
  const setProgress = (p) => { if (!p) return; $("#l-step").textContent = `${p.module}: ${p.step}/${p.steps} ${p.label}`; $("#l-bar").style.width = (100 * p.step / p.steps) + "%"; };
  setProgress(d.progress);
  let lastDraw = 0;
  state.live = {
    id,
    onEvent(ev) {
      if (ev.type === "telemetry") {
        samples.push(ev.sample); if (samples.length > 3600) samples.splice(0, samples.length - 3600);
        if (Date.now() - lastDraw > 900) { pushChartSample(samples); lastDraw = Date.now(); }
      } else if (ev.type === "log") addLog(ev.stream, ev.line);
      else if (ev.type === "event") addEvent(ev);
      else if (ev.type === "result") { results.push(ev); showResults(); }
      else if (ev.type === "progress") setProgress(ev);
      else if (ev.type === "status" && ev.status !== "running") {
        $("#l-status").textContent = ev.status; $("#l-status").className = "pill " + ev.status;
        toast(`Run ${ev.status}${ev.error ? ": " + ev.error : ""}`, 8000);
        setTimeout(() => { if (location.hash === "#/live/" + id) location.hash = "#/detail/" + id; }, 2500);
      }
    },
  };
}

function summarizeLocal(results) {
  const g = {};
  for (const r of results) {
    if (typeof r.value !== "number") continue;
    const k = [r.module, r.variant, r.metric].join("|");
    (g[k] = g[k] || { module: r.module, variant: r.variant, metric: r.metric, unit: r.unit, vals: [] }).vals.push(r.value);
  }
  const out = {};
  for (const x of Object.values(g)) {
    const v = x.vals.slice().sort((a, b) => a - b), n = v.length;
    const med = n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
    ((out[x.module] = out[x.module] || {})[x.variant] = out[x.module][x.variant] || {})[x.metric] = { median: med, min: v[0], max: v[n - 1], n, unit: x.unit };
  }
  return out;
}

const MOD_ORDER = ["speed", "server", "offload", "agent"];
const byModule = (a, b) => MOD_ORDER.indexOf(a[0]) - MOD_ORDER.indexOf(b[0]);
function resultsTable(summary) {
  const rows = [];
  for (const [mod, vars] of Object.entries(summary).sort(byModule)) for (const [v, mets] of Object.entries(vars)) for (const [m, s] of Object.entries(mets))
    rows.push(`<tr><td>${esc(mod)}</td><td>${esc(v)}</td><td>${esc(m)}</td><td class="num">${fmt(s.median, 2)} ${esc(s.unit)}</td><td class="num muted">${fmt(s.min, 2)}–${fmt(s.max, 2)}</td><td class="num">${s.n}</td></tr>`);
  return rows.length ? `<table><tr><th>module</th><th>variant</th><th>metric</th><th class="num">median</th><th class="num">range</th><th class="num">n</th></tr>${rows.join("")}</table>` : '<p class="muted">No results yet.</p>';
}

// ---------------- History ----------------
async function viewHistory(_, gen = state.gen) {
  const runs = await api("/runs");
  fresh(gen);
  $("#view").innerHTML = `
  <h1>History</h1>
  <div class="card row"><button id="h-compare" disabled>Compare selected</button>
    <label class="secondary"><input type="file" id="h-import" accept=".json,application/json"> Import a result file</label></div>
  <div class="card table-wrap"><table>
    <tr><th></th><th>when</th><th>label</th><th>hardware</th><th>model</th><th>engine</th><th>GPUs</th><th>modules</th><th>status</th></tr>
    ${runs.map((r) => `<tr><td><input type="checkbox" class="h-sel" value="${esc(r.id)}"></td>
      <td class="small">${esc(when(r.created_at))}</td>
      <td><a href="#/${r.status === "running" ? "live" : "detail"}/${esc(r.id)}">${esc(r.label || r.id)}</a></td>
      <td class="small">${esc(r.hardware)}</td><td class="small">${esc(r.model)}${r.comparable === false ? ' <span class="pill">custom</span>' : ""}</td>
      <td>${esc(r.engine)}</td><td>${esc((r.gpus || []).join(","))}</td><td class="small">${esc((r.modules || []).join(", "))}</td>
      <td><span class="pill ${esc(r.status)}">${esc(r.status)}</span>${r.invalid ? ' <span class="pill invalid">invalid</span>' : ""}</td></tr>`).join("")
      || '<tr><td colspan="9" class="muted">No runs yet.</td></tr>'}
  </table></div>`;
  const upd = () => { $("#h-compare").disabled = $$(".h-sel:checked").length < 2; };
  $$(".h-sel").forEach((c) => (c.onchange = upd));
  $("#h-compare").onclick = () => { location.hash = "#/compare/" + $$(".h-sel:checked").map((c) => c.value).join(","); };
  $("#h-import").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    try { const b = JSON.parse(await f.text()); const r = await api("/import", { json: b }); toast("Imported"); location.hash = "#/detail/" + r.id; }
    catch (err) { toast("Import failed: " + err.message, 8000); }
  };
}

// ---------------- Run detail ----------------
async function viewDetail(id, gen = state.gen) {
  const d = await api("/runs/" + id);
  fresh(gen);
  const r = d.run, spec = r.spec;
  const snap = r.snapshot || {};
  $("#view").innerHTML = `
  <h1>${esc(spec.label || r.id)} <span class="pill ${esc(r.status)}">${esc(r.status)}</span>${r.invalid ? ' <span class="pill invalid">invalid</span>' : ""}</h1>
  ${r.error ? `<div class="card err">${esc(r.error)}</div>` : ""}
  <div class="grid">
    <div class="card"><h2>Run</h2><dl class="kv">
      <dt>Model</dt><dd>${esc(spec.model)}${spec.comparable === false ? " (custom, not comparable)" : ""}</dd><dt>Engine</dt><dd>${esc(spec.engine)}</dd>
      <dt>GPUs</dt><dd>${esc((spec.gpus || []).join(", "))}${(spec.gpus || []).length > 1 && spec.split ? ", " + esc(spec.split) + " split" : ""}</dd><dt>Started</dt><dd>${esc(when(r.started_at))}</dd>
      <dt>Finished</dt><dd>${esc(when(r.finished_at))}</dd><dt>Hardware</dt><dd>${esc(hwLine(snap))}</dd></dl>
      <div class="row" style="margin-top:10px">
        <a href="/api/runs/${esc(id)}/export.json" download><button class="secondary">Download result file</button></a>
        <a href="/api/runs/${esc(id)}/export.md" target="_blank" rel="noopener"><button class="secondary">Markdown</button></a>
        <button class="danger" id="d-del">Delete</button></div></div>
    ${d.imported ? "" : `<div class="card"><h2>Notes</h2>
      <textarea id="d-note">${esc(r.note)}</textarea>
      <label class="field"><span>Tags (comma separated)</span><input type="text" id="d-tags" value="${esc((r.tags || []).join(","))}"></label>
      <div class="row" style="margin-top:8px"><label><input type="checkbox" id="d-invalid" ${r.invalid ? "checked" : ""}> invalid</label>
        <input type="text" id="d-reason" placeholder="reason" value="${esc(r.invalid_reason || "")}" style="flex:1"></div>
      <div class="row" style="margin-top:8px"><button id="d-save">Save</button></div></div>`}
  </div>
  <div class="card"><h2>Results</h2><div class="table-wrap">${resultsTable(d.summary)}</div></div>
  ${d.agent_tasks.length ? `<div class="card"><h2>Agent tasks</h2><div class="table-wrap"><table><tr><th>task</th><th class="num">run</th><th class="num">wall</th><th class="num">exit</th><th>pass</th></tr>
    ${d.agent_tasks.map((t) => `<tr><td>${esc(t.task)}</td><td class="num">${t.rep + 1}</td><td class="num">${fmt(t.wall_s, 0)} s</td><td class="num">${esc(t.exit_code)}</td><td class="${t.success ? "ok" : "err"}">${t.success ? "pass" : "fail"}</td></tr>`).join("")}</table></div></div>` : ""}
  ${d.events.length ? `<div class="card"><h2>Events</h2><ul class="events small">${d.events.map((e) => `<li class="${e.level === "info" ? "" : e.level === "warn" ? "warn" : "err"}">${esc(new Date(e.t * 1000).toLocaleTimeString())} ${esc(e.message)}</li>`).join("")}</ul></div>` : ""}
  ${d.imported ? "" : `<details class="card"><summary>Telemetry charts</summary><div id="d-picker" style="margin:8px 0"></div><div class="charts" id="d-charts"></div></details>
  <details class="card"><summary>Log</summary><pre class="log" id="d-log"></pre></details>`}`;
  $("#d-del").onclick = async () => { if (confirm("Delete this run and all its data?")) { await api("/runs/" + id, { method: "DELETE" }); location.hash = "#/history"; } };
  if (d.imported) return;
  $("#d-save").onclick = async () => {
    try {
      await api("/runs/" + id, { method: "PATCH", json: { note: $("#d-note").value, tags: $("#d-tags").value.split(","), invalid: $("#d-invalid").checked, invalid_reason: $("#d-reason").value } });
      toast("Saved");
    } catch (e) { toast(e.message); }
  };
  const det = $("details.card");
  det.addEventListener("toggle", async () => {
    if (!det.open || det.dataset.loaded) return; det.dataset.loaded = 1;
    const samples = await api(`/runs/${id}/telemetry?max_points=1500`); const gpus = (spec.gpus || []).map(String);
    const layout = await getLayout();
    if (gen !== state.gen) return;
    renderCharts($("#d-charts"), layout, samples, gpus);
    metricPicker($("#d-picker"), layout, (ids) => renderCharts($("#d-charts"), ids, samples, gpus));
  });
  const logs = await api(`/runs/${id}/logs?tail=2000`);
  fresh(gen);
  $("#d-log").textContent = logs.map((l) => `[${l.stream}] ${l.line}`).join("\n");
}

function hwLine(snap) {
  const g = {}; (snap.gpus || []).forEach((x) => (g[x.name] = (g[x.name] || 0) + 1));
  const m = snap.memory || {};
  return [Object.entries(g).map(([n, c]) => `${c}x ${n}`).join(", ") || "?", (snap.cpu || {}).model, `${m.total_gb ?? "?"} GB ${m.type || ""} ${m.configured_speed || ""}`].filter(Boolean).join("; ");
}

// ---------------- Compare ----------------
async function viewCompare(ids, gen = state.gen) {
  const runs = await api("/compare?ids=" + encodeURIComponent(ids));
  fresh(gen);
  const keys = new Map();
  for (const r of runs) for (const [mod, vars] of Object.entries(r.summary).sort(byModule)) for (const [v, mets] of Object.entries(vars)) for (const [m, s] of Object.entries(mets))
    keys.set([mod, v, m].join("|"), { mod, v, m, unit: s.unit });
  const lowerBetter = (m) => /wall/.test(m);
  const rows = [...keys.entries()].map(([k, info]) => {
    const vals = runs.map((r) => (((r.summary[info.mod] || {})[info.v] || {})[info.m] || {}).median);
    const nums = vals.filter((x) => typeof x === "number");
    const spread = nums.length > 1 && Math.max(...nums) !== Math.min(...nums);
    const best = spread ? (lowerBetter(info.m) ? Math.min(...nums) : Math.max(...nums)) : null;
    const base = vals[0];
    return `<tr><td>${esc(info.mod)}</td><td>${esc(info.v)}</td><td>${esc(info.m)}</td>${vals.map((x, i) =>
      `<td class="num ${x === best ? "best" : ""}">${fmt(x, 2)}${i && typeof x === "number" && typeof base === "number" && base ? ` <span class="muted small">${(100 * (x - base) / base >= 0 ? "+" : "") + (100 * (x - base) / base).toFixed(0)}%</span>` : ""}</td>`).join("")}<td class="muted">${esc(info.unit)}</td></tr>`;
  });
  $("#view").innerHTML = `
  <h1>Compare</h1>
  <div class="card table-wrap"><table>
    <tr><th></th>${runs.map((r, i) => `<th>${i ? "" : "baseline: "}<a href="#/detail/${esc(r.id)}">${esc(r.label || r.id)}</a>${r.invalid ? ' <span class="pill invalid">invalid</span>' : ""}${r.comparable === false ? ' <span class="pill">custom</span>' : ""}</th>`).join("")}</tr>
    <tr><td class="muted">hardware</td>${runs.map((r) => `<td class="small">${esc(r.hardware)}</td>`).join("")}</tr>
    <tr><td class="muted">engine / model / GPUs</td>${runs.map((r) => `<td class="small">${esc(r.engine)} / ${esc(r.model)} / ${esc((r.gpus || []).join(","))}</td>`).join("")}</tr>
  </table></div>
  <div class="card table-wrap"><table><tr><th>module</th><th>variant</th><th>metric</th>${runs.map((r) => `<th class="num">${esc(r.label || r.id)}</th>`).join("")}<th>unit</th></tr>${rows.join("")}</table>
  <p class="small muted">Medians. Bold = best; % is against the first run. Only compare runs with the same model file and settings.</p></div>`;
}

// ---------------- Models ----------------
async function viewModels(_, gen = state.gen) {
  const m = await api("/models");
  fresh(gen);
  $("#view").innerHTML = `
  <h1>Models</h1>
  <p class="muted">Folder: <code>${esc(m.models_dir)}</code> · free: ${m.free_bytes != null ? gb(m.free_bytes) : "?"}. Results compare across machines only with the same file; verify the checksum before sharing.</p>
  <div class="card table-wrap"><table><tr><th>model</th><th class="num">size</th><th>VRAM</th><th>status</th><th></th></tr>
    ${m.standard.map((s) => `<tr><td>${esc(s.label)}<div class="small muted">${esc(s.repo)}${s.mtp ? " · MTP" : ""} · ${esc(s.kind)}</div></td>
      <td class="num">${gb(s.size)}</td><td>${esc(s.vram_gb)} GB</td>
      <td>${s.job && s.job.status === "running" ? `${esc(s.job.kind)} ${(100 * s.job.done / s.job.total).toFixed(0)}%`
        : s.verified ? '<span class="ok">verified</span>' : s.present ? '<span class="warn">present, not verified</span>' : '<span class="muted">not downloaded</span>'}
        ${s.job && s.job.status === "failed" ? `<div class="err small">${esc(s.job.error)}</div>` : ""}</td>
      <td>${s.present ? (s.verified ? "" : `<button class="secondary" data-verify="${esc(s.id)}">Verify</button>`) : `<button data-dl="${esc(s.id)}">Download</button>`}</td></tr>`).join("")}
  </table></div>
  <div class="card"><h2>Custom models</h2>${m.custom.length ? `<ul>${m.custom.map((c) => `<li>${esc(c.label)} <span class="muted">(${gb(c.size)})</span></li>`).join("")}</ul>` : '<p class="muted">Any other .gguf in the models folder shows up here. Results with custom models are marked not comparable.</p>'}</div>`;
  $$("[data-dl]").forEach((b) => (b.onclick = async () => { if (!confirm("Download this model? Check you have the disk space.")) return; try { await api(`/models/${b.dataset.dl}/download`, { method: "POST" }); route(); } catch (e) { toast(e.message, 8000); } }));
  $$("[data-verify]").forEach((b) => (b.onclick = async () => { try { await api(`/models/${b.dataset.verify}/verify`, { method: "POST" }); route(); } catch (e) { toast(e.message, 8000); } }));
  if (m.standard.some((s) => s.job && s.job.status === "running")) setTimeout(() => { if (location.hash === "#/models" && gen === state.gen) route(); }, 3000);
}

// ---------------- System ----------------
async function viewSystem(_, gen = state.gen) {
  const s = await api("/system?refresh=1"); state.system = s;
  fresh(gen);
  const snap = s.snapshot, mem = snap.memory || {};
  $("#view").innerHTML = `
  <h1>System</h1>
  <div class="card table-wrap"><h2>GPUs</h2><table><tr><th>#</th><th>name</th><th>VRAM</th><th>power limit</th><th>PCIe now / max</th><th>max SM clock</th><th>driver</th></tr>
    ${(snap.gpus || []).map((g) => `<tr><td>${g.index}</td><td>${esc(g.name)}</td><td>${Math.round((g.memory_total || 0) / 1024)} GB</td><td>${esc(g.power_limit)} W</td>
      <td>gen ${esc(g.pcie_link_gen_current)} x${esc(g.pcie_link_width_current)} / gen ${esc(g.pcie_link_gen_max)} x${esc(g.pcie_link_width_max)}</td><td>${esc(g.clocks_max_sm)} MHz</td><td>${esc(g.driver_version)}</td></tr>`).join("")}</table>
    <p class="small muted">PCIe gen drops at idle; check it under load if it looks low.</p></div>
  <div class="grid">
    <div class="card"><h2>CPU and memory</h2><dl class="kv"><dt>CPU</dt><dd>${esc(snap.cpu.model)} (${esc(snap.cpu.cores)} cores, ${esc(snap.cpu.threads)} threads)</dd>
      <dt>RAM</dt><dd>${esc(mem.total_gb)} GB ${esc(mem.type || "")} ${esc(mem.configured_speed || "")}</dd><dt>DIMMs</dt><dd>${esc(mem.populated_slots ?? "unknown (dmidecode not available)")}</dd>
      <dt>Board</dt><dd>${esc([snap.board.board_vendor, snap.board.board_name].filter(Boolean).join(" ") || "?")}</dd><dt>Kernel</dt><dd>${esc(snap.os.kernel)}</dd>
      <dt>CUDA (driver)</dt><dd>${esc(snap.cuda_driver_version)}</dd></dl></div>
    <div class="card"><h2>Engines</h2><ul>${s.engines.map((e) => `<li><b>${esc(e.label || e.name)}</b> <span class="muted small">${esc(e.commit)} · archs ${esc(e.cuda_archs)} · multi-GPU ${esc(e.multi_gpu_split)}${e.mtp ? " · MTP" : ""}${e.available ? "" : " · missing binaries"}</span>${e.notes ? `<div class="small muted">${esc(e.notes)}</div>` : ""}</li>`).join("")}</ul>
      <p class="small muted">Kit ${esc(s.kit_version)}</p></div>
  </div>
  ${snap.gpu_topology ? `<details class="card"><summary>GPU topology</summary><pre>${esc(snap.gpu_topology)}</pre></details>` : ""}
  ${Object.keys(snap.errors || {}).length ? `<div class="card warn"><h2>Not available</h2><ul>${Object.entries(snap.errors).map(([k, v]) => `<li>${esc(k)}: ${esc(v)}</li>`).join("")}</ul></div>` : ""}
  <div class="row"><button class="secondary" id="s-logout">Log out</button></div>`;
  $("#s-logout").onclick = async () => { await api("/logout", { method: "POST" }); showLogin(); };
}

boot().catch((e) => { if (e.message !== "login required") toast(e.message, 8000); });

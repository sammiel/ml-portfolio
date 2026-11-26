// Shared helpers: loading exported models and scoring logistic regressions.

async function loadModel(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Could not load ${path}`);
  return res.json();
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

// Returns the probability and each feature's contribution to the log-odds,
// measured against an average customer (standardised value of zero).
function scoreLogistic(model, values) {
  let z = model.intercept;
  const contributions = model.features.map((f, i) => {
    const standardised = (values[i] - model.mean[i]) / model.scale[i];
    const c = model.coef[i] * standardised;
    z += c;
    return { label: f.label, value: c };
  });
  return { probability: sigmoid(z), contributions };
}

function formatValue(feature, v) {
  if (feature.unit === "yes/no") return v >= 0.5 ? "Yes" : "No";
  const n = Number.isInteger(feature.step) ? Math.round(v) : v.toFixed(1);
  if (feature.unit === "%") return `${n}%`;
  if (feature.unit === "pts") return `${v > 0 ? "+" : ""}${n} pts`;
  return `${n} ${feature.unit}`;
}

// Builds one slider per feature inside `container` and calls onChange with
// the current values whenever any slider moves.
function buildSliders(container, features, initial, onChange) {
  container.innerHTML = "";
  const inputs = features.map((f, i) => {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const id = `f-${f.name}`;
    wrap.innerHTML = `
      <div class="row"><label for="${id}">${f.label}</label><output id="${id}-out"></output></div>
      <input type="range" id="${id}" min="${f.min}" max="${f.max}" step="${f.step}" value="${initial[i]}">`;
    container.appendChild(wrap);
    const input = wrap.querySelector("input");
    const out = wrap.querySelector("output");
    const sync = () => { out.textContent = formatValue(f, Number(input.value)); };
    sync();
    input.addEventListener("input", () => { sync(); onChange(read()); });
    return { input, sync };
  });
  function read() { return inputs.map((x) => Number(x.input.value)); }
  function set(values) {
    inputs.forEach((x, i) => { x.input.value = values[i]; x.sync(); });
    onChange(read());
  }
  return { read, set };
}

// Renders contribution bars. `goodIsPositive` flips colours for models where
// a higher log-odds is the good outcome (renewal) rather than the bad one (churn).
function renderDrivers(container, contributions, goodIsPositive) {
  const max = Math.max(1, ...contributions.map((c) => Math.abs(c.value)));
  const sorted = [...contributions].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  container.innerHTML = sorted.map((c) => {
    const width = (Math.abs(c.value) / max) * 50;
    const left = c.value >= 0 ? 50 : 50 - width;
    const good = goodIsPositive ? c.value >= 0 : c.value < 0;
    const colour = good ? "var(--good)" : "var(--bad)";
    return `<div class="driver"><span>${c.label}</span>
      <div class="bar-wrap"><div class="bar" style="left:${left}%;width:${width}%;background:${colour}"></div></div></div>`;
  }).join("");
}

function pct(x, digits = 0) {
  return `${(x * 100).toFixed(digits)}%`;
}

// Shared behaviour across all pages: mobile nav toggle + live ticker strip.

document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("nav-toggle");
  const nav = document.querySelector(".mainnav");
  if (toggle && nav) {
    toggle.addEventListener("click", () => nav.classList.toggle("open"));
  }

  populateTicker();
});

async function populateTicker() {
  const track = document.getElementById("ticker-track");
  const footerModel = document.getElementById("footer-model");
  if (!track) return;

  try {
    const res = await fetch("/api/model-metrics");
    const data = await res.json();
    const best = data.comparison.find(r => r.Model === data.metadata.best_model_name) || data.comparison[data.comparison.length - 1];

    const items = [
      `MODEL: ${data.metadata.best_model_name.toUpperCase()}`,
      `WAPE: ${best["WAPE (%)"]}%`,
      `MAE: $${Number(best["MAE"]).toLocaleString()}`,
      `RMSE: $${Number(best["RMSE"]).toLocaleString()}`,
      `LEAD TIME: ${data.metadata.lead_time_weeks}W`,
      `SAFETY STOCK: ${(data.metadata.safety_stock_pct * 100).toFixed(0)}%`,
      `STATUS: LIVE`,
    ];

    // Duplicate the list so the CSS scroll animation loops seamlessly.
    const html = items.map(t => `<span class="ticker-item">${t}</span>`).join("");
    track.innerHTML = html + html;

    if (footerModel) footerModel.textContent = data.metadata.best_model_name;
  } catch (e) {
    track.innerHTML = `<span class="ticker-item">SIGNAL UNAVAILABLE — CHECK API</span>`;
  }
}

// ---- small shared helpers used by page-specific scripts ----

function fmtCurrency(n) {
  return "$" + Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function fmtNumber(n) {
  return Number(n).toLocaleString();
}

function qs(sel, root = document) {
  return root.querySelector(sel);
}

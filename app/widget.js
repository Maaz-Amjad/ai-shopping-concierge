/* Shopping Concierge drop-in widget. Paste into any store page:
 *   <div id="concierge-widget"></div>
 *   <script src="https://YOUR-API/widget.js" data-store="mystore"></script>
 * Optional attributes: data-api (defaults to the script's origin), data-user (shopper id for personalization),
 * data-n (results, default 5). Product text and answers are inserted as text, never as HTML.
 */
(function () {
  "use strict";
  var script = document.currentScript;
  var api = (script.getAttribute("data-api") || new URL(script.src).origin).replace(/\/$/, "");
  var store = script.getAttribute("data-store") || null;
  var user = script.getAttribute("data-user") || null;
  var n = parseInt(script.getAttribute("data-n") || "5", 10);

  var root = document.getElementById("concierge-widget");
  if (!root) {
    root = document.createElement("div");
    root.id = "concierge-widget";
    script.parentNode.insertBefore(root, script);
  }

  var css = [
    "#concierge-widget{font:15px/1.45 system-ui,sans-serif;max-width:760px;color:#1d1d1b}",
    "#concierge-widget form{display:flex;gap:8px}",
    "#concierge-widget input{flex:1;padding:10px 12px;border:1px solid #c9c7bf;border-radius:8px;font:inherit}",
    "#concierge-widget button{padding:10px 16px;border:0;border-radius:8px;background:#2a78d6;color:#fff;font:inherit;cursor:pointer}",
    "#concierge-widget button:disabled{opacity:.6;cursor:wait}",
    ".cw-chips{margin:10px 0;display:flex;flex-wrap:wrap;gap:6px}",
    ".cw-chip{background:#eef4fc;color:#1d4f8f;border-radius:999px;padding:2px 10px;font-size:13px}",
    ".cw-answer{white-space:pre-wrap;background:#f6f5f1;border-radius:8px;padding:12px;margin:10px 0}",
    ".cw-card{border:1px solid #e3e1da;border-radius:8px;padding:10px 12px;margin:8px 0}",
    ".cw-title{font-weight:600}.cw-meta{color:#6b6a64;font-size:13px}.cw-why{color:#6b6a64;font-size:12px}",
    ".cw-error{color:#b3261e}"
  ].join("");
  var style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }

  var form = el("form");
  var input = el("input");
  input.placeholder = "Describe what you need, e.g. quiet ice maker under $150";
  input.setAttribute("aria-label", "Search products");
  var button = el("button", null, "Search");
  form.appendChild(input);
  form.appendChild(button);
  var out = el("div");
  root.appendChild(form);
  root.appendChild(out);

  function price(p) { return p === null || p === undefined ? "price not listed" : "$" + Number(p).toFixed(2); }

  function render(data) {
    out.textContent = "";
    var pq = data.parsed || {};
    var chips = el("div", "cw-chips");
    if (pq.max_price !== null && pq.max_price !== undefined) chips.appendChild(el("span", "cw-chip", "max $" + pq.max_price));
    if (pq.min_price !== null && pq.min_price !== undefined) chips.appendChild(el("span", "cw-chip", "min $" + pq.min_price));
    if (pq.category) chips.appendChild(el("span", "cw-chip", "category: " + pq.category));
    if (pq.brand) chips.appendChild(el("span", "cw-chip", "brand: " + pq.brand));
    (pq.exclude_terms || []).forEach(function (t) { chips.appendChild(el("span", "cw-chip", "not " + t)); });
    if (chips.childNodes.length) out.appendChild(chips);
    out.appendChild(el("div", "cw-answer", data.answer));
    (data.products || []).forEach(function (p) {
      var card = el("div", "cw-card");
      card.appendChild(el("div", "cw-title", p.title));
      card.appendChild(el("div", "cw-meta", [p.brand || "", p.category || "", price(p.price)].filter(Boolean).join(" · ")));
      if (p.reason) card.appendChild(el("div", null, p.reason));
      card.appendChild(el("div", "cw-why", "why: relevance " + (p.s_sem || 0).toFixed(2) + ", personal " +
        (p.s_pers || 0).toFixed(2) + ", popularity " + (p.s_pop || 0).toFixed(2)));
      out.appendChild(card);
    });
  }

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    var q = input.value.trim();
    if (!q) return;
    button.disabled = true;
    out.textContent = "";
    out.appendChild(el("div", "cw-meta", "Searching…"));
    fetch(api + "/search", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query: q, store: store, user_id: user, n: n})
    }).then(function (r) {
      return r.json().then(function (body) { if (!r.ok) throw new Error(body.detail || r.statusText); return body; });
    }).then(render).catch(function (e) {
      out.textContent = "";
      out.appendChild(el("div", "cw-error", "Search failed: " + e.message));
    }).finally(function () { button.disabled = false; });
  });
})();

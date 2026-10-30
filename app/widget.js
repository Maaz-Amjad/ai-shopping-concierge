/* Shopping Concierge drop-in widget. Paste into any store page:
 *   <div id="concierge-widget"></div>
 *   <script src="https://YOUR-API/widget.js" data-store="mystore"></script>
 * Shows a floating chat bubble (shopping assistant); data-mode="search" shows the plain search box instead.
 * Optional attributes: data-api (defaults to the script's origin), data-user (shopper id for personalization),
 * data-n (results, default 5). Product text and answers are inserted as text, never as HTML; only https product
 * links are opened, in a new tab without access to this page (noopener).
 */
(function () {
  "use strict";
  var script = document.currentScript;
  var api = (script.getAttribute("data-api") || new URL(script.src).origin).replace(/\/$/, "");
  var store = script.getAttribute("data-store") || null;
  var headers = {"Content-Type": "application/json"};      // data-key: the store's API key, when keys are required
  if (script.getAttribute("data-key")) headers["X-API-Key"] = script.getAttribute("data-key");
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

  // Phase 13.6: read a newline-delimited JSON stream; onEvent gets every event, the promise the final one
  // ("answer" or "turn"; anything else means the stream was cut). Browsers without streams use the plain endpoint.
  function plainJson(path, body) {
    return fetch(api + path, {method: "POST", headers: headers, body: JSON.stringify(body)}).then(function (r) {
      return r.json().then(function (b) { if (!r.ok) throw new Error(b.detail || r.statusText); return b; });
    });
  }

  function streamJson(path, body, onEvent) {
    if (typeof ReadableStream === "undefined" || typeof TextDecoder === "undefined") {
      return plainJson(path.replace("/stream", ""), body);
    }
    return fetch(api + path, {method: "POST", headers: headers, body: JSON.stringify(body)}).then(function (r) {
      if (!r.ok) return r.json().then(function (b) { throw new Error(b.detail || r.statusText); });
      if (!r.body || !r.body.getReader) return plainJson(path.replace("/stream", ""), body);
      var reader = r.body.getReader(), decoder = new TextDecoder(), buffer = "", last = null;
      function pump() {
        return reader.read().then(function (chunk) {
          buffer += decoder.decode(chunk.value || new Uint8Array(), {stream: !chunk.done});
          var lines = buffer.split("\n");
          buffer = chunk.done ? "" : lines.pop();
          lines.forEach(function (line) {
            if (!line.trim()) return;
            var e = JSON.parse(line);
            if (e.type === "error") throw new Error(e.detail);
            last = e;
            onEvent(e);
          });
          if (!chunk.done) return pump();
          if (!last || (last.type !== "answer" && last.type !== "turn")) {
            throw new Error("the connection was interrupted; please try again");
          }
          return last;
        });
      }
      return pump();
    });
  }

  // Phase 13.8: what shoppers do (shown / click / save / buy) feeds the store's popularity. A random id per browser,
  // never the query text. Fire-and-forget: failures are ignored.
  var clientId = (function () {
    try {
      var id = localStorage.getItem("cw-client");
      if (!id) { id = Math.random().toString(36).slice(2) + Date.now().toString(36); localStorage.setItem("cw-client", id); }
      return id;
    } catch (e) { return "anonymous"; }
  })();
  function track(type, ids) {
    if (!store || !ids || !ids.length) return;
    try {
      fetch(api + "/events", {method: "POST", headers: headers, keepalive: true, body: JSON.stringify(
        {store: store, client_id: clientId, events: ids.slice(0, 50).map(function (id) { return {type: type, product_id: id}; })})
      }).catch(function () {});
    } catch (e) { /* never break the page */ }
  }

  function price(p) { return p === null || p === undefined ? "price not listed" : "$" + Number(p).toFixed(2); }
  function stars(p) {
    return p.avg_rating === null || p.avg_rating === undefined ? "" :
      "\u2605 " + Number(p.avg_rating).toFixed(1) + " (" + Number(p.rating_count || 0).toLocaleString() + " ratings)";
  }

  // ---------------- chat bubble (default; data-mode="search" keeps the plain search box) ----------------
  if ((script.getAttribute("data-mode") || "chat") === "chat") {
    var key = "concierge-chat:" + (store || "demo");
    var saved = null;
    try { saved = JSON.parse(sessionStorage.getItem(key) || "null"); } catch (e) { saved = null; }
    var state = (saved && saved.state) || null;
    var log = (saved && saved.log) || [];

    var chatCss = [
      ".cw-bubble{position:fixed;right:20px;bottom:20px;z-index:2147483000;border:0;border-radius:999px;",
      "padding:12px 18px;background:#2a78d6;color:#fff;font:15px system-ui,sans-serif;cursor:pointer;",
      "box-shadow:0 4px 16px rgba(0,0,0,.18)}",
      ".cw-panel{position:fixed;right:20px;bottom:76px;z-index:2147483000;width:min(380px,calc(100vw - 32px));",
      "height:min(560px,calc(100vh - 110px));display:none;flex-direction:column;background:#fff;color:#1d1d1b;",
      "border:1px solid #e3e1da;border-radius:14px;box-shadow:0 8px 28px rgba(0,0,0,.2);font:14px/1.45 system-ui,sans-serif}",
      ".cw-panel.open{display:flex}.cw-head{padding:12px 14px;font-weight:700;border-bottom:1px solid #eee;display:flex;justify-content:space-between}",
      ".cw-x{border:0;background:none;font-size:18px;cursor:pointer;color:#6b6a64}",
      ".cw-msgs{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}",
      ".cw-me{align-self:flex-end;background:#2a78d6;color:#fff;border-radius:12px 12px 2px 12px;padding:8px 11px;max-width:85%}",
      ".cw-bot{align-self:flex-start;background:#f2f1ec;border-radius:12px 12px 12px 2px;padding:8px 11px;max-width:92%;white-space:pre-wrap}",
      ".cw-pcard{border:1px solid #e3e1da;border-radius:8px;padding:8px;margin-top:6px;background:#fff}",
      ".cw-pcard .cw-t{font-weight:600}.cw-acts{display:flex;gap:6px;margin-top:6px}",
      ".cw-acts button{padding:4px 9px;border:1px solid #c9c7bf;border-radius:6px;background:#fff;color:#1d1d1b;cursor:pointer;font:12px system-ui}",
      ".cw-tbl{border-collapse:collapse;margin-top:6px;font-size:12px}.cw-tbl td,.cw-tbl th{border:1px solid #e3e1da;padding:3px 5px;vertical-align:top}",
      ".cw-form{display:flex;gap:6px;padding:10px;border-top:1px solid #eee}",
      ".cw-form input{flex:1;padding:9px 10px;border:1px solid #c9c7bf;border-radius:8px;font:inherit}",
      ".cw-form button{padding:9px 13px;border:0;border-radius:8px;background:#2a78d6;color:#fff;cursor:pointer}"
    ].join("");
    var st2 = document.createElement("style");
    st2.textContent = chatCss;
    document.head.appendChild(st2);

    var bubble = el("button", "cw-bubble", "💬 Shopping assistant");
    var panel = el("div", "cw-panel");
    var head = el("div", "cw-head", "Shopping assistant");
    var close = el("button", "cw-x", "×");
    close.setAttribute("aria-label", "Close");
    head.appendChild(close);
    var msgs = el("div", "cw-msgs");
    var cform = el("form", "cw-form");
    var cinput = el("input");
    cinput.placeholder = "Ask for a product, e.g. quiet ice maker under $150";
    cinput.setAttribute("aria-label", "Message the shopping assistant");
    var send = el("button", null, "Send");
    cform.appendChild(cinput);
    cform.appendChild(send);
    panel.appendChild(head);
    panel.appendChild(msgs);
    panel.appendChild(cform);
    document.body.appendChild(panel);
    document.body.appendChild(bubble);
    if (root && !root.childNodes.length) root.style.display = "none";

    function persist() {
      try { sessionStorage.setItem(key, JSON.stringify({state: state, log: log.slice(-30)})); } catch (e) { /* ignore */ }
    }
    function safeUrl(u) { return typeof u === "string" && /^https:\/\//.test(u) ? u : null; }

    function botMessage(data) {
      var box = el("div", "cw-bot", data.text);
      (data.products || []).forEach(function (p) {
        var c = el("div", "cw-pcard");
        c.appendChild(el("div", "cw-t", p.number + ". " + p.title));
        c.appendChild(el("div", "cw-meta", [p.brand || "", price(p.price), stars(p),
          p.in_stock === false ? "Out of stock" : ""].filter(Boolean).join(" · ")));
        if (p.reason) c.appendChild(el("div", null, p.reason));
        var acts = el("div", "cw-acts");
        [["Save", "save"], ["Similar", "similar"], ["Buy", "buy"]].forEach(function (a) {
          a.push({product_id: p.product_id});          // the button acts on this card's product
          var b = el("button", null, a[0]);
          b.addEventListener("click", function () {
          track(a[1] === "similar" ? "click" : a[1], [p.product_id]);
          action(a[1], a[2]);
        });
          acts.appendChild(b);
        });
        c.appendChild(acts);
        box.appendChild(c);
      });
      if (data.comparison && data.comparison.length) {
        var tbl = el("table", "cw-tbl");
        var cols = Object.keys(data.comparison[0]);
        var tr = el("tr");
        cols.forEach(function (k) { tr.appendChild(el("th", null, k)); });
        tbl.appendChild(tr);
        data.comparison.forEach(function (row) {
          var r = el("tr");
          cols.forEach(function (k) { r.appendChild(el("td", null, row[k])); });
          tbl.appendChild(r);
        });
        box.appendChild(tbl);
      }
      var url = safeUrl(data.link);
      if (url) {
        var a = el("a", null, "Open the product page");
        a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer";
        box.appendChild(el("div")).appendChild(a);
      }
      return box;
    }

    function show(entry, fresh) {
      if (fresh && entry.bot && entry.bot.products) track("shown", entry.bot.products.map(function (p) { return p.product_id; }));
      msgs.appendChild(entry.me ? el("div", "cw-me", entry.me) : botMessage(entry.bot));
      msgs.scrollTop = msgs.scrollHeight;
    }

    function post(path, body) {
      send.disabled = true;
      var wait = el("div", "cw-meta", "…");
      msgs.appendChild(wait);
      var early = null;                      // products shown before the answer arrives (/chat/stream)
      var request = path === "/chat" ?
        streamJson("/chat/stream", body, function (e) {
          if (e.type !== "products" || early) return;      // one early list per message
          early = botMessage({text: "Here is what I found; writing a short summary…", products: e.products});
          msgs.insertBefore(early, wait);
          msgs.scrollTop = msgs.scrollHeight;
        }) :
        plainJson(path, body);
      return request
        .then(function (data) {
          if (early) early.remove();
          state = data.state;
          var entry = {bot: {text: data.text, products: data.products, comparison: data.comparison, link: data.link}};
          log.push(entry); persist(); show(entry, true);
          var url = safeUrl(data.link);
          if (url && path === "/chat/action") window.open(url, "_blank", "noopener");
        })
        .catch(function (e) {
          if (early) early.remove();
          show({bot: {text: "Sorry, something went wrong: " + e.message}});
        })
        .finally(function () { wait.remove(); send.disabled = false; });
    }

    function action(name, args) {
      post("/chat/action", {action: name, args: args, store: store, state: state, n: n});
    }

    cform.addEventListener("submit", function (ev) {
      ev.preventDefault();
      var text = cinput.value.trim();
      if (!text) return;
      cinput.value = "";
      var entry = {me: text};
      log.push(entry); persist(); show(entry);
      post("/chat", {message: text, store: store, user_id: user, state: state, n: n});
    });
    bubble.addEventListener("click", function () {
      panel.classList.toggle("open");
      if (panel.classList.contains("open")) cinput.focus();
    });
    close.addEventListener("click", function () { panel.classList.remove("open"); });

    if (log.length) log.forEach(show);
    else show({bot: {text: "Hi! Tell me what you're shopping for, and I'll find it, compare options and save favourites."}});
    return;
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


  function render(data) {
    if (data.type === "answer") track("shown", (data.products || []).map(function (p) { return p.product_id; }));
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
      card.appendChild(el("div", "cw-meta", [p.brand || "", p.category || "", price(p.price), stars(p)]
        .filter(Boolean).join(" · ")));
      if (p.reason) card.appendChild(el("div", null, p.reason));
      card.appendChild(el("div", "cw-why", "why: relevance " + (p.s_sem || 0).toFixed(2) + ", personal " +
        (p.s_pers || 0).toFixed(2) + ", popularity " + (p.s_pop || 0).toFixed(2) +
        (p.avg_rating == null ? "" : ", rating " + (p.s_rat || 0).toFixed(2))));
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
    streamJson("/search/stream", {query: q, store: store, user_id: user, n: n}, function (e) {
      if (e.type === "products") render({parsed: e.parsed, products: e.products, answer: "Writing a short summary…"});
    }).then(render).catch(function (e) {
      out.textContent = "";
      out.appendChild(el("div", "cw-error", "Search failed: " + e.message));
    }).finally(function () { button.disabled = false; });
  });
})();

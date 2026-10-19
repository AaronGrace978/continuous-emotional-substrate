const dims = ["valence", "arousal", "dominance", "curiosity", "solitude"];
const scroll = document.getElementById("scroll");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const veil = document.getElementById("veil");
const veilKicker = document.getElementById("veilKicker");
const veilDetail = document.getElementById("veilDetail");
const dreamText = document.getElementById("dreamText");

let api = location.protocol === "http:" || location.protocol === "https:" ? location.origin : "";
let generating = false;
let modelReady = false;
let modelStatus = "booting";
let forceSync = true;
let lastDream = "";
let memKey = "";
let misses = 0;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function showVeil(kicker, detail) {
  veil.classList.remove("hidden");
  if (kicker) veilKicker.textContent = kicker;
  if (detail) veilDetail.textContent = detail;
}

function paintAffect(affect, mood) {
  if (!affect) return;
  const valence = Number(affect.valence) || 0;
  const v01 = Math.min(1, Math.max(0, (valence + 1) / 2));
  document.documentElement.style.setProperty("--v01", v01.toFixed(4));
  for (const key of ["arousal", "curiosity", "solitude", "dominance"]) {
    const value = Math.min(1, Math.max(0, Number(affect[key]) || 0));
    document.documentElement.style.setProperty("--" + key, value.toFixed(4));
  }
  const gauge = (valence >= 0 ? "+" : "") + valence.toFixed(2);
  document.getElementById("gauge").textContent = gauge;
  if (mood) document.getElementById("mood").textContent = mood;
  for (const key of dims) {
    const li = document.querySelector(`.meters li[data-k="${key}"]`);
    const value = Number(affect[key]) || 0;
    const level = key === "valence" ? (value + 1) / 2 : value;
    li.style.setProperty("--level", String(Math.min(1, Math.max(0, level))));
    li.querySelector("b").textContent = key === "valence"
      ? (value >= 0 ? "+" : "") + value.toFixed(2)
      : value.toFixed(2);
  }
}

function paintModel(model) {
  if (!model) return;
  modelStatus = model.status || "booting";
  modelReady = modelStatus === "ready";
  const el = document.getElementById("modelStatus");
  el.textContent = model.detail || modelStatus;
  el.classList.toggle("bad", modelStatus === "error");
  const orb = document.getElementById("orb");
  orb.classList.toggle("waking", modelStatus !== "ready");
  orb.classList.toggle("failed", modelStatus === "error");
  if (!generating) {
    input.disabled = !modelReady;
    sendBtn.disabled = !modelReady;
    if (modelReady) input.placeholder = "Say something. I have been here the whole time.";
    else if (modelStatus === "error") input.placeholder = "The voice did not wake.";
    else input.placeholder = "Weights are still settling…";
  }
  if (model.rank) {
    document.getElementById("plateAdapter").textContent = `PEMS · r${model.rank} α${model.alpha}`;
  }
}

function renderMemories(list) {
  const host = document.getElementById("memories");
  const key = (list || []).map((m) => m.id + "|" + m.content).join("||");
  if (key !== memKey) {
    memKey = key;
    host.replaceChildren();
    if (!list || !list.length) {
      const p = document.createElement("p");
      p.className = "empty-mem";
      p.textContent = "No enclaves yet. They form when something matters.";
      host.appendChild(p);
    } else {
      for (const mem of list) {
        const el = document.createElement("article");
        el.className = "slip";
        const bar = document.createElement("i");
        const body = document.createElement("p");
        body.textContent = mem.content;
        const meta = document.createElement("span");
        const tags = (mem.tags || []).slice(0, 3).join(" · ");
        meta.textContent = tags || "memory";
        el.append(bar, body, meta);
        host.appendChild(el);
      }
    }
  }
  if (!list) return;
  const slips = host.querySelectorAll(".slip");
  list.forEach((mem, index) => {
    const el = slips[index];
    if (!el) return;
    const level = Math.max(0.08, Math.min(1, Number(mem.resonance) / 1.6));
    el.style.setProperty("--r", String(level));
  });
}

function renderTranscript(list) {
  scroll.querySelectorAll(".line").forEach((node) => node.remove());
  const quiet = document.getElementById("quiet");
  quiet.hidden = list.length > 0;
  for (const message of list) addLine(message.role, message.content);
  scroll.scrollTop = scroll.scrollHeight;
}

function addLine(role, text) {
  const art = document.createElement("article");
  const mind = role !== "user";
  art.className = "line " + (mind ? "line-mind" : "line-user");
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = mind ? "ChronoSoul" : "You";
  const p = document.createElement("p");
  p.textContent = text || "";
  art.append(who, p);
  scroll.appendChild(art);
  document.getElementById("quiet").hidden = true;
  return p;
}

function applyState(state) {
  paintAffect(state.affect, state.mood);
  paintModel(state.model);
  renderMemories(state.memories || []);
  document.getElementById("plateTick").textContent = "tick " + Number(state.tick || 0).toLocaleString();
  document.getElementById("plateMem").textContent = Number(state.memory_count || 0) + " enclaves";
  document.getElementById("since").textContent = state.since || "";
  const dream = state.daydream || "";
  if (dream && dream !== lastDream) {
    lastDream = dream;
    dreamText.style.opacity = "0";
    setTimeout(() => {
      dreamText.textContent = dream;
      dreamText.style.opacity = "1";
    }, 180);
  }
  if (forceSync && !generating) {
    renderTranscript(state.transcript || []);
    forceSync = false;
  }
}

async function pollOnce() {
  const res = await fetch(api + "/state", { cache: "no-store" });
  if (!res.ok) throw new Error("status " + res.status);
  const data = await res.json();
  misses = 0;
  veil.classList.add("hidden");
  applyState(data);
}

async function loop() {
  for (;;) {
    if (api) {
      try {
        await pollOnce();
      } catch {
        misses += 1;
        if (misses >= 4) {
          showVeil("Interrupted", "Lost the substrate process. It may still be waking.");
        }
      }
    }
    await sleep(800);
  }
}

async function speak() {
  const text = input.value.trim();
  if (!text || generating) return;
  if (!api) {
    showVeil("Kindling", "The runtime is not open yet.");
    return;
  }
  if (!modelReady) return;
  generating = true;
  input.value = "";
  input.style.height = "auto";
  input.disabled = true;
  sendBtn.disabled = false;
  sendBtn.dataset.mode = "rest";
  sendBtn.textContent = "Rest";
  addLine("user", text);
  const reply = addLine("assistant", "");
  const line = reply.closest(".line");
  line.classList.add("pending");
  scroll.scrollTop = scroll.scrollHeight;
  let finished = false;
  try {
    const res = await fetch(api + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok || !res.body) throw new Error("The chamber refused the message (" + res.status + ").");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const row = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!row) continue;
        let evt;
        try {
          evt = JSON.parse(row);
        } catch {
          continue;
        }
        if (evt.type === "token") {
          reply.textContent += evt.text || "";
          scroll.scrollTop = scroll.scrollHeight;
        } else if (evt.type === "replace") {
          reply.textContent = evt.text || "";
        } else if (evt.type === "done") {
          finished = true;
          reply.textContent = evt.reply || reply.textContent;
          paintAffect(evt.affect, evt.mood);
        } else if (evt.type === "error" && !finished) {
          line.classList.add("notice");
          reply.textContent = evt.message || "The thought broke.";
        }
      }
    }
    if (!finished && !reply.textContent) {
      line.classList.add("notice");
      reply.textContent = "The stream closed before a thought arrived.";
    }
  } catch (err) {
    line.classList.add("notice");
    if (!reply.textContent) reply.textContent = err.message || "The stream closed.";
  } finally {
    line.classList.remove("pending");
    generating = false;
    sendBtn.dataset.mode = "";
    sendBtn.textContent = "Speak";
    input.disabled = !modelReady;
    sendBtn.disabled = !modelReady;
    input.focus();
  }
}

document.getElementById("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!generating) speak();
});

sendBtn.addEventListener("click", () => {
  if (generating) {
    if (api) fetch(api + "/stop", { method: "POST" });
    return;
  }
  speak();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!generating) speak();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(140, input.scrollHeight) + "px";
});

document.getElementById("journal").addEventListener("click", async () => {
  if (!api || generating) return;
  await fetch(api + "/journal/close", { method: "POST" });
  forceSync = true;
});

document.getElementById("modelStatus").addEventListener("click", () => {
  if (modelStatus === "error" && api) fetch(api + "/reload", { method: "POST" });
});

document.getElementById("retry").addEventListener("click", () => {
  document.getElementById("retry").hidden = true;
  if (window.chronosoul) window.chronosoul.retry();
});

document.querySelectorAll("[data-act]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const act = btn.dataset.act;
    if (window.chronosoul) window.chronosoul.win[act]();
  });
});

if (!window.chronosoul) document.getElementById("winbtns").hidden = true;

if (window.chronosoul) {
  window.chronosoul.onBoot((data) => {
    if (!data) return;
    if (data.phase === "error") {
      document.getElementById("retry").hidden = false;
      showVeil("Interrupted", data.detail || "The mind process failed.");
      return;
    }
    if (location.protocol === "file:" && data.detail) {
      const kicker = data.phase === "packages" ? "Installing the voice" : "Kindling";
      showVeil(kicker, data.detail);
    }
  });
}

document.querySelector(".topbar").addEventListener("dblclick", (event) => {
  if (event.target.closest("button")) return;
  if (window.chronosoul) window.chronosoul.win.max();
});

loop();

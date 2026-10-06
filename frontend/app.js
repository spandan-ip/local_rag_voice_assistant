// Empty string = same origin as this page. Works because api.py now
// serves this frontend itself (python api.py, then open http://127.0.0.1:8000).
// If you're running the frontend on a separate dev server instead, set
// this back to "http://127.0.0.1:8000".
const API_BASE = "";

const el = (id) => document.getElementById(id);
const banner = el("banner");
const emptyState = el("empty-state");
const transcript = el("transcript");
const orb = el("orb");
const miniOrb = el("mini-orb");
const composerForm = el("composer-form");
const questionInput = el("question-input");
const micBtn = el("mic-btn");
const sendBtn = el("send-btn");
const stopBtn = el("stop-btn");
const sessionList = el("session-list");
const sourceList = el("source-list");
const voiceModeToggle = el("voice-mode-toggle");
const themeToggle = el("theme-toggle");
const swatches = el("swatches");

let sessionId = null;
let abortController = null;
let generating = false;
let mediaRecorder = null;
let audioChunks = [];
let timerInterval = null;
let voiceModeOn = localStorage.getItem("rag_voice_mode") === "on";

// --- Banner ---

function showBanner(text, ms = 5000) {
  banner.textContent = text;
  banner.classList.remove("hidden");
  if (ms) setTimeout(() => banner.classList.add("hidden"), ms);
}

// --- Orb state ---

function setOrbState(state) {
  for (const o of [orb, miniOrb]) {
    o.classList.remove("listening", "thinking");
    if (state) o.classList.add(state);
  }
}

// --- Theme (light/dark + 5 accent colors) ---

function applyTheme(mode, accent) {
  document.documentElement.setAttribute("data-theme", mode);
  document.documentElement.setAttribute("data-accent", accent);
  themeToggle.textContent = mode === "dark" ? "🌙" : "☀️";
  for (const sw of swatches.querySelectorAll(".swatch")) {
    sw.classList.toggle("active", sw.dataset.accent === accent);
  }
}

function initTheme() {
  const mode = localStorage.getItem("rag_theme_mode") || "dark";
  const accent = localStorage.getItem("rag_theme_accent") || "ember";
  applyTheme(mode, accent);
}

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  localStorage.setItem("rag_theme_mode", next);
  applyTheme(next, document.documentElement.getAttribute("data-accent"));
});

swatches.addEventListener("click", (e) => {
  const btn = e.target.closest(".swatch");
  if (!btn) return;
  localStorage.setItem("rag_theme_accent", btn.dataset.accent);
  applyTheme(document.documentElement.getAttribute("data-theme"), btn.dataset.accent);
});

// --- Voice mode toggle ---

function applyVoiceModeUI() {
  voiceModeToggle.setAttribute("aria-pressed", voiceModeOn ? "true" : "false");
}

voiceModeToggle.addEventListener("click", () => {
  voiceModeOn = !voiceModeOn;
  localStorage.setItem("rag_voice_mode", voiceModeOn ? "on" : "off");
  applyVoiceModeUI();
});

// --- Backend calls ---

async function apiGet(path) {
  const resp = await fetch(`${API_BASE}${path}`);
  return resp.json();
}

async function apiPost(path, body) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return resp.json();
}

async function apiPatch(path, body) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return resp.json();
}

async function checkBackend() {
  try {
    const data = await apiGet("/health");
    return data.status === "ok";
  } catch {
    return false;
  }
}

// --- Session handling (reload-safe via localStorage) ---

async function loadOrCreateSession() {
  const saved = localStorage.getItem("rag_session_id");
  if (saved) {
    sessionId = saved;
    const data = await apiGet(`/sessions/${sessionId}`);
    renderTranscript(data.messages || []);
  } else {
    const data = await apiPost("/sessions");
    sessionId = data.session_id;
    localStorage.setItem("rag_session_id", sessionId);
  }
  await refreshSessionList();
}

async function startNewChat() {
  const data = await apiPost("/sessions");
  sessionId = data.session_id;
  localStorage.setItem("rag_session_id", sessionId);
  transcript.innerHTML = "";
  transcript.classList.add("hidden");
  emptyState.classList.remove("hidden");
  await refreshSessionList();
}

async function switchSession(id) {
  sessionId = id;
  localStorage.setItem("rag_session_id", id);
  const data = await apiGet(`/sessions/${id}`);
  renderTranscript(data.messages || []);
  await refreshSessionList();
}

async function renameSessionPrompt(s, textEl) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "session-rename-input";
  input.value = s.title || s.session_id;

  textEl.replaceWith(input);
  input.focus();
  input.select();

  const commit = async () => {
    const newTitle = input.value.trim();
    if (newTitle) {
      await apiPatch(`/sessions/${s.session_id}`, { title: newTitle });
    }
    await refreshSessionList();
  };

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") input.blur();
    if (e.key === "Escape") { input.value = s.title || ""; input.blur(); }
  });
  input.addEventListener("blur", commit);
}

async function refreshSessionList() {
  const data = await apiGet("/sessions");
  const sessions = data.sessions || [];
  sessionList.innerHTML = "";
  for (const s of sessions) {
    const li = document.createElement("li");
    li.className = "session-item";
    if (s.session_id === sessionId) li.classList.add("active");

    const textWrap = document.createElement("div");
    textWrap.className = "session-text";

    const titleEl = document.createElement("div");
    titleEl.className = "session-title";
    titleEl.textContent = s.title || `Chat ${s.session_id}`;

    const dateEl = document.createElement("div");
    dateEl.className = "session-date";
    dateEl.textContent = s.created_at.slice(0, 16).replace("T", " ");

    textWrap.appendChild(titleEl);
    textWrap.appendChild(dateEl);

    const renameBtn = document.createElement("button");
    renameBtn.className = "session-rename-btn";
    renameBtn.textContent = "✎";
    renameBtn.title = "Rename chat";
    renameBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      renameSessionPrompt(s, titleEl);
    });

    li.appendChild(textWrap);
    li.appendChild(renameBtn);
    li.addEventListener("click", () => switchSession(s.session_id));
    sessionList.appendChild(li);
  }
}

// --- Sources sidebar (what the assistant is currently drawing from) ---

function sourceDisplayName(entry) {
  if (entry.source_type === "web") return entry.source;
  const parts = entry.source.split(/[\\/]/);
  return parts[parts.length - 1];
}

async function refreshSourceList() {
  const data = await apiGet("/sources");
  const sources = data.sources || [];
  sourceList.innerHTML = "";

  if (sources.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No documents added yet.";
    sourceList.appendChild(li);
    return;
  }

  for (const s of sources) {
    const li = document.createElement("li");
    const icon = document.createElement("span");
    icon.className = "source-icon";
    icon.textContent = s.source_type === "web" ? "🌐" : "📄";
    const label = document.createElement("span");
    label.textContent = sourceDisplayName(s);
    li.appendChild(icon);
    li.appendChild(label);
    sourceList.appendChild(li);
  }
}

// --- Rendering ---

function renderTranscript(messages) {
  transcript.innerHTML = "";
  if (messages.length === 0) {
    transcript.classList.add("hidden");
    emptyState.classList.remove("hidden");
    return;
  }
  emptyState.classList.add("hidden");
  transcript.classList.remove("hidden");
  for (const m of messages) {
    appendMessageBubble(m.role, m.content, m.sources || []);
  }
  transcript.scrollTop = transcript.scrollHeight;
}

function appendMessageBubble(role, content, sources) {
  emptyState.classList.add("hidden");
  transcript.classList.remove("hidden");

  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;

  const textEl = document.createElement("div");
  textEl.className = "text";
  textEl.textContent = content;
  bubble.appendChild(textEl);

  if (role === "assistant") {
    const footer = document.createElement("div");
    footer.className = "msg-footer";

    const playBtn = document.createElement("button");
    playBtn.className = "play-btn";
    playBtn.textContent = "▶ Play";
    playBtn.addEventListener("click", () => speak(textEl.textContent));
    footer.appendChild(playBtn);

    bubble.appendChild(footer);
  }

  if (role === "assistant" && sources && sources.length > 0) {
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = "Sources";
    details.appendChild(summary);
    const ol = document.createElement("ol");
    for (const s of sources) {
      const li = document.createElement("li");
      li.textContent = s.page != null ? `${s.source}, page ${s.page}` : s.source;
      ol.appendChild(li);
    }
    details.appendChild(ol);
    bubble.appendChild(details);
  }

  transcript.appendChild(bubble);
  transcript.scrollTop = transcript.scrollHeight;
  return bubble;
}

// --- Sending a question (streamed, with working Stop) ---

async function sendQuestion(question) {
  if (generating || !question || !question.trim()) return;

  generating = true;
  questionInput.value = "";
  questionInput.disabled = true;
  micBtn.disabled = true;
  sendBtn.classList.add("hidden");
  stopBtn.classList.remove("hidden");
  miniOrb.classList.remove("hidden");
  setOrbState("thinking");

  appendMessageBubble("user", question, []);

  const assistantBubble = appendMessageBubble("assistant", "", []);
  const textEl = assistantBubble.querySelector(".text");
  const footer = assistantBubble.querySelector(".msg-footer");
  const timerEl = document.createElement("span");
  timerEl.className = "timer";
  footer.prepend(timerEl);

  const startTime = Date.now();
  timerInterval = setInterval(() => {
    timerEl.textContent = `Answering... ${((Date.now() - startTime) / 1000).toFixed(1)}s`;
  }, 100);

  abortController = new AbortController();
  let fullAnswer = "";
  let wasStopped = false;

  try {
    const resp = await fetch(`${API_BASE}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
      signal: abortController.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      fullAnswer += decoder.decode(value, { stream: true });
      textEl.textContent = fullAnswer;
      transcript.scrollTop = transcript.scrollHeight;
    }
  } catch (err) {
    if (err.name === "AbortError") {
      wasStopped = true;
    } else {
      showBanner(`Connection error: ${err.message}`);
    }
  }

  clearInterval(timerInterval);
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  timerEl.textContent = wasStopped ? `Stopped at ${elapsed}s` : `Answered in ${elapsed}s`;

  // Give the backend a brief moment to finish writing the (possibly
  // partial) turn to SQLite, then pull the authoritative version back
  // -- this is also how the (relevance-filtered) sources for this
  // answer get attached, and how the session's auto-title shows up.
  await new Promise((r) => setTimeout(r, wasStopped ? 400 : 100));
  const data = await apiGet(`/sessions/${sessionId}`);
  renderTranscript(data.messages || []);
  await refreshSessionList();

  if (voiceModeOn && fullAnswer.trim()) {
    await speak(fullAnswer);
  }

  generating = false;
  questionInput.disabled = false;
  micBtn.disabled = false;
  sendBtn.classList.remove("hidden");
  stopBtn.classList.add("hidden");
  miniOrb.classList.add("hidden");
  setOrbState(null);
  questionInput.focus();
}

function stopGenerating() {
  if (abortController) abortController.abort();
}

// --- Text-to-speech playback ---

async function speak(text) {
  try {
    const resp = await fetch(`${API_BASE}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) return;
    const blob = await resp.blob();
    const audioEl = el("tts-audio");
    audioEl.src = URL.createObjectURL(blob);
    audioEl.play().catch(() => {});
  } catch {
    // TTS is a nice-to-have; don't interrupt the chat over it.
  }
}

// --- Voice input (record -> transcribe -> ask) ---

async function toggleMic() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    showBanner("Microphone access denied or unavailable.");
    return;
  }

  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);

  mediaRecorder.onstart = () => {
    micBtn.classList.add("recording");
    setOrbState("listening");
  };

  mediaRecorder.onstop = async () => {
    micBtn.classList.remove("recording");
    setOrbState(null);
    stream.getTracks().forEach((t) => t.stop());

    const blob = new Blob(audioChunks, { type: "audio/webm" });
    const formData = new FormData();
    formData.append("file", blob, "question.webm");

    setOrbState("thinking");
    let text = null;
    try {
      const resp = await fetch(`${API_BASE}/transcribe`, { method: "POST", body: formData });
      const data = await resp.json();
      if (data.error) {
        showBanner(data.error);
      } else {
        text = data.text;
      }
    } catch (err) {
      showBanner(`Transcription failed: ${err.message}`);
    }
    setOrbState(null);

    if (text) {
      showBanner(`Heard: "${text}"`, 3000);
      sendQuestion(text);
    }
  };

  mediaRecorder.start();
}

// --- Document / URL ingestion ---

el("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = el("file-input");
  const status = el("upload-status");
  if (!fileInput.files.length) return;

  status.textContent = "Ingesting and rebuilding index... this can take a while.";
  status.className = "status-line";

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const resp = await fetch(`${API_BASE}/ingest/upload`, { method: "POST", body: formData });
    const data = await resp.json();
    status.textContent = data.error || data.message;
    status.className = "status-line " + (data.error ? "error" : "success");
    if (!data.error) await refreshSourceList();
  } catch (err) {
    status.textContent = `Failed: ${err.message}`;
    status.className = "status-line error";
  }
  fileInput.value = "";
});

el("url-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const urlInput = el("url-input");
  const status = el("url-status");
  if (!urlInput.value.trim()) return;

  status.textContent = "Fetching page and rebuilding index...";
  status.className = "status-line";

  try {
    const data = await apiPost("/ingest/web", { url: urlInput.value.trim() });
    status.textContent = data.error || data.message;
    status.className = "status-line " + (data.error ? "error" : "success");
    if (!data.error) await refreshSourceList();
  } catch (err) {
    status.textContent = `Failed: ${err.message}`;
    status.className = "status-line error";
  }
  urlInput.value = "";
});

// --- Wire up static controls ---

composerForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendQuestion(questionInput.value);
});

micBtn.addEventListener("click", toggleMic);
stopBtn.addEventListener("click", stopGenerating);
el("new-chat-btn").addEventListener("click", startNewChat);

// --- Init ---

(async function init() {
  initTheme();
  applyVoiceModeUI();

  const ok = await checkBackend();
  if (!ok) {
    showBanner(`Cannot reach the backend at ${API_BASE}. Start it with: python api.py`, 0);
    questionInput.disabled = true;
    sendBtn.disabled = true;
    micBtn.disabled = true;
    return;
  }
  await loadOrCreateSession();
  await refreshSourceList();
})();

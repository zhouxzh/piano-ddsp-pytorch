"use strict";

const elements = Object.fromEntries(
  [
    "statusDot", "statusText", "startButton", "stopButton", "panicButton",
    "recordButton", "downloadButton", "modelName", "streamContract",
    "pianoModel", "midiDevice", "midiButton", "velocityCurve", "velocity",
    "velocityValue", "velocityMeter", "velocityRaw", "velocitySent", "dynamicMark",
    "serverGain", "serverGainValue", "outputGain", "outputGainValue",
    "sustainButton", "scoreSelect", "scoreMeta", "scorePlayButton",
    "scoreStopButton", "loopButton", "tempoScale", "scoreProgress", "scoreTime",
    "octaveDown", "octaveUp", "keyboardRange", "keyboardScroll", "keyboard",
    "sequenceMetric", "renderMetric", "rtfMetric", "latencyMetric", "notesMetric",
    "stealMetric", "lateMetric", "clipMetric", "bufferMetric", "activeNotes",
    "midiStatus", "eventLog", "clearLogButton",
  ].map((id) => [id, document.getElementById(id)]),
);

const PIANO_MIN = 21;
const PIANO_MAX = 108;
const KEYBOARD_NOTE_COUNT = 24;
const KEYBOARD_MIN_START = 29;
const KEYBOARD_MAX_START = 77;
const KEY_CODES = [
  "Tab", "Digit1", "KeyQ", "Digit2", "KeyW", "Digit3", "KeyE", "KeyR",
  "Digit5", "KeyT", "Digit6", "KeyY", "KeyU", "Digit8", "KeyI", "Digit9",
  "KeyO", "Digit0", "KeyP", "BracketLeft", "Equal", "BracketRight",
  "Backspace", "Backslash",
];
const KEY_LABELS = [
  "Tab", "1", "Q", "2", "W", "3", "E", "R", "5", "T", "6", "Y",
  "U", "8", "I", "9", "O", "0", "P", "[", "=", "]", "⌫", "\\",
];
const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const BLACK_PITCH_CLASSES = new Set([1, 3, 6, 8, 10]);

let socket = null;
let player = null;
let streaming = false;
let finishing = false;
let pingTimer = null;
let midiAccess = null;
let attachedMidiId = "";
let recording = false;
let recordedChunks = [];
let recordedSampleRate = 16000;
let keyboardStart = 53;
let serverSustain = false;
let scoreState = "stopped";
let scorePosition = 0;
let scoreDuration = 0;
let scoreSeeking = false;
const scoreFiles = new Map();
const activeSources = new Map();
const sustainSources = new Set();
const keyByPitch = new Map();
const pressedComputerKeys = new Map();

class StreamingResampler {
  constructor(inputRate, outputRate) {
    this.inputRate = inputRate;
    this.outputRate = outputRate;
    this.step = inputRate / outputRate;
    this.buffer = new Float32Array(0);
    this.position = 0;
  }

  process(input) {
    if (this.inputRate === this.outputRate) return input;
    const combined = new Float32Array(this.buffer.length + input.length);
    combined.set(this.buffer);
    combined.set(input, this.buffer.length);
    const count = Math.max(
      0,
      Math.floor((combined.length - 1 - this.position) / this.step) + 1,
    );
    const output = new Float32Array(count);
    let written = 0;
    while (this.position + 1 < combined.length && written < count) {
      const index = Math.floor(this.position);
      const fraction = this.position - index;
      output[written] = combined[index]
        + (combined[index + 1] - combined[index]) * fraction;
      written += 1;
      this.position += this.step;
    }
    const consumed = Math.floor(this.position);
    this.buffer = combined.slice(consumed);
    this.position -= consumed;
    return written === output.length ? output : output.slice(0, written);
  }
}

class PcmPlayer {
  constructor() {
    this.context = null;
    this.node = null;
    this.gain = null;
    this.resamplers = new Map();
    this.fallbackQueue = [];
    this.fallbackOffset = 0;
    this.fallbackSamples = 0;
    this.fallbackPrimed = false;
    this.fallbackStartThreshold = 0;
    this.fallbackMaxStartThreshold = 0;
    this.usingWorklet = false;
    this.lastUnderrunLog = 0;
  }

  async open() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) throw new Error("浏览器不支持 Web Audio");
    this.context = new AudioContext({ latencyHint: "interactive" });
    this.gain = this.context.createGain();
    this.gain.gain.value = Number(elements.outputGain.value);
    try {
      await this.context.audioWorklet.addModule("/pcm-worklet.js");
      this.node = new AudioWorkletNode(this.context, "pcm-queue", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: {
          startThreshold: this.context.sampleRate * 0.08,
          maxStartThreshold: this.context.sampleRate * 0.24,
        },
      });
      this.usingWorklet = true;
      this.node.port.onmessage = (event) => {
        if (event.data.type === "buffer") {
          elements.bufferMetric.textContent = `${event.data.milliseconds.toFixed(0)} ms`;
        } else if (event.data.type === "underrun") {
          const now = performance.now();
          if (now - this.lastUnderrunLog > 2000) {
            const threshold = Number(event.data.restartThresholdMilliseconds || 0);
            logEvent(
              threshold
                ? `浏览器播放缓冲不足，重新缓冲 ${threshold.toFixed(0)} ms`
                : "浏览器播放缓冲不足",
              true,
            );
            this.lastUnderrunLog = now;
          }
        }
      };
      logEvent("Web AudioWorklet 已启动");
    } catch (error) {
      this.node = this.context.createScriptProcessor(1024, 0, 1);
      this.fallbackStartThreshold = this.context.sampleRate * 0.08;
      this.fallbackMaxStartThreshold = this.context.sampleRate * 0.24;
      this.node.onaudioprocess = (event) => {
        this.renderFallback(event.outputBuffer.getChannelData(0));
      };
      logEvent(`使用 Web Audio 兼容模式: ${error.message}`);
    }
    this.node.connect(this.gain);
    this.gain.connect(this.context.destination);
    await this.context.resume();
  }

  enqueue(samples, sourceRate) {
    if (!this.context) return;
    let resampler = this.resamplers.get(sourceRate);
    if (!resampler) {
      resampler = new StreamingResampler(sourceRate, this.context.sampleRate);
      this.resamplers.set(sourceRate, resampler);
    }
    const output = resampler.process(samples);
    if (!output.length) return;
    if (this.usingWorklet) {
      this.node.port.postMessage({ type: "samples", samples: output }, [output.buffer]);
    } else {
      this.fallbackQueue.push(output);
      this.fallbackSamples += output.length;
      elements.bufferMetric.textContent = `${(
        this.fallbackSamples * 1000 / this.context.sampleRate
      ).toFixed(0)} ms`;
    }
  }

  renderFallback(output) {
    output.fill(0);
    if (!this.fallbackPrimed && this.fallbackSamples >= this.fallbackStartThreshold) {
      this.fallbackPrimed = true;
    }
    if (!this.fallbackPrimed) return;
    let target = 0;
    while (target < output.length && this.fallbackQueue.length) {
      const source = this.fallbackQueue[0];
      const count = Math.min(
        source.length - this.fallbackOffset,
        output.length - target,
      );
      output.set(source.subarray(this.fallbackOffset, this.fallbackOffset + count), target);
      target += count;
      this.fallbackOffset += count;
      this.fallbackSamples -= count;
      if (this.fallbackOffset === source.length) {
        this.fallbackQueue.shift();
        this.fallbackOffset = 0;
      }
    }
    if (target < output.length) {
      this.fallbackPrimed = false;
      this.fallbackStartThreshold = Math.min(
        this.fallbackMaxStartThreshold,
        Math.ceil(this.fallbackStartThreshold * 1.5),
      );
    }
  }

  setGain(value) {
    if (this.gain && this.context) {
      this.gain.gain.setTargetAtTime(value, this.context.currentTime, 0.01);
    }
  }

  clear() {
    if (this.usingWorklet && this.node) {
      this.node.port.postMessage({ type: "clear" });
    } else {
      this.fallbackQueue = [];
      this.fallbackOffset = 0;
      this.fallbackSamples = 0;
      this.fallbackPrimed = false;
      if (this.context) {
        this.fallbackStartThreshold = this.context.sampleRate * 0.08;
      }
    }
    this.resamplers.clear();
    elements.bufferMetric.textContent = "0 ms";
  }

  async close() {
    if (!this.context) return;
    const context = this.context;
    this.context = null;
    if (this.usingWorklet && this.node) this.node.port.postMessage({ type: "clear" });
    this.node?.disconnect();
    this.gain?.disconnect();
    await context.close();
  }
}

function setStatus(state, text) {
  elements.statusDot.className = `status-dot ${state}`;
  elements.statusText.textContent = text;
}

function logEvent(message, isError = false) {
  const item = document.createElement("li");
  item.className = isError ? "error" : "";
  item.textContent = `${new Date().toLocaleTimeString()}  ${message}`;
  elements.eventLog.prepend(item);
  while (elements.eventLog.children.length > 40) elements.eventLog.lastChild.remove();
}

function send(payload) {
  if (socket?.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(payload));
  return true;
}

function noteName(pitch) {
  return `${NOTE_NAMES[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function dynamicName(velocity) {
  if (velocity <= 20) return "ppp";
  if (velocity <= 35) return "pp";
  if (velocity <= 50) return "p";
  if (velocity <= 70) return "mp";
  if (velocity <= 90) return "mf";
  if (velocity <= 108) return "f";
  if (velocity <= 120) return "ff";
  return "fff";
}

function mapVelocity(rawVelocity) {
  const raw = Math.min(127, Math.max(1, Math.round(Number(rawVelocity) || 1)));
  const normalized = raw / 127;
  let sent;
  switch (elements.velocityCurve.value) {
    case "soft":
      sent = Math.round(127 * Math.sqrt(normalized));
      break;
    case "firm":
      sent = Math.round(127 * normalized ** 1.5);
      break;
    case "fixed":
      sent = Number(elements.velocity.value);
      break;
    case "balanced":
      sent = Math.round(32 + (raw - 1) * 80 / 126);
      break;
    default:
      sent = raw;
  }
  return Math.min(127, Math.max(1, sent));
}

function updateVelocityReadout(rawVelocity) {
  const raw = Math.min(127, Math.max(1, Math.round(Number(rawVelocity) || 1)));
  const sent = mapVelocity(raw);
  elements.velocityRaw.textContent = raw;
  elements.velocitySent.textContent = sent;
  elements.velocityMeter.value = sent;
  elements.dynamicMark.textContent = dynamicName(sent);
  return sent;
}

function updateScoreProgress(position, duration) {
  const safeDuration = Math.max(0, Number(duration) || 0);
  const safePosition = Math.min(Math.max(0, Number(position) || 0), safeDuration);
  scorePosition = safePosition;
  scoreDuration = safeDuration;
  elements.scoreProgress.max = safeDuration || 1;
  if (!scoreSeeking) elements.scoreProgress.value = safePosition;
  elements.scoreTime.textContent = `${formatDuration(safePosition)} / ${formatDuration(safeDuration)}`;
}

function updateScoreMetadata() {
  const file = scoreFiles.get(elements.scoreSelect.value);
  if (!file) {
    elements.scoreMeta.textContent = "没有可用曲谱";
    updateScoreProgress(0, 0);
    return;
  }
  elements.scoreMeta.textContent = `${file.event_count} events · ${formatDuration(file.duration_seconds)}`;
  if (!["playing", "paused"].includes(scoreState)) {
    updateScoreProgress(0, file.duration_seconds);
  }
}

function populateScoreFiles(files) {
  scoreFiles.clear();
  elements.scoreSelect.replaceChildren();
  for (const file of files) {
    scoreFiles.set(file.id, file);
    const option = document.createElement("option");
    option.value = file.id;
    option.textContent = file.name;
    elements.scoreSelect.append(option);
  }
  if (!files.length) elements.scoreSelect.append(new Option("没有可用曲谱", ""));
  updateScoreMetadata();
  syncScoreControls();
}

function syncScoreControls() {
  const hasScore = scoreFiles.has(elements.scoreSelect.value);
  const active = scoreState === "playing" || scoreState === "paused";
  elements.scoreSelect.disabled = !scoreFiles.size || active;
  elements.scorePlayButton.disabled = !streaming || !hasScore;
  elements.scoreStopButton.disabled = !streaming || !active;
  elements.loopButton.disabled = !hasScore;
  elements.tempoScale.disabled = !hasScore;
  elements.scoreProgress.disabled = !hasScore;
  const isPlaying = scoreState === "playing";
  elements.scorePlayButton.textContent = isPlaying ? "Ⅱ" : "▶";
  elements.scorePlayButton.title = isPlaying ? "暂停" : "播放";
  elements.scorePlayButton.setAttribute("aria-label", isPlaying ? "暂停" : "播放");
}

function playOrPauseScore() {
  if (!streaming || !elements.scoreSelect.value) return;
  if (scoreState === "playing") {
    send({ type: "pause_midi" });
    return;
  }
  player?.clear();
  if (scoreState === "paused") {
    send({ type: "resume_midi" });
    return;
  }
  const position = scoreState === "ended" ? 0 : Number(elements.scoreProgress.value);
  send({
    type: "play_midi",
    midi_id: elements.scoreSelect.value,
    position_seconds: position,
    tempo_scale: Number(elements.tempoScale.value),
    loop: elements.loopButton.getAttribute("aria-pressed") === "true",
  });
}

function stopSelectedScore() {
  if (["playing", "paused"].includes(scoreState)) send({ type: "stop_midi" });
}

function configureScoreTransport() {
  if (!["playing", "paused"].includes(scoreState)) return;
  send({
    type: "set_midi_transport",
    tempo_scale: Number(elements.tempoScale.value),
    loop: elements.loopButton.getAttribute("aria-pressed") === "true",
  });
}

function seekScore() {
  const position = Number(elements.scoreProgress.value);
  updateScoreProgress(position, scoreDuration);
  if (["playing", "paused"].includes(scoreState)) {
    player?.clear();
    send({ type: "seek_midi", position_seconds: position });
  }
}

function createKey(pitch, black, whitePitches) {
  const key = document.createElement("button");
  key.type = "button";
  key.className = `piano-key ${black ? "black" : "white"}`;
  key.dataset.pitch = pitch;
  key.setAttribute("aria-label", noteName(pitch));
  if (black) {
    const whiteBefore = whitePitches.filter((whitePitch) => whitePitch < pitch).length;
    key.style.left = `${whiteBefore * 26}px`;
  }
  const label = document.createElement("span");
  label.className = "key-label";
  const note = document.createElement("strong");
  note.className = "note-name";
  note.textContent = pitch % 12 === 0 || pitch === PIANO_MIN || pitch === PIANO_MAX
    ? noteName(pitch)
    : "";
  const binding = document.createElement("span");
  binding.className = "binding";
  label.append(note, binding);
  key.append(label);
  key.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    key.setPointerCapture(event.pointerId);
    pressNote(pitch, Number(elements.velocity.value), `pointer-${event.pointerId}`);
  });
  const release = (event) => releaseNote(pitch, `pointer-${event.pointerId}`);
  key.addEventListener("pointerup", release);
  key.addEventListener("pointercancel", release);
  keyByPitch.set(pitch, key);
  return key;
}

function buildKeyboard() {
  const whitePitches = [];
  for (let pitch = PIANO_MIN; pitch <= PIANO_MAX; pitch += 1) {
    if (!BLACK_PITCH_CLASSES.has(pitch % 12)) whitePitches.push(pitch);
  }
  for (const pitch of whitePitches) {
    elements.keyboard.append(createKey(pitch, false, whitePitches));
  }
  for (let pitch = PIANO_MIN; pitch <= PIANO_MAX; pitch += 1) {
    if (BLACK_PITCH_CLASSES.has(pitch % 12)) {
      elements.keyboard.append(createKey(pitch, true, whitePitches));
    }
  }
  updateKeyboardMapping(false);
}

function updateKeyboardMapping(scroll = true) {
  keyByPitch.forEach((key, pitch) => {
    const index = pitch - keyboardStart;
    const mapped = index >= 0 && index < KEYBOARD_NOTE_COUNT;
    key.classList.toggle("mapped", mapped);
    key.querySelector(".binding").textContent = mapped ? KEY_LABELS[index] : "";
  });
  elements.keyboardRange.textContent = `${noteName(keyboardStart)}–${noteName(
    keyboardStart + KEYBOARD_NOTE_COUNT - 1,
  )}`;
  elements.octaveDown.disabled = keyboardStart <= KEYBOARD_MIN_START;
  elements.octaveUp.disabled = keyboardStart >= KEYBOARD_MAX_START;
  if (scroll) {
    const key = keyByPitch.get(keyboardStart);
    const target = key.offsetLeft - elements.keyboardScroll.clientWidth / 2 + key.offsetWidth / 2;
    elements.keyboardScroll.scrollTo({ left: Math.max(0, target), behavior: "smooth" });
  }
}

function shiftKeyboard(semitones) {
  releaseSources((source) => source.startsWith("key-"));
  pressedComputerKeys.clear();
  keyboardStart = Math.min(
    KEYBOARD_MAX_START,
    Math.max(KEYBOARD_MIN_START, keyboardStart + semitones),
  );
  updateKeyboardMapping();
}

function controlConsumesKeyboard(event) {
  const target = event.target;
  if (!(target instanceof Element)) return false;
  if (target.closest("textarea, [contenteditable]:not([contenteditable='false'])")) {
    return true;
  }
  const input = target.closest("input");
  if (input && !["range", "button", "checkbox", "radio", "reset", "submit"].includes(input.type)) {
    return true;
  }
  return Boolean(
    target.closest("button, input, select")
      && ["Space", "Enter", "NumpadEnter"].includes(event.code),
  );
}

function pressNote(pitch, rawVelocity, source) {
  if (!streaming || pitch < PIANO_MIN || pitch > PIANO_MAX) return;
  let sources = activeSources.get(pitch);
  if (!sources) {
    sources = new Set();
    activeSources.set(pitch, sources);
  }
  if (sources.has(source)) return;
  const velocity = updateVelocityReadout(rawVelocity);
  if (sources.size === 0) send({ type: "note_on", pitch, velocity });
  sources.add(source);
  keyByPitch.get(pitch)?.classList.add("active");
}

function releaseNote(pitch, source) {
  const sources = activeSources.get(pitch);
  if (!sources?.has(source)) return;
  sources.delete(source);
  if (sources.size === 0) {
    activeSources.delete(pitch);
    if (streaming) send({ type: "note_off", pitch });
    keyByPitch.get(pitch)?.classList.remove("active");
  }
}

function releaseSources(predicate) {
  for (const [pitch, sources] of [...activeSources.entries()]) {
    for (const source of [...sources]) {
      if (predicate(source)) releaseNote(pitch, source);
    }
  }
}

function syncSustainIndicator() {
  elements.sustainButton.setAttribute(
    "aria-pressed",
    String(serverSustain || sustainSources.size > 0),
  );
}

function setSustainSource(source, enabled, transmit = true) {
  const wasEnabled = sustainSources.size > 0;
  if (enabled) sustainSources.add(source);
  else sustainSources.delete(source);
  const isEnabled = sustainSources.size > 0;
  if (transmit && streaming && wasEnabled !== isEnabled) {
    send({ type: "control_change", controller: 64, value: isEnabled ? 127 : 0 });
  }
  syncSustainIndicator();
}

function panic() {
  player?.clear();
  if (streaming) send({ type: "panic" });
  activeSources.clear();
  keyByPitch.forEach((key) => key.classList.remove("active"));
  sustainSources.clear();
  serverSustain = false;
  syncSustainIndicator();
}

function parseWav(buffer) {
  const view = new DataView(buffer);
  const text = (offset, size) => String.fromCharCode(
    ...new Uint8Array(buffer, offset, size),
  );
  if (text(0, 4) !== "RIFF" || text(8, 4) !== "WAVE") {
    throw new Error("收到的音频块不是 WAV");
  }
  let offset = 12;
  let format = null;
  let dataOffset = 0;
  let dataLength = 0;
  while (offset + 8 <= view.byteLength) {
    const id = text(offset, 4);
    const size = view.getUint32(offset + 4, true);
    if (id === "fmt ") {
      format = {
        type: view.getUint16(offset + 8, true),
        channels: view.getUint16(offset + 10, true),
        sampleRate: view.getUint32(offset + 12, true),
        bits: view.getUint16(offset + 22, true),
      };
    } else if (id === "data") {
      dataOffset = offset + 8;
      dataLength = size;
      break;
    }
    offset += 8 + size + (size % 2);
  }
  if (!format || format.type !== 1 || format.channels !== 1
      || format.bits !== 16 || !dataLength) {
    throw new Error("仅支持 mono PCM16 WAV 音频块");
  }
  const pcm = new Int16Array(buffer.slice(dataOffset, dataOffset + dataLength));
  const samples = new Float32Array(pcm.length);
  for (let index = 0; index < pcm.length; index += 1) {
    samples[index] = pcm[index] / 32768;
  }
  return { samples, sampleRate: format.sampleRate };
}

async function startSession() {
  elements.startButton.disabled = true;
  setStatus("loading", "连接中");
  try {
    player = new PcmPlayer();
    await player.open();
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const queryToken = new URLSearchParams(location.search).get("token");
    const query = queryToken ? `?token=${encodeURIComponent(queryToken)}` : "";
    socket = new WebSocket(`${protocol}//${location.host}/ws${query}`);
    socket.binaryType = "arraybuffer";
    socket.onmessage = handleSocketMessage;
    socket.onerror = () => setStatus("error", "网络错误");
    socket.onclose = () => { void finishSession("连接已关闭"); };
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("WebSocket 连接超时")), 8000);
      socket.addEventListener("open", () => {
        clearTimeout(timeout);
        resolve();
      }, { once: true });
      socket.addEventListener("error", () => {
        clearTimeout(timeout);
        reject(new Error("WebSocket 无法连接"));
      }, { once: true });
    });
    logEvent("WebSocket 已连接");
  } catch (error) {
    logEvent(error.message, true);
    socket?.close();
    await finishSession("启动失败");
  }
}

function handleSocketMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    try {
      const wav = parseWav(event.data);
      if (recording) {
        recordedChunks.push(wav.samples.slice());
        recordedSampleRate = wav.sampleRate;
      }
      player?.enqueue(wav.samples, wav.sampleRate);
    } catch (error) {
      logEvent(error.message, true);
    }
    return;
  }
  let message;
  try {
    message = JSON.parse(event.data);
  } catch (_error) {
    logEvent("服务器发送了无效 JSON", true);
    return;
  }
  if (message.type === "hello") {
    elements.modelName.textContent = `${message.release_version} · ${message.model}`;
    populatePianoModels(message.piano_model_years);
    populateScoreFiles(message.midi_files || []);
    send({
      type: "start",
      piano_model: Number(elements.pianoModel.value),
      server_gain: Number(elements.serverGain.value),
    });
  } else if (message.type === "status") {
    handleStatusMessage(message);
  } else if (message.type === "metrics") {
    handleMetricsMessage(message);
  } else if (message.type === "midi_playback") {
    handleMidiPlaybackMessage(message);
  } else if (message.type === "pong") {
    elements.latencyMetric.textContent = `${(Date.now() - message.client_time).toFixed(0)} ms`;
  } else if (message.type === "panic_ack") {
    player?.clear();
  } else if (message.type === "error") {
    setStatus("error", "合成错误");
    logEvent(message.message, true);
  }
}

function handleStatusMessage(message) {
  if (message.state === "loading") {
    streaming = false;
    setStatus("loading", "加载模型");
    elements.pianoModel.disabled = true;
  } else if (message.state === "streaming") {
    streaming = true;
    setStatus("streaming", "实时合成");
    elements.stopButton.disabled = false;
    elements.panicButton.disabled = false;
    elements.recordButton.disabled = false;
    elements.sustainButton.disabled = false;
    elements.pianoModel.disabled = false;
    const contract = message.contract;
    elements.streamContract.textContent = `${contract.host_dsp_profile} · ${
      contract.sample_rate / 1000
    } kHz · ${contract.chunk_ms.toFixed(0)} ms · ${contract.max_polyphony} voices · KeyOff ${
      contract.keyoff_fade_ms.toFixed(0)
    } ms`;
    logEvent(`${contract.release_version} 已启动，钢琴年份 ${contract.piano_year}`);
    clearInterval(pingTimer);
    pingTimer = setInterval(() => send({
      type: "ping",
      client_time: performance.timeOrigin + performance.now(),
    }), 1000);
  } else if (message.state === "stopped") {
    streaming = false;
    setStatus("", "已停止");
  }
  syncScoreControls();
}

function handleMetricsMessage(message) {
  elements.sequenceMetric.textContent = message.sequence;
  elements.renderMetric.textContent = `${message.render_ms.toFixed(1)} ms`;
  elements.rtfMetric.textContent = message.realtime_factor.toFixed(2);
  elements.notesMetric.textContent = message.active_notes.length;
  elements.stealMetric.textContent = message.voice_steals;
  elements.lateMetric.textContent = message.late_blocks;
  elements.clipMetric.textContent = message.clipped_samples;
  serverSustain = Boolean(message.sustain);
  syncSustainIndicator();
  elements.activeNotes.textContent = message.active_notes.length
    ? message.active_notes.map(noteName).join("  ")
    : "无按键";
  const sounding = new Set(message.active_notes);
  keyByPitch.forEach((key, pitch) => {
    key.classList.toggle("active", sounding.has(pitch) || activeSources.has(pitch));
  });
  if (message.midi_playback) {
    scoreState = message.midi_playback.state;
    if (!scoreSeeking) {
      updateScoreProgress(
        message.midi_playback.position_seconds,
        message.midi_playback.duration_seconds,
      );
    }
    elements.tempoScale.value = String(message.midi_playback.tempo_scale);
    elements.loopButton.setAttribute("aria-pressed", String(message.midi_playback.loop));
    syncScoreControls();
  }
}

function handleMidiPlaybackMessage(message) {
  const previousState = scoreState;
  scoreState = message.state;
  const file = scoreFiles.get(message.midi_id);
  if (file) elements.scoreSelect.value = message.midi_id;
  elements.tempoScale.value = String(message.tempo_scale ?? elements.tempoScale.value);
  elements.loopButton.setAttribute("aria-pressed", String(Boolean(message.loop)));
  updateScoreProgress(
    message.position_seconds ?? 0,
    message.duration_seconds ?? file?.duration_seconds ?? 0,
  );
  if (message.state === "playing") {
    if (previousState !== "playing" && !message.reason) {
      logEvent(`播放曲谱: ${file?.name || message.midi_id}`);
    } else if (message.reason === "loop") {
      logEvent(`循环: ${file?.name || message.midi_id}`);
    }
  } else if (message.state === "paused") {
    player?.clear();
    if (message.reason !== "seek") logEvent("曲谱已暂停");
  } else if (message.state === "ended") {
    logEvent(`曲谱播放完成: ${file?.name || message.midi_id}`);
  } else if (message.state === "stopped") {
    player?.clear();
    updateScoreProgress(0, message.duration_seconds ?? file?.duration_seconds ?? 0);
    logEvent(`曲谱播放已停止: ${file?.name || message.midi_id}`);
  }
  updateScoreMetadata();
  syncScoreControls();
}

function populatePianoModels(years) {
  const previous = Number(elements.pianoModel.value);
  elements.pianoModel.replaceChildren();
  years.forEach((year, index) => {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = `音色 ${index} · MAESTRO ${year}`;
    elements.pianoModel.append(option);
  });
  if (!years.length) elements.pianoModel.append(new Option("默认音色", "0"));
  elements.pianoModel.value = years[previous]
    ? String(previous)
    : String(Math.max(0, years.length - 1));
}

function changePianoTimbre() {
  if (!streaming) return;
  streaming = false;
  elements.pianoModel.disabled = true;
  setStatus("loading", "切换音色");
  player?.clear();
  logEvent(`切换钢琴音色：${
    elements.pianoModel.selectedOptions[0]?.textContent || elements.pianoModel.value
  }`);
  send({
    type: "start",
    piano_model: Number(elements.pianoModel.value),
    server_gain: Number(elements.serverGain.value),
  });
  syncScoreControls();
}

async function stopSession() {
  if (socket?.readyState === WebSocket.OPEN) send({ type: "stop" });
  socket?.close(1000, "user stopped");
  await finishSession("已停止");
}

async function finishSession(status) {
  if (finishing) return;
  finishing = true;
  try {
    streaming = false;
    clearInterval(pingTimer);
    pingTimer = null;
    activeSources.clear();
    pressedComputerKeys.clear();
    sustainSources.clear();
    serverSustain = false;
    keyByPitch.forEach((key) => key.classList.remove("active"));
    syncSustainIndicator();
    socket = null;
    const activePlayer = player;
    player = null;
    if (activePlayer) await activePlayer.close();
    setStatus("", status);
    elements.startButton.disabled = false;
    elements.stopButton.disabled = true;
    elements.panicButton.disabled = true;
    elements.recordButton.disabled = true;
    elements.sustainButton.disabled = true;
    scoreState = "stopped";
    elements.pianoModel.disabled = false;
    if (recording) toggleRecording();
    syncScoreControls();
  } finally {
    finishing = false;
  }
}

async function enableMidi() {
  if (!navigator.requestMIDIAccess) {
    elements.midiStatus.textContent = "Web MIDI 不可用";
    logEvent("Web MIDI 需要安全浏览器上下文", true);
    return;
  }
  try {
    midiAccess = await navigator.requestMIDIAccess({ sysex: false });
    midiAccess.onstatechange = populateMidiDevices;
    populateMidiDevices();
    elements.midiDevice.disabled = false;
  } catch (error) {
    logEvent(`MIDI 授权失败: ${error.message}`, true);
  }
}

function populateMidiDevices() {
  if (!midiAccess) return;
  const selected = elements.midiDevice.value || attachedMidiId;
  elements.midiDevice.replaceChildren(new Option("电脑键盘 / 屏幕键盘", ""));
  const connected = [...midiAccess.inputs.values()].filter(
    (input) => input.state === "connected",
  );
  for (const input of connected) {
    elements.midiDevice.append(new Option(input.name || "MIDI Input", input.id));
  }
  if (connected.some((input) => input.id === selected)) {
    elements.midiDevice.value = selected;
  } else if (connected.length === 1) {
    elements.midiDevice.value = connected[0].id;
  }
  attachMidiDevice(true);
}

function attachMidiDevice(force = false) {
  if (!midiAccess) return;
  const nextId = elements.midiDevice.value;
  if (!force && nextId === attachedMidiId) return;
  for (const input of midiAccess.inputs.values()) input.onmidimessage = null;
  releaseSources((source) => source.startsWith("midi-"));
  setSustainSource("midi", false);
  attachedMidiId = nextId;
  const input = midiAccess.inputs.get(nextId);
  if (input?.state === "connected") {
    input.onmidimessage = handleMidiMessage;
    elements.midiStatus.textContent = input.name || "MIDI 输入";
    logEvent(`MIDI 输入: ${input.name || input.id}`);
  } else {
    attachedMidiId = "";
    elements.midiStatus.textContent = "电脑键盘 / 屏幕键盘";
  }
}

function handleMidiMessage(event) {
  const [status, data1, data2] = event.data;
  const command = status & 0xf0;
  const channel = status & 0x0f;
  const source = `midi-${channel}-${data1}`;
  if (command === 0x90 && data2 > 0) {
    pressNote(data1, data2, source);
  } else if (command === 0x80 || (command === 0x90 && data2 === 0)) {
    releaseNote(data1, source);
  } else if (command === 0xb0 && data1 >= 64 && data1 <= 67) {
    send({ type: "control_change", controller: data1, value: data2 });
    if (data1 === 64) setSustainSource("midi", data2 >= 64, false);
  } else if (command === 0xb0 && (data1 === 120 || data1 === 123)) {
    releaseSources((activeSource) => activeSource.startsWith("midi-"));
    setSustainSource("midi", false);
  }
}

function toggleRecording() {
  recording = !recording;
  elements.recordButton.textContent = recording ? "停止录音" : "录音";
  elements.recordButton.classList.toggle("recording", recording);
  if (recording) {
    recordedChunks = [];
    elements.downloadButton.disabled = true;
    logEvent("开始浏览器端录音");
  } else {
    elements.downloadButton.disabled = recordedChunks.length === 0;
    const count = recordedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    logEvent(`录音结束，共 ${count} 个采样`);
  }
}

function downloadRecording() {
  const length = recordedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  if (!length) return;
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, value) => [...value].forEach((character, index) => {
    view.setUint8(offset + index, character.charCodeAt(0));
  });
  writeText(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  writeText(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, recordedSampleRate, true);
  view.setUint32(28, recordedSampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, length * 2, true);
  let offset = 44;
  for (const chunk of recordedChunks) {
    for (const sample of chunk) {
      view.setInt16(
        offset,
        Math.round(Math.max(-1, Math.min(1, sample)) * 32767),
        true,
      );
      offset += 2;
    }
  }
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
  link.download = `onnx-midi-${new Date().toISOString().replaceAll(":", "-")}.wav`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

document.addEventListener("keydown", (event) => {
  if (controlConsumesKeyboard(event)) return;
  if (event.code === "Space") {
    if (!event.repeat) {
      pressedComputerKeys.set(event.code, null);
      setSustainSource("keyboard", true);
    }
    event.preventDefault();
    return;
  }
  const index = KEY_CODES.indexOf(event.code);
  if (index < 0) return;
  event.preventDefault();
  if (!event.repeat) {
    const pitch = keyboardStart + index;
    pressedComputerKeys.set(event.code, pitch);
    pressNote(
      pitch,
      Number(elements.velocity.value),
      `key-${event.code}`,
    );
  }
});

document.addEventListener("keyup", (event) => {
  if (event.code === "Space") {
    if (!pressedComputerKeys.has(event.code)) return;
    pressedComputerKeys.delete(event.code);
    setSustainSource("keyboard", false);
    event.preventDefault();
    return;
  }
  const pitch = pressedComputerKeys.get(event.code);
  if (pitch === undefined) return;
  pressedComputerKeys.delete(event.code);
  event.preventDefault();
  releaseNote(pitch, `key-${event.code}`);
});

window.addEventListener("blur", () => {
  releaseSources((source) => source.startsWith("key-") || source.startsWith("pointer-"));
  pressedComputerKeys.clear();
  setSustainSource("keyboard", false);
});

elements.startButton.addEventListener("click", startSession);
elements.stopButton.addEventListener("click", stopSession);
elements.panicButton.addEventListener("click", panic);
elements.sustainButton.addEventListener("click", () => {
  setSustainSource("button", !sustainSources.has("button"));
});
elements.midiButton.addEventListener("click", enableMidi);
elements.midiDevice.addEventListener("change", () => attachMidiDevice());
elements.pianoModel.addEventListener("change", changePianoTimbre);
elements.scoreSelect.addEventListener("change", () => {
  scoreState = "stopped";
  updateScoreMetadata();
  syncScoreControls();
});
elements.scorePlayButton.addEventListener("click", playOrPauseScore);
elements.scoreStopButton.addEventListener("click", stopSelectedScore);
elements.loopButton.addEventListener("click", () => {
  const enabled = elements.loopButton.getAttribute("aria-pressed") !== "true";
  elements.loopButton.setAttribute("aria-pressed", String(enabled));
  configureScoreTransport();
});
elements.tempoScale.addEventListener("change", configureScoreTransport);
elements.scoreProgress.addEventListener("pointerdown", () => { scoreSeeking = true; });
elements.scoreProgress.addEventListener("input", () => {
  const position = Number(elements.scoreProgress.value);
  elements.scoreTime.textContent = `${formatDuration(position)} / ${formatDuration(scoreDuration)}`;
});
elements.scoreProgress.addEventListener("change", () => {
  scoreSeeking = false;
  seekScore();
});
elements.octaveDown.addEventListener("click", () => shiftKeyboard(-12));
elements.octaveUp.addEventListener("click", () => shiftKeyboard(12));
elements.recordButton.addEventListener("click", toggleRecording);
elements.downloadButton.addEventListener("click", downloadRecording);
elements.clearLogButton.addEventListener("click", () => elements.eventLog.replaceChildren());
elements.velocity.addEventListener("input", () => {
  elements.velocityValue.textContent = elements.velocity.value;
  updateVelocityReadout(elements.velocity.value);
});
elements.velocityCurve.addEventListener("change", () => {
  updateVelocityReadout(elements.velocityRaw.textContent);
});
elements.serverGain.addEventListener("input", () => {
  elements.serverGainValue.textContent = `${Number(elements.serverGain.value).toFixed(2)}×`;
  send({ type: "set_server_gain", server_gain: Number(elements.serverGain.value) });
});
elements.outputGain.addEventListener("input", () => {
  const value = Number(elements.outputGain.value);
  elements.outputGainValue.textContent = `${value.toFixed(2)}×`;
  player?.setGain(value);
});

buildKeyboard();
updateVelocityReadout(elements.velocity.value);
syncScoreControls();
requestAnimationFrame(() => updateKeyboardMapping(true));

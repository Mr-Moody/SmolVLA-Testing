// ── Constants ─────────────────────────────────────────────────────────────
const CHANNEL_COLORS = [
  'rgba(99,102,241,0.82)',  'rgba(168,85,247,0.82)', 'rgba(236,72,153,0.82)',
  'rgba(6,182,212,0.82)',   'rgba(52,211,153,0.82)', 'rgba(245,158,11,0.82)',
  'rgba(251,191,36,0.82)',  'rgba(34,211,238,0.82)',
];
const DEFAULT_CHANNEL_DEFS = [
  { id: 'approach_MSD_plug',               label: 'Approach MSD Plug',          color: CHANNEL_COLORS[0] },
  { id: 'positioning_the_gripper',         label: 'Position Gripper',           color: CHANNEL_COLORS[1] },
  { id: 'grasp_the_plug',                  label: 'Grasp Plug',                 color: CHANNEL_COLORS[2] },
  { id: 'move_the_plug_to_the_socket',     label: 'Move Plug to Socket',        color: CHANNEL_COLORS[3] },
  { id: 'align_handle',                    label: 'Align Handle',               color: CHANNEL_COLORS[4] },
  { id: 'nudging_the_plug_into_the_socket',label: 'Nudge Plug into Socket',     color: CHANNEL_COLORS[5] },
  { id: 'place_the_plug_in_the_socket',    label: 'Place Plug in Socket',       color: CHANNEL_COLORS[6] },
  { id: 'push_down_on_the_plug',           label: 'Push Down on Plug',          color: CHANNEL_COLORS[7] },
];

// ── State ──────────────────────────────────────────────────────────────────
let datasets = [];
let episodes = [];
let currentDataset = null;
let currentEpIdx = 0;
let isPlaying = false;
let seekSuppressed = false;
let videoLoadGeneration = 0;
let videoStatusPollers = [];
let videos = [];
let seekRafId = null;
let pendingSeekRafId = null;
let pendingSeekOffset = null;
let pendingSeekPreview = false;
let lastRequestedOffset = null;
let trimState = {
  start_s: 0,
  end_s: 0,
  dragging: null,
  dirty: false,
};
let timelineSeekState = {
  dragging: false,
  element: null,
};
let subtaskState = {
  channels: [],  // [{ id, label, color, clips: [{ start_s, end_s }, ...] }, ...]
  drag: null,    // { type: 'move'|'resize-left'|'resize-right', channelId, clipIndex, startX, origStart, origEnd }
  dirty: false,
};

function mediaErrorName(code) {
  if (code === 1) return 'MEDIA_ERR_ABORTED';
  if (code === 2) return 'MEDIA_ERR_NETWORK';
  if (code === 3) return 'MEDIA_ERR_DECODE';
  if (code === 4) return 'MEDIA_ERR_SRC_NOT_SUPPORTED';
  return 'UNKNOWN_ERROR';
}

function hideFallback(i) {
  const el = document.getElementById(`fallback-cam${i}`);
  if (!el) return;
  el.classList.remove('show');
  el.innerHTML = '';
}

function showFallback(i, camName, src, reason) {
  const el = document.getElementById(`fallback-cam${i}`);
  if (!el) return;
  const safeCam = camName || `camera ${i}`;
  const safeReason = reason || 'Playback failed.';
  el.innerHTML = `<div><strong>${safeCam} could not be played</strong>${safeReason}<br><a href="${src}" target="_blank" rel="noopener">Open video directly</a></div>`;
  el.classList.add('show');
}

function showConversionProgress(i, camName, progress, message, src) {
  const videoEl = document.getElementById(`video${i}`);
  if (videoEl && videoEl.readyState >= 2 && (videoEl.currentSrc || videoEl.src)) {
    hideFallback(i);
    return;
  }
  const el = document.getElementById(`fallback-cam${i}`);
  if (!el) return;
  const safeCam = camName || `camera ${i}`;
  const pct = Math.max(0, Math.min(100, Number.isFinite(progress) ? progress : 0));
  const text = message || `Converting... ${Math.round(pct)}%`;
  el.innerHTML = `<div><strong>Preparing ${safeCam} for web playback</strong><div class="video-progress-wrap"><div class="video-progress-fill" style="width:${pct}%"></div></div><div class="video-progress-text">${text}</div><a href="${src}" target="_blank" rel="noopener">Open original video</a></div>`;
  el.classList.add('show');
}

function clearVideoPoller(i) {
  if (videoStatusPollers[i]) {
    clearInterval(videoStatusPollers[i]);
    videoStatusPollers[i] = null;
  }
}

async function prepareAndLoadCamera(cam, i, generation) {
  const videoEl = document.getElementById(`video${i}`);
  const src = `/video/${currentDataset}/${cam}`;
  const statusUrl = `/api/video/${currentDataset}/${cam}/status`;
  const prepareUrl = `/api/video/${currentDataset}/${cam}/prepare`;
  if (!videoEl) return;

  clearVideoPoller(i);
  if (!(videoEl.readyState >= 2 && (videoEl.currentSrc || videoEl.src))) {
    showConversionProgress(i, cam, 0, 'Checking video format...', src);
  } else {
    hideFallback(i);
  }

  try {
    await fetch(prepareUrl, { method: 'POST' });
  } catch {
    // Keep polling status; route may still become available.
  }

  const applyReadySource = () => {
    if (videoEl.dataset.src !== src) {
      videoEl.dataset.src = src;
      videoEl.src = src;
      videoEl.load();
    }
    hideFallback(i);
  };

  videoStatusPollers[i] = setInterval(async () => {
    if (generation !== videoLoadGeneration) {
      clearVideoPoller(i);
      return;
    }
    try {
      const res = await fetch(statusUrl);
      if (!res.ok) return;
      const status = await res.json();
      const state = status.state || 'pending';
      const progress = Number(status.progress || 0);
      const message = status.message || '';

      if (videoEl.readyState >= 2 && (videoEl.currentSrc || videoEl.src)) {
        hideFallback(i);
      }

      if (state === 'ready') {
        clearVideoPoller(i);
        applyReadySource();
        return;
      }
      if (state === 'source_only') {
        clearVideoPoller(i);
        applyReadySource();
        return;
      }
      if (state === 'missing') {
        clearVideoPoller(i);
        showFallback(i, cam, src, 'Video file rgb.mp4 was not found.');
        return;
      }
      if (state === 'error') {
        clearVideoPoller(i);
        showFallback(i, cam, src, `Conversion failed: ${message || 'unknown error'}`);
        return;
      }

      showConversionProgress(i, cam, progress, message || `Converting... ${Math.round(progress)}%`, src);
    } catch {
      showConversionProgress(i, cam, 0, 'Waiting for converter...', src);
    }
  }, 350);
}

// ── Dynamic video grid ─────────────────────────────────────────────────────
function rebuildVideoGrid(camNames) {
  videoStatusPollers.forEach((_, i) => clearVideoPoller(i));
  videoStatusPollers = camNames.map(() => null);

  const grid = document.getElementById('video-grid');
  grid.innerHTML = '';
  videos = [];

  camNames.forEach((cam, i) => {
    const panel = document.createElement('div');
    panel.className = 'cam-panel';
    panel.id = `panel-cam${i}`;

    const label = document.createElement('div');
    label.className = 'cam-label';
    label.id = `label-cam${i}`;
    label.textContent = cam;

    const video = document.createElement('video');
    video.id = `video${i}`;
    video.preload = 'auto';
    video.setAttribute('playsinline', '');

    const fallback = document.createElement('div');
    fallback.className = 'video-fallback';
    fallback.id = `fallback-cam${i}`;

    panel.appendChild(label);
    panel.appendChild(video);
    panel.appendChild(fallback);
    grid.appendChild(panel);

    video.addEventListener('loadstart', () => hideFallback(i));
    video.addEventListener('loadeddata', () => hideFallback(i));
    video.addEventListener('error', () => {
      const camLabel = document.getElementById(`label-cam${i}`)?.textContent || `camera ${i}`;
      const errCode = video.error ? video.error.code : 0;
      const reason = `Browser media error: ${mediaErrorName(errCode)} (${errCode}).`;
      showFallback(i, camLabel, video.currentSrc || video.src || '#', reason);
    });

    videos.push(video);
  });

  if (videos[0]) {
    videos[0].addEventListener('timeupdate', () => {
      const ep = episodes[currentEpIdx];
      if (!ep) return;
      const primaryCam = Object.keys(ep.cameras)[0];
      if (!primaryCam) return;
      const end = currentTrimEnd();
      if (videos[0].currentTime >= end - 0.05) {
        pauseAll();
        videos[0].currentTime = end;
      }
    });
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function fmt(sec) {
  if (!isFinite(sec)) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function epDuration() {
  if (!episodes.length) return 0;
  const ep = episodes[currentEpIdx];
  const cams = Object.values(ep.cameras);
  if (!cams.length) return 0;
  return cams[0].end_s - cams[0].start_s;
}

function currentEpisode() {
  return episodes[currentEpIdx] || null;
}

function primaryCameraName() {
  const ep = currentEpisode();
  if (!ep) return null;
  return Object.keys(ep.cameras)[0] || null;
}

function currentTrimStart() {
  const ep = currentEpisode();
  if (!ep) return 0;
  return trimState.start_s ?? ep.trim_start_s ?? primaryStart();
}

function currentTrimEnd() {
  const ep = currentEpisode();
  if (!ep) return 0;
  return trimState.end_s ?? ep.trim_end_s ?? primaryEnd();
}

function primaryStart() {
  if (!episodes.length) return 0;
  const ep = episodes[currentEpIdx];
  const cams = Object.values(ep.cameras);
  return cams.length ? cams[0].start_s : 0;
}

function primaryEnd() {
  if (!episodes.length) return 0;
  const ep = episodes[currentEpIdx];
  const cams = Object.values(ep.cameras);
  return cams.length ? cams[0].end_s : 0;
}

function primaryFrameDuration() {
  const ep = currentEpisode();
  if (!ep) return 1 / 30;
  const primaryCam = primaryCameraName();
  const fps = primaryCam ? Number(ep.cameras[primaryCam]?.fps) : 0;
  return fps > 0 ? 1 / fps : 1 / 30;
}

function currentEpisodeOffset() {
  if (lastRequestedOffset !== null && (!isPlaying || timelineSeekState.dragging || seekSuppressed)) {
    return clampEpisodeOffset(lastRequestedOffset);
  }
  const v0 = videos[0];
  if (!v0) return 0;
  return clampEpisodeOffset(v0.currentTime - primaryStart());
}

function relativeTime() {
  return currentEpisodeOffset();
}

function episodePctForTime(value) {
  const dur = epDuration();
  if (dur <= 0) return 0;
  return ((value - primaryStart()) / dur) * 100;
}

function clampEpisodeOffset(offset) {
  return Math.max(0, Math.min(offset, epDuration()));
}

function videoTimeForOffset(cam, offset) {
  const ep = currentEpisode();
  if (!ep || !ep.cameras[cam]) return 0;
  const info = ep.cameras[cam];
  const camDur = Math.max(0, info.end_s - info.start_s);
  return info.start_s + Math.max(0, Math.min(offset, camDur));
}

function setVideoTime(video, time, preview = false) {
  if (!video) return;
  if (preview && typeof video.fastSeek === 'function') {
    try {
      video.fastSeek(time);
      return;
    } catch {
      // Fall back to precise seeking below.
    }
  }
  video.currentTime = time;
}

function applyAllVideosToOffset(offset, preview = false) {
  const ep = currentEpisode();
  if (!ep) return;
  const camNames = Object.keys(ep.cameras);
  const sharedOffset = clampEpisodeOffset(offset);
  lastRequestedOffset = sharedOffset;
  camNames.forEach((cam, i) => {
    const v = videos[i];
    if (!v) return;
    setVideoTime(v, videoTimeForOffset(cam, sharedOffset), preview);
  });
  updateSeekbar();
}

function setAllVideosToOffset(offset, options = {}) {
  const sharedOffset = clampEpisodeOffset(offset);
  lastRequestedOffset = sharedOffset;
  if (options.deferred) {
    pendingSeekOffset = sharedOffset;
    pendingSeekPreview = Boolean(options.preview);
    if (!pendingSeekRafId) {
      pendingSeekRafId = requestAnimationFrame(() => {
        pendingSeekRafId = null;
        const nextOffset = pendingSeekOffset;
        const preview = pendingSeekPreview;
        pendingSeekOffset = null;
        pendingSeekPreview = false;
        if (nextOffset !== null) applyAllVideosToOffset(nextOffset, preview);
      });
    }
    updateSeekbar();
    return;
  }
  applyAllVideosToOffset(sharedOffset, Boolean(options.preview));
}

function finishPendingSeek() {
  const nextOffset = pendingSeekOffset;
  if (pendingSeekRafId) {
    cancelAnimationFrame(pendingSeekRafId);
    pendingSeekRafId = null;
  }
  pendingSeekOffset = null;
  pendingSeekPreview = false;
  if (nextOffset !== null) {
    applyAllVideosToOffset(nextOffset, false);
  } else if (lastRequestedOffset !== null) {
    applyAllVideosToOffset(lastRequestedOffset, false);
  }
}

function clearPendingSeek() {
  if (pendingSeekRafId) {
    cancelAnimationFrame(pendingSeekRafId);
    pendingSeekRafId = null;
  }
  pendingSeekOffset = null;
  pendingSeekPreview = false;
  lastRequestedOffset = null;
}


function offsetFromTimelinePointer(element, clientX) {
  if (!element) return currentEpisodeOffset();
  const rect = element.getBoundingClientRect();
  const frac = rect.width > 0 ? Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)) : 0;
  return frac * epDuration();
}

function beginTimelineSeek(element, event) {
  if (!episodes.length || !element) return;
  timelineSeekState.dragging = true;
  timelineSeekState.element = element;
  event.preventDefault();
  document.body.classList.add('timeline-seeking');
  if (isPlaying) pauseAll();
  setAllVideosToOffset(offsetFromTimelinePointer(element, event.clientX), { deferred: true, preview: true });
}

function endTimelineSeek() {
  if (!timelineSeekState.dragging) return;
  finishPendingSeek();
  timelineSeekState.dragging = false;
  timelineSeekState.element = null;
  document.body.classList.remove('timeline-seeking');
  updateSeekbar();
}

function syncPeerVideosToPrimary(force = false) {
  const ep = currentEpisode();
  const v0 = videos[0];
  if (!ep || !v0) return;
  const sharedOffset = currentEpisodeOffset();
  Object.keys(ep.cameras).forEach((cam, i) => {
    if (i === 0) return;
    const v = videos[i];
    if (!v) return;
    const target = videoTimeForOffset(cam, sharedOffset);
    if (force || Math.abs(v.currentTime - target) > 0.08) {
      v.currentTime = target;
    }
  });
}

function clampTrimValue(value) {
  return Math.max(primaryStart(), Math.min(value, primaryEnd()));
}

function snapTrimValue(value) {
  const v0 = videos[0];
  if (!v0) return value;
  const SNAP_THRESHOLD_S = 0.15;
  return Math.abs(v0.currentTime - value) <= SNAP_THRESHOLD_S ? v0.currentTime : value;
}

function updateTrimDirtyState() {
  const ep = currentEpisode();
  if (!ep) {
    trimState.dirty = false;
    return;
  }
  const originalStart = ep.trim_start_s ?? primaryStart();
  const originalEnd = ep.trim_end_s ?? primaryEnd();
  trimState.dirty =
    Math.abs(trimState.start_s - originalStart) > 0.001 ||
    Math.abs(trimState.end_s - originalEnd) > 0.001;
}

function updateTrimUi() {
  const timeline = document.getElementById('trim-timeline');
  const region = document.getElementById('trim-region');
  const handleStart = document.getElementById('trim-handle-start');
  const handleEnd = document.getElementById('trim-handle-end');
  const playhead = document.getElementById('trim-playhead');
  const display = document.getElementById('trim-display');
  const saveBtn = document.getElementById('btn-save-trim');
  const dur = epDuration();

  if (!timeline || !region || !handleStart || !handleEnd || !playhead || !display || !saveBtn || dur <= 0) {
    return;
  }

  const startPct = ((currentTrimStart() - primaryStart()) / dur) * 100;
  const endPct = ((currentTrimEnd() - primaryStart()) / dur) * 100;
  const playheadPct = (relativeTime() / dur) * 100;

  region.style.left = `${startPct}%`;
  region.style.width = `${Math.max(0, endPct - startPct)}%`;
  handleStart.style.left = `${startPct}%`;
  handleEnd.style.left = `${endPct}%`;
  playhead.style.left = `${Math.max(0, Math.min(100, playheadPct))}%`;
  display.textContent = `${fmt(currentTrimStart() - primaryStart())} - ${fmt(currentTrimEnd() - primaryStart())}`;
  saveBtn.disabled = !trimState.dirty;
}

function initializeTrimState() {
  trimState.start_s = currentEpisode()?.trim_start_s ?? primaryStart();
  trimState.end_s = currentEpisode()?.trim_end_s ?? primaryEnd();
  trimState.dragging = null;
  trimState.dirty = false;
  updateTrimUi();
}

function setTrimBoundary(handle, nextValue) {
  const MIN_TRIM_GAP_S = 0.1;
  if (handle === 'start') {
    trimState.start_s = Math.min(snapTrimValue(clampTrimValue(nextValue)), trimState.end_s - MIN_TRIM_GAP_S);
  } else {
    trimState.end_s = Math.max(snapTrimValue(clampTrimValue(nextValue)), trimState.start_s + MIN_TRIM_GAP_S);
  }
  trimState.start_s = clampTrimValue(trimState.start_s);
  trimState.end_s = clampTrimValue(trimState.end_s);
  updateTrimDirtyState();
  updateTrimUi();
  setAllVideosToOffset((handle === 'start' ? trimState.start_s : trimState.end_s) - primaryStart(), {
    deferred: true,
    preview: true,
  });
}

function trimValueFromPointer(clientX) {
  const timeline = document.getElementById('trim-timeline');
  const rect = timeline.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  return primaryStart() + frac * epDuration();
}

// ── Subtask channels ───────────────────────────────────────────────────────
function channelDefsKey() {
  return `subtask_channel_defs_${currentDataset || '__default__'}`;
}

function loadChannelDefs() {
  try {
    const raw = localStorage.getItem(channelDefsKey());
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch { /* fall through */ }
  return DEFAULT_CHANNEL_DEFS.map(d => ({ ...d }));
}

function saveChannelDefs() {
  try {
    const raw = localStorage.getItem(channelDefsKey());
    const existing = raw ? JSON.parse(raw) : [];
    const currentIds = new Set(subtaskState.channels.map(c => c.id));
    const current = subtaskState.channels.map(({ id, label, color }) => ({ id, label, color }));
    const preserved = Array.isArray(existing) ? existing.filter(d => !currentIds.has(d.id)) : [];
    localStorage.setItem(channelDefsKey(), JSON.stringify([...current, ...preserved]));
  } catch { /* ignore storage errors */ }
}

function nextChannelColor() {
  return CHANNEL_COLORS[subtaskState.channels.length % CHANNEL_COLORS.length];
}

function buildSubtaskTimeline() {
  const container = document.getElementById('subtask-channels');
  if (!container) return;
  container.innerHTML = '';
  subtaskState.channels.forEach(ch => {
    container.appendChild(buildChannelRow(ch));
  });
}

function buildClipEl(ch, clipIndex) {
  const clip = document.createElement('div');
  clip.className = 'ch-clip';
  clip.dataset.channelId = ch.id;
  clip.dataset.clipIndex = clipIndex;
  clip.style.background = ch.color;

  const rL = document.createElement('div');
  rL.className = 'ch-resize-left';
  rL.dataset.channelId = ch.id;
  rL.dataset.clipIndex = clipIndex;

  const lbl = document.createElement('span');
  lbl.className = 'ch-clip-label';
  lbl.textContent = ch.label;

  const rR = document.createElement('div');
  rR.className = 'ch-resize-right';
  rR.dataset.channelId = ch.id;
  rR.dataset.clipIndex = clipIndex;

  clip.appendChild(rL);
  clip.appendChild(lbl);
  clip.appendChild(rR);
  return clip;
}

function buildChannelRow(ch) {
  const row = document.createElement('div');
  row.className = 'ch-row';
  row.dataset.channelId = ch.id;

  const labelCol = document.createElement('div');
  labelCol.className = 'ch-label-col';

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'ch-label-input';
  input.value = ch.label;
  input.addEventListener('input', () => {
    ch.label = input.value;
    row.querySelectorAll('.ch-clip-label').forEach(el => { el.textContent = ch.label; });
    saveChannelDefs();
  });

  const delBtn = document.createElement('button');
  delBtn.className = 'ch-delete-btn';
  delBtn.title = 'Delete channel';
  delBtn.textContent = '×';
  delBtn.addEventListener('click', () => deleteChannel(ch.id));

  labelCol.appendChild(input);
  labelCol.appendChild(delBtn);

  const track = document.createElement('div');
  track.className = 'ch-track';
  track.dataset.channelId = ch.id;

  ch.clips.forEach((_, i) => track.appendChild(buildClipEl(ch, i)));

  row.appendChild(labelCol);
  row.appendChild(track);
  return row;
}

function normalizeSubtaskTimes(subtasks) {
  if (!subtasks || subtasks.length === 0) return subtasks;
  const minTime = Math.min(...subtasks.map(s => s.start_s));
  if (minTime <= primaryEnd() + 1) return subtasks;
  // Times are absolute recording timestamps — shift so the earliest clip aligns with primaryStart()
  const offset = minTime - primaryStart();
  return subtasks.map(s => ({ ...s, start_s: s.start_s - offset, end_s: s.end_s - offset }));
}

function initializeSubtaskState() {
  const ep = currentEpisode();
  subtaskState.drag = null;
  subtaskState.dirty = false;

  const defs = loadChannelDefs();

  if (!ep || !ep.subtasks || ep.subtasks.length === 0) {
    const trimStart = currentTrimStart();
    const trimEnd = currentTrimEnd();
    const trimDur = Math.max(0, trimEnd - trimStart);
    const N = defs.length || 1;
    subtaskState.channels = defs.map((def, i) => ({
      ...def,
      clips: [{ start_s: trimStart + i * (trimDur / N), end_s: trimStart + (i + 1) * (trimDur / N) }],
    }));
    buildSubtaskTimeline();
    updateSubtaskUi();
    return;
  }

  // Build channels from saved subtask data — handles any order and repeated phases
  const normalizedSubtasks = normalizeSubtaskTimes(ep.subtasks);

  const defMap = new Map(defs.map(d => [d.id, d]));
  const seenIds = [];
  const seenSet = new Set();
  normalizedSubtasks.forEach(s => {
    if (!seenSet.has(s.phase)) { seenSet.add(s.phase); seenIds.push(s.phase); }
  });

  // Assign colors to phases not yet in the dataset's def registry
  let colorIdx = defs.length;
  const newDefs = [];
  seenIds.forEach(id => {
    if (!defMap.has(id)) {
      const def = { id, label: id.replace(/_/g, ' '), color: CHANNEL_COLORS[colorIdx % CHANNEL_COLORS.length] };
      colorIdx++;
      defMap.set(id, def);
      newDefs.push(def);
    }
  });

  // Group all clips by phase (preserving time order within each channel)
  const clipsByPhase = new Map();
  seenIds.forEach(id => clipsByPhase.set(id, []));
  normalizedSubtasks.forEach(s => clipsByPhase.get(s.phase)?.push({ start_s: s.start_s, end_s: s.end_s }));

  subtaskState.channels = seenIds.map(id => ({ ...defMap.get(id), clips: clipsByPhase.get(id) }));

  if (newDefs.length > 0) saveChannelDefs();

  buildSubtaskTimeline();
  updateSubtaskUi();
}

function updateSubtaskUi() {
  const saveBtn = document.getElementById('btn-save-subtasks');
  const dur = epDuration();

  if (!episodes.length || !subtaskState.channels.length || dur <= 0) {
    if (saveBtn) saveBtn.disabled = true;
    const ph = document.getElementById('subtask-playhead');
    if (ph) ph.style.display = 'none';
    return;
  }

  const playheadPct = videos[0] ? episodePctForTime(videos[0].currentTime) : 0;

  subtaskState.channels.forEach(ch => {
    ch.clips.forEach((clip, clipIndex) => {
      const el = document.querySelector(
        `.ch-clip[data-channel-id="${CSS.escape(ch.id)}"][data-clip-index="${clipIndex}"]`
      );
      if (el) {
        el.style.left  = `${episodePctForTime(clip.start_s)}%`;
        el.style.width = `${Math.max(0, episodePctForTime(clip.end_s) - episodePctForTime(clip.start_s))}%`;
      }
    });
  });

  const singlePh = document.getElementById('subtask-playhead');
  if (singlePh) {
    singlePh.style.setProperty('--subtask-ph-pct', (Math.max(0, Math.min(100, playheadPct)) / 100).toFixed(4));
    singlePh.style.display = 'block';
  }

  if (saveBtn) saveBtn.disabled = !subtaskState.dirty;
}

function beginSubtaskDrag(type, channelId, clipIndex, event) {
  if (!episodes.length) return;
  const ch = subtaskState.channels.find(c => c.id === channelId);
  const clip = ch?.clips[clipIndex];
  if (!ch || !clip) return;
  subtaskState.drag = {
    type,
    channelId,
    clipIndex,
    startX: event.clientX,
    origStart: clip.start_s,
    origEnd: clip.end_s,
  };
  event.preventDefault();
  document.body.classList.add('subtask-dragging');
}

function updateSubtaskDrag(clientX) {
  if (!subtaskState.drag) return;
  const { type, channelId, clipIndex, startX, origStart, origEnd } = subtaskState.drag;
  const ch = subtaskState.channels.find(c => c.id === channelId);
  const clip = ch?.clips[clipIndex];
  if (!ch || !clip) return;

  const trackEl = document.querySelector('.ch-track');
  if (!trackEl) return;
  const trackWidth = trackEl.getBoundingClientRect().width;
  const dur = epDuration();
  if (trackWidth <= 0 || dur <= 0) return;

  const MIN_CLIP = 0.05;
  const pxPerSec = trackWidth / dur;
  const deltaT = (clientX - startX) / pxPerSec;

  if (type === 'move') {
    const clipDur = origEnd - origStart;
    const newStart = Math.max(primaryStart(), Math.min(origStart + deltaT, primaryEnd() - clipDur));
    clip.start_s = newStart;
    clip.end_s = newStart + clipDur;
  } else if (type === 'resize-left') {
    clip.start_s = Math.max(primaryStart(), Math.min(origStart + deltaT, origEnd - MIN_CLIP));
    setAllVideosToOffset(clip.start_s - primaryStart(), { deferred: true, preview: true });
  } else if (type === 'resize-right') {
    clip.end_s = Math.max(origStart + MIN_CLIP, Math.min(origEnd + deltaT, primaryEnd()));
    setAllVideosToOffset(clip.end_s - primaryStart(), { deferred: true, preview: true });
  }

  subtaskState.dirty = true;
  updateSubtaskUi();
}

function endSubtaskDrag() {
  if (!subtaskState.drag) return;
  if (subtaskState.drag.type !== 'move') finishPendingSeek();
  document.body.classList.remove('subtask-dragging');
  subtaskState.drag = null;
}

function addChannel() {
  if (!episodes.length) return;
  const trimStart = currentTrimStart();
  const trimEnd = currentTrimEnd();
  const trimDur = Math.max(0, trimEnd - trimStart);
  const id = 'channel_' + Date.now();
  subtaskState.channels.push({
    id,
    label: 'New Channel',
    color: nextChannelColor(),
    clips: [{ start_s: trimStart + trimDur * 0.4, end_s: trimStart + trimDur * 0.6 }],
  });
  saveChannelDefs();
  buildSubtaskTimeline();
  updateSubtaskUi();
  subtaskState.dirty = true;
  document.getElementById('btn-save-subtasks')?.removeAttribute('disabled');
}

function deleteChannel(channelId) {
  subtaskState.channels = subtaskState.channels.filter(c => c.id !== channelId);
  saveChannelDefs();
  buildSubtaskTimeline();
  updateSubtaskUi();
  subtaskState.dirty = true;
  document.getElementById('btn-save-subtasks')?.removeAttribute('disabled');
}

function addClip(channelId, time) {
  const ch = subtaskState.channels.find(c => c.id === channelId);
  if (!ch) return;
  const dur = epDuration();
  const defaultDur = Math.max(0.5, dur * 0.02);
  ch.clips.push({ start_s: time, end_s: Math.min(time + defaultDur, primaryEnd()) });
  ch.clips.sort((a, b) => a.start_s - b.start_s);
  buildSubtaskTimeline();
  updateSubtaskUi();
  subtaskState.dirty = true;
  document.getElementById('btn-save-subtasks')?.removeAttribute('disabled');
}

function deleteClip(channelId, clipIndex) {
  const ch = subtaskState.channels.find(c => c.id === channelId);
  if (!ch || ch.clips.length <= 1) return;
  ch.clips.splice(clipIndex, 1);
  buildSubtaskTimeline();
  updateSubtaskUi();
  subtaskState.dirty = true;
  document.getElementById('btn-save-subtasks')?.removeAttribute('disabled');
}

// ── Dataset loading ─────────────────────────────────────────────────────────
async function loadDatasets() {
  const res = await fetch('/api/datasets');
  datasets = await res.json();
  const sel = document.getElementById('dataset-select');
  sel.innerHTML = datasets.map(d =>
    `<option value="${d.name}">${d.name} (${d.episode_count} episodes)</option>`
  ).join('');
  document.getElementById('loading').classList.add('hidden');
  if (datasets.length) selectDataset(datasets[0].name);
}

async function selectDataset(name) {
  currentDataset = name;
  document.getElementById('loading').classList.remove('hidden');
  const res = await fetch(`/api/datasets/${name}/episodes`);
  episodes = await res.json();
  document.getElementById('loading').classList.add('hidden');

  if (!episodes.length) {
    currentEpIdx = 0;
    document.getElementById('ep-info').textContent = '0 / 0';
    document.getElementById('btn-prev').disabled = true;
    document.getElementById('btn-next').disabled = true;
    document.getElementById('btn-delete').disabled = true;
    document.getElementById('btn-save-trim').disabled = true;
    document.getElementById('btn-save-subtasks').disabled = true;
    document.getElementById('task-input').value = '';
    document.getElementById('ep-annotation-status').textContent = '';
    document.getElementById('trim-display').textContent = '—';
    subtaskState.channels = [];
    subtaskState.drag = null;
    subtaskState.dirty = false;
    buildSubtaskTimeline();
    pauseAll();
    updateProgressBadge();
    return;
  }

  currentEpIdx = 0;
  loadEpisode(0);
  updateProgressBadge();
}

// ── Episode loading ─────────────────────────────────────────────────────────
function loadEpisode(idx) {
  if (!episodes.length) return;
  idx = Math.max(0, Math.min(idx, episodes.length - 1));
  currentEpIdx = idx;
  clearPendingSeek();
  videoLoadGeneration += 1;
  const generation = videoLoadGeneration;

  const ep = episodes[idx];
  const camNames = Object.keys(ep.cameras);

  if (camNames.length !== videos.length ||
      camNames.some((cam, i) => document.getElementById(`label-cam${i}`)?.textContent !== cam)) {
    rebuildVideoGrid(camNames);
  }

  camNames.forEach((cam, i) => {
    hideFallback(i);
    prepareAndLoadCamera(cam, i, generation);
  });

  videos.forEach((v, i) => {
    const cam = camNames[i];
    if (!cam) return;
    const primaryCam = camNames[0];
    const trimOffset = (ep.trim_start_s ?? ep.cameras[primaryCam].start_s) - ep.cameras[primaryCam].start_s;
    const seekTo = () => {
      v.currentTime = videoTimeForOffset(cam, trimOffset);
    };
    if (v.readyState >= 1) {
      seekTo();
    } else {
      v.addEventListener('loadedmetadata', seekTo, { once: true });
    }
  });

  document.getElementById('ep-info').textContent = `${idx + 1} / ${episodes.length}`;
  document.getElementById('btn-prev').disabled = idx === 0;
  document.getElementById('btn-next').disabled = idx === episodes.length - 1;
  document.getElementById('btn-delete').disabled = false;
  document.getElementById('jump-input').value = idx + 1;

  const taskInput = document.getElementById('task-input');
  const annoStatus = document.getElementById('ep-annotation-status');
  const statusMsg = document.getElementById('status-msg');
  statusMsg.textContent = '';
  statusMsg.className = '';
  if (ep.annotation) {
    taskInput.value = ep.annotation;
    annoStatus.textContent = 'labeled';
    annoStatus.className = 'anno-status labeled';
    taskInput.classList.add('has-annotation');
  } else {
    taskInput.value = '';
    annoStatus.textContent = 'unlabeled';
    annoStatus.className = 'anno-status';
    taskInput.classList.remove('has-annotation');
  }

  pauseAll();
  updateSeekbar();
  initializeTrimState();
  initializeSubtaskState();
}

// ── Playback controls ──────────────────────────────────────────────────────
function playAll() {
  if (episodes.length) {
    const ep = episodes[currentEpIdx];
    const camNames = Object.keys(ep.cameras);
    const primaryCam = camNames[0];
    const trimStart = currentTrimStart();
    const trimEnd = currentTrimEnd();
    if (primaryCam && videos[0] && (videos[0].currentTime < trimStart || videos[0].currentTime >= trimEnd - 0.1)) {
      setAllVideosToOffset(trimStart - ep.cameras[primaryCam].start_s);
    } else {
      syncPeerVideosToPrimary(true);
    }
  }
  videos.forEach(v => { if (v.src) v.play().catch(() => {}); });
  document.getElementById('btn-playpause').innerHTML = '<span class="btn-icon pause-bars"></span> Pause';
  lastRequestedOffset = null;
  isPlaying = true;
  startSeekbarAnimation();
}

function pauseAll() {
  videos.forEach(v => v.pause());
  document.getElementById('btn-playpause').innerHTML = '<span class="btn-icon">&#9654;</span>&nbsp;Play';
  lastRequestedOffset = null;
  isPlaying = false;
  stopSeekbarAnimation();
}

function togglePlay() {
  if (isPlaying) pauseAll(); else playAll();
}

function jumpToStart() {
  if (!episodes.length) return;
  const ep = episodes[currentEpIdx];
  const camNames = Object.keys(ep.cameras);
  const primaryCam = camNames[0];
  if (!primaryCam) return;
  const trimStart = currentTrimStart();
  setAllVideosToOffset(trimStart - ep.cameras[primaryCam].start_s);
}

function jumpToEnd() {
  if (!episodes.length) return;
  const ep = episodes[currentEpIdx];
  const camNames = Object.keys(ep.cameras);
  const primaryCam = camNames[0];
  if (!primaryCam) return;
  const trimEnd = currentTrimEnd();
  setAllVideosToOffset(trimEnd - ep.cameras[primaryCam].start_s);
}

function skipSeconds(delta) {
  if (!episodes.length) return;
  setAllVideosToOffset(currentEpisodeOffset() + delta);
}

function stepFrames(direction) {
  if (!episodes.length) return;
  if (isPlaying) pauseAll();
  setAllVideosToOffset(currentEpisodeOffset() + direction * primaryFrameDuration());
}

function seekToFraction(frac) {
  if (!episodes.length) return;
  setAllVideosToOffset(frac * epDuration());
}

// ── Seekbar sync ──────────────────────────────────────────────────────────
function updateSeekbar() {
  if (seekSuppressed) return;
  const dur = epDuration();
  const rel = relativeTime();
  const seekbar = document.getElementById('seekbar');
  seekbar.value = dur > 0 ? (rel / dur) * 1000 : 0;
  document.getElementById('time-display').textContent = `${fmt(rel)} / ${fmt(dur)}`;
  updateTrimUi();
  updateSubtaskUi();
}

function seekbarRafLoop() {
  if (isPlaying) syncPeerVideosToPrimary();
  updateSeekbar();
  seekRafId = requestAnimationFrame(seekbarRafLoop);
}

function startSeekbarAnimation() {
  if (!seekRafId) seekRafId = requestAnimationFrame(seekbarRafLoop);
}

function stopSeekbarAnimation() {
  if (seekRafId) {
    cancelAnimationFrame(seekRafId);
    seekRafId = null;
  }
  updateSeekbar();
}

function updateProgressBadge() {
  const labeled = episodes.filter(e => e.annotation).length;
  document.getElementById('progress-badge').textContent = `${labeled} / ${episodes.length}`;
}

// ── Annotation submit ─────────────────────────────────────────────────────
async function submitAnnotation(silent = false) {
  if (!currentDataset || !episodes.length) return;
  const task = document.getElementById('task-input').value.trim();
  if (!task) {
    if (!silent) showStatus('Please enter a task description.', false);
    return;
  }
  const ep = episodes[currentEpIdx];
  const res = await fetch(`/api/datasets/${currentDataset}/episodes/${ep.index}/annotate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task }),
  });
  if (res.ok) {
    ep.annotation = task;
    const annoStatus = document.getElementById('ep-annotation-status');
    annoStatus.textContent = 'labeled';
    annoStatus.className = 'anno-status labeled';
    document.getElementById('task-input').classList.add('has-annotation');
    updateProgressBadge();
    if (!silent) showStatus('Saved', true);
  } else {
    const body = await res.json().catch(() => ({}));
    if (!silent) showStatus('Error: ' + (body.error || res.statusText), false);
    throw new Error(body.error || res.statusText);
  }
}

async function saveTrim(silent = false) {
  if (!currentDataset || !episodes.length) return;
  const ep = currentEpisode();
  const res = await fetch(`/api/datasets/${currentDataset}/episodes/${ep.index}/trim`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      trim_start_s: Number(trimState.start_s.toFixed(4)),
      trim_end_s: Number(trimState.end_s.toFixed(4)),
    }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    if (!silent) showStatus('Error: ' + (body.error || res.statusText), false);
    throw new Error(body.error || res.statusText);
  }

  ep.trim_start_s = trimState.start_s;
  ep.trim_end_s = trimState.end_s;
  updateTrimDirtyState();
  updateTrimUi();
  if (!silent) showStatus('Trim saved', true);
}

async function saveSubtasks(silent = false) {
  if (!currentDataset || !episodes.length) return;
  const ep = currentEpisode();
  const subtasks = [];
  subtaskState.channels.forEach(ch => {
    ch.clips.forEach(clip => subtasks.push({
      phase:   ch.id,
      start_s: Number(clip.start_s.toFixed(4)),
      end_s:   Number(clip.end_s.toFixed(4)),
    }));
  });
  subtasks.sort((a, b) => a.start_s - b.start_s);
  const res = await fetch(`/api/datasets/${currentDataset}/episodes/${ep.index}/subtasks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ subtasks }),
  });
  if (res.ok) {
    ep.subtasks = subtasks;
    subtaskState.dirty = false;
    updateSubtaskUi();
    if (!silent) showStatus('Subtasks saved', true);
  } else {
    const body = await res.json().catch(() => ({}));
    if (!silent) showStatus('Error: ' + (body.error || res.statusText), false);
    throw new Error(body.error || res.statusText);
  }
}

async function deleteEpisode() {
  if (!currentDataset || !episodes.length) return;
  const ep = episodes[currentEpIdx];
  const confirmed = window.confirm(`Delete episode ${ep.index} from the cleaned dataset?`);
  if (!confirmed) return;

  const previousIdx = currentEpIdx;
  const res = await fetch(`/api/datasets/${currentDataset}/episodes/${ep.index}`, {
    method: 'DELETE',
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    showStatus('Error: ' + (body.error || res.statusText), false);
    return;
  }

  await selectDataset(currentDataset);
  if (episodes.length) {
    loadEpisode(Math.min(Math.max(0, previousIdx - 1), episodes.length - 1));
    showStatus('Episode deleted', true);
  } else {
    showStatus('No episodes remaining', true);
  }
}

function showStatus(msg, ok) {
  const el = document.getElementById('status-msg');
  el.textContent = msg;
  el.className = ok ? 'ok' : 'err';
  setTimeout(() => {
    el.style.opacity = '0';
    setTimeout(() => { el.className = ''; el.textContent = ''; el.style.opacity = ''; }, 400);
  }, 2800);
}

// ── Event wiring ──────────────────────────────────────────────────────────
document.getElementById('dataset-select').addEventListener('change', e => {
  selectDataset(e.target.value);
});
document.getElementById('btn-prev').addEventListener('click', () => loadEpisode(currentEpIdx - 1));
document.getElementById('btn-next').addEventListener('click', () => loadEpisode(currentEpIdx + 1));
document.getElementById('btn-go').addEventListener('click', () => {
  const v = parseInt(document.getElementById('jump-input').value, 10);
  if (!isNaN(v)) loadEpisode(v - 1);
});
document.getElementById('jump-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('btn-go').click();
});
document.getElementById('btn-playpause').addEventListener('click', togglePlay);
document.getElementById('btn-jump-start').addEventListener('click', jumpToStart);
document.getElementById('btn-jump-end').addEventListener('click', jumpToEnd);
document.getElementById('btn-skip-back').addEventListener('click', () => skipSeconds(-5));
document.getElementById('btn-skip-fwd').addEventListener('click', () => skipSeconds(5));
document.getElementById('btn-submit').addEventListener('click', submitAnnotation);
document.getElementById('btn-delete').addEventListener('click', deleteEpisode);
document.getElementById('btn-save-trim').addEventListener('click', saveTrim);
document.getElementById('btn-save-subtasks').addEventListener('click', saveSubtasks);

async function saveAll() {
  if (!currentDataset || !episodes.length) return;
  const task = document.getElementById('task-input').value.trim();
  const toSave = [];
  if (task) toSave.push(() => submitAnnotation(true));
  if (trimState.dirty) toSave.push(() => saveTrim(true));
  if (subtaskState.dirty) toSave.push(() => saveSubtasks(true));
  if (!toSave.length) { showStatus('Nothing to save', true); return; }
  const errors = await Promise.allSettled(toSave.map(fn => fn()));
  const failed = errors.filter(r => r.status === 'rejected');
  if (failed.length) {
    showStatus('Error: ' + failed[0].reason?.message, false);
  } else {
    showStatus('Saved', true);
  }
}

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.code === 'KeyS') {
    e.preventDefault();
    saveAll();
    return;
  }
  // Block shortcuts when typing in text fields, but allow them when a range
  // input (seekbar) is focused — just prevent the default slider nudge.
  if (e.target.tagName === 'TEXTAREA') return;
  if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  if (e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      if (e.code === 'ArrowLeft') loadEpisode(currentEpIdx - 1);
      else loadEpisode(currentEpIdx + 1);
    } else {
      stepFrames(e.code === 'ArrowLeft' ? -1 : 1);
    }
  }
});

const seekbar = document.getElementById('seekbar');
seekbar.addEventListener('mousedown', () => { seekSuppressed = true; });
seekbar.addEventListener('touchstart', () => { seekSuppressed = true; });
seekbar.addEventListener('input', () => {
  setAllVideosToOffset((seekbar.value / 1000) * epDuration(), { deferred: true, preview: true });
  const dur = epDuration();
  document.getElementById('time-display').textContent = `${fmt((seekbar.value / 1000) * dur)} / ${fmt(dur)}`;
  updateTrimUi();
  updateSubtaskUi();
});
seekbar.addEventListener('mouseup', () => { finishPendingSeek(); seekSuppressed = false; updateSeekbar(); });
seekbar.addEventListener('touchend', () => { finishPendingSeek(); seekSuppressed = false; updateSeekbar(); });

const trimTimeline = document.getElementById('trim-timeline');
const trimHandleStart = document.getElementById('trim-handle-start');
const trimHandleEnd = document.getElementById('trim-handle-end');

function beginTrimDrag(handle, event) {
  if (!episodes.length) return;
  trimState.dragging = handle;
  event.preventDefault();
  document.body.classList.add('trim-dragging');
  document.getElementById(`trim-handle-${handle}`).classList.add('dragging');
  setTrimBoundary(handle, trimValueFromPointer(event.clientX));
}

function endTrimDrag() {
  if (!trimState.dragging) return;
  finishPendingSeek();
  document.body.classList.remove('trim-dragging');
  document.getElementById(`trim-handle-${trimState.dragging}`)?.classList.remove('dragging');
  trimState.dragging = null;
}

trimHandleStart.addEventListener('mousedown', event => beginTrimDrag('start', event));
trimHandleEnd.addEventListener('mousedown', event => beginTrimDrag('end', event));

trimTimeline.addEventListener('mousedown', event => {
  if (!episodes.length) return;
  if (event.target === trimHandleStart || event.target === trimHandleEnd) return;
  beginTimelineSeek(trimTimeline, event);
});

document.getElementById('subtask-channels').addEventListener('mousedown', event => {
  if (!episodes.length) return;
  const rL = event.target.closest('.ch-resize-left');
  const rR = event.target.closest('.ch-resize-right');
  const cl = event.target.closest('.ch-clip');
  const tr = event.target.closest('.ch-track');
  if (rL)      beginSubtaskDrag('resize-left',  rL.dataset.channelId, parseInt(rL.dataset.clipIndex), event);
  else if (rR) beginSubtaskDrag('resize-right', rR.dataset.channelId, parseInt(rR.dataset.clipIndex), event);
  else if (cl) beginSubtaskDrag('move',         cl.dataset.channelId, parseInt(cl.dataset.clipIndex), event);
  else if (tr) beginTimelineSeek(tr, event);
});

document.getElementById('subtask-channels').addEventListener('dblclick', event => {
  if (!episodes.length) return;
  const tr = event.target.closest('.ch-track');
  if (!tr || event.target.closest('.ch-clip')) return;
  const rect = tr.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  addClip(tr.dataset.channelId, primaryStart() + frac * epDuration());
});

document.getElementById('subtask-channels').addEventListener('contextmenu', event => {
  const cl = event.target.closest('.ch-clip');
  if (!cl) return;
  event.preventDefault();
  deleteClip(cl.dataset.channelId, parseInt(cl.dataset.clipIndex));
});

document.addEventListener('mousemove', event => {
  if (timelineSeekState.dragging) {
    setAllVideosToOffset(offsetFromTimelinePointer(timelineSeekState.element, event.clientX), { deferred: true, preview: true });
  }
  if (trimState.dragging) {
    setTrimBoundary(trimState.dragging, trimValueFromPointer(event.clientX));
  }
  if (subtaskState.drag !== null) {
    updateSubtaskDrag(event.clientX);
  }
});

document.addEventListener('mouseup', () => {
  endTimelineSeek();
  endTrimDrag();
  endSubtaskDrag();
});

// ── Boot ──────────────────────────────────────────────────────────────────
document.getElementById('btn-add-channel').addEventListener('click', addChannel);
buildSubtaskTimeline();
loadDatasets();

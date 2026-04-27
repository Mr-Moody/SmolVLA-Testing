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
let trimState = {
  start_s: 0,
  end_s: 0,
  dragging: null,
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

function relativeTime() {
  const v0 = videos[0];
  if (!v0) return 0;
  return Math.max(0, v0.currentTime - primaryStart());
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
}

function trimValueFromPointer(clientX) {
  const timeline = document.getElementById('trim-timeline');
  const rect = timeline.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  return primaryStart() + frac * epDuration();
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
    document.getElementById('task-input').value = '';
    document.getElementById('ep-annotation-status').textContent = '';
    document.getElementById('trim-display').textContent = '—';
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
    const info = ep.cameras[cam];
    const primaryCam = camNames[0];
    const trimOffset = (ep.trim_start_s ?? ep.cameras[primaryCam].start_s) - ep.cameras[primaryCam].start_s;
    const seekTo = () => {
      v.currentTime = Math.max(info.start_s, Math.min(info.start_s + trimOffset, info.end_s));
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
  document.getElementById('jump-input').value = idx;

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
      videos.forEach((v, i) => {
        const cam = camNames[i];
        if (!cam) return;
        const offset = trimStart - ep.cameras[primaryCam].start_s;
        v.currentTime = Math.max(
          ep.cameras[cam].start_s,
          Math.min(ep.cameras[cam].start_s + offset, ep.cameras[cam].end_s),
        );
      });
    }
  }
  videos.forEach(v => { if (v.src) v.play().catch(() => {}); });
  document.getElementById('btn-playpause').innerHTML = '<span class="btn-icon pause-bars"></span> Pause';
  isPlaying = true;
  startSeekbarAnimation();
}

function pauseAll() {
  videos.forEach(v => v.pause());
  document.getElementById('btn-playpause').innerHTML = '<span class="btn-icon">▶</span> Play';
  isPlaying = false;
  stopSeekbarAnimation();
}

function togglePlay() {
  if (isPlaying) pauseAll(); else playAll();
}

function skipSeconds(delta) {
  if (!episodes.length) return;
  const ep = episodes[currentEpIdx];
  const camNames = Object.keys(ep.cameras);
  const primaryCam = camNames[0];
  if (!primaryCam) return;
  const info = ep.cameras[primaryCam];
  const newTime = Math.max(info.start_s, Math.min(videos[0].currentTime + delta, info.end_s));
  const offset = newTime - videos[0].currentTime;
  videos.forEach((v, i) => {
    const cam = camNames[i];
    if (!cam) return;
    v.currentTime = Math.max(ep.cameras[cam].start_s, Math.min(v.currentTime + offset, ep.cameras[cam].end_s));
  });
}

function seekToFraction(frac) {
  if (!episodes.length) return;
  const ep = episodes[currentEpIdx];
  const camNames = Object.keys(ep.cameras);
  camNames.forEach((cam, i) => {
    const v = videos[i];
    if (!v) return;
    const info = ep.cameras[cam];
    v.currentTime = info.start_s + frac * (info.end_s - info.start_s);
  });
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
}

function seekbarRafLoop() {
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
async function submitAnnotation() {
  if (!currentDataset || !episodes.length) return;
  const task = document.getElementById('task-input').value.trim();
  if (!task) {
    showStatus('Please enter a task description.', false);
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
    showStatus('Saved', true);
  } else {
    const body = await res.json().catch(() => ({}));
    showStatus('Error: ' + (body.error || res.statusText), false);
  }
}

async function saveTrim() {
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
    showStatus('Error: ' + (body.error || res.statusText), false);
    return;
  }

  ep.trim_start_s = trimState.start_s;
  ep.trim_end_s = trimState.end_s;
  updateTrimDirtyState();
  updateTrimUi();
  showStatus('Trim saved', true);
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
  if (!isNaN(v)) loadEpisode(v);
});
document.getElementById('jump-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('btn-go').click();
});
document.getElementById('btn-playpause').addEventListener('click', togglePlay);
document.getElementById('btn-skip-back').addEventListener('click', () => skipSeconds(-5));
document.getElementById('btn-skip-fwd').addEventListener('click', () => skipSeconds(5));
document.getElementById('btn-submit').addEventListener('click', submitAnnotation);
document.getElementById('btn-delete').addEventListener('click', deleteEpisode);
document.getElementById('btn-save-trim').addEventListener('click', saveTrim);

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  if (e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      const sel = document.getElementById('dataset-select');
      const dir = e.code === 'ArrowLeft' ? -1 : 1;
      const next = sel.selectedIndex + dir;
      if (next >= 0 && next < sel.options.length) {
        sel.selectedIndex = next;
        selectDataset(sel.value);
      }
    } else {
      if (e.code === 'ArrowLeft') loadEpisode(currentEpIdx - 1);
      else loadEpisode(currentEpIdx + 1);
    }
  }
});

const seekbar = document.getElementById('seekbar');
seekbar.addEventListener('mousedown', () => { seekSuppressed = true; });
seekbar.addEventListener('input', () => {
  seekToFraction(seekbar.value / 1000);
  const dur = epDuration();
  document.getElementById('time-display').textContent = `${fmt((seekbar.value / 1000) * dur)} / ${fmt(dur)}`;
});
seekbar.addEventListener('mouseup', () => { seekSuppressed = false; });
seekbar.addEventListener('touchend', () => { seekSuppressed = false; });

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
  document.body.classList.remove('trim-dragging');
  document.getElementById(`trim-handle-${trimState.dragging}`)?.classList.remove('dragging');
  trimState.dragging = null;
}

trimHandleStart.addEventListener('mousedown', event => beginTrimDrag('start', event));
trimHandleEnd.addEventListener('mousedown', event => beginTrimDrag('end', event));

trimTimeline.addEventListener('mousedown', event => {
  if (!episodes.length) return;
  if (event.target === trimHandleStart || event.target === trimHandleEnd) return;
  const targetValue = trimValueFromPointer(event.clientX);
  const startDist = Math.abs(targetValue - trimState.start_s);
  const endDist = Math.abs(targetValue - trimState.end_s);
  beginTrimDrag(startDist <= endDist ? 'start' : 'end', event);
});

document.addEventListener('mousemove', event => {
  if (!trimState.dragging) return;
  setTrimBoundary(trimState.dragging, trimValueFromPointer(event.clientX));
});

document.addEventListener('mouseup', endTrimDrag);

// ── Boot ──────────────────────────────────────────────────────────────────
loadDatasets();

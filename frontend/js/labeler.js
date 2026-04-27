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
      const end = ep.cameras[primaryCam].end_s;
      if (videos[0].currentTime >= end - 0.05) {
        pauseAll();
        videos[0].currentTime = end;
      }
      updateSeekbar();
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
    document.getElementById('task-input').value = '';
    document.getElementById('ep-annotation-status').textContent = '';
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
    const seekTo = () => { v.currentTime = info.start_s; };
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
  } else {
    taskInput.value = '';
    annoStatus.textContent = 'unlabeled';
    annoStatus.className = 'anno-status';
  }

  pauseAll();
  updateSeekbar();
}

// ── Playback controls ──────────────────────────────────────────────────────
function playAll() {
  if (episodes.length) {
    const ep = episodes[currentEpIdx];
    const camNames = Object.keys(ep.cameras);
    const primaryCam = camNames[0];
    if (primaryCam && videos[0] && videos[0].currentTime >= ep.cameras[primaryCam].end_s - 0.1) {
      videos.forEach((v, i) => {
        const cam = camNames[i];
        if (cam) v.currentTime = ep.cameras[cam].start_s;
      });
    }
  }
  videos.forEach(v => { if (v.src) v.play().catch(() => {}); });
  document.getElementById('btn-playpause').textContent = '❚❚ Pause';
  isPlaying = true;
}

function pauseAll() {
  videos.forEach(v => v.pause());
  document.getElementById('btn-playpause').textContent = '▶ Play';
  isPlaying = false;
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
  seekbar.value = dur > 0 ? Math.round((rel / dur) * 1000) : 0;
  document.getElementById('time-display').textContent = `${fmt(rel)} / ${fmt(dur)}`;
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
    updateProgressBadge();
    showStatus('Saved', true);
  } else {
    const body = await res.json().catch(() => ({}));
    showStatus('Error: ' + (body.error || res.statusText), false);
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

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  if (e.code === 'ArrowLeft') skipSeconds(-5);
  if (e.code === 'ArrowRight') skipSeconds(5);
  if (e.code === 'ArrowUp') loadEpisode(currentEpIdx - 1);
  if (e.code === 'ArrowDown') loadEpisode(currentEpIdx + 1);
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

// ── Boot ──────────────────────────────────────────────────────────────────
loadDatasets();

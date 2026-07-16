"""Small, self-contained web interface for manual Aegis TTS checks.

The API is the product surface of this service; this module deliberately keeps
the browser UI dependency-free and local.  ``render_ui`` returns a static page
so ``main.py`` can serve it without a separate frontend build step.
"""

from __future__ import annotations


HTML_PAGE = r"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark light">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; connect-src 'self'; img-src 'self' data:; media-src 'self' blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'self'">
  <title>Aegis TTS</title>
  <style>
    :root { --bg: #10151d; --panel: #18212d; --panel-2: #202c3b; --text: #eaf1f8; --muted: #aebdcb; --accent: #5cb7ff; --ok: #63d49a; --warn: #ffd166; --bad: #ff7b7b; --border: #32445b; }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; background: var(--bg); color: var(--text); font: 16px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { max-width: 980px; margin: 0 auto; padding: 28px 20px 48px; }
    h1 { margin: 0; font-size: clamp(1.65rem, 4vw, 2.2rem); }
    h2 { margin: 0 0 12px; font-size: 1.1rem; }
    p { color: var(--muted); }
    .subtitle { margin-top: 4px; }
    .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
    .card { border: 1px solid var(--border); background: var(--panel); border-radius: 12px; padding: 18px; }
    .status { display: inline-flex; align-items: center; gap: 7px; border-radius: 999px; padding: 4px 10px; background: var(--panel-2); font-size: .9rem; }
    .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--muted); }
    .dot.ready { background: var(--ok); } .dot.loading { background: var(--warn); } .dot.error { background: var(--bad); }
    .meta { display: grid; grid-template-columns: auto 1fr; gap: 6px 12px; font-size: .9rem; }
    .meta dt { color: var(--muted); } .meta dd { margin: 0; overflow-wrap: anywhere; }
    label { display: block; margin: 14px 0 6px; font-weight: 600; }
    input, select, textarea, button { font: inherit; }
    input, select, textarea { width: 100%; color: var(--text); background: #0e141c; border: 1px solid var(--border); border-radius: 7px; padding: 9px 10px; }
    textarea { min-height: 145px; resize: vertical; }
    input[type="checkbox"] { width: auto; accent-color: var(--accent); }
    .inline { display: flex; align-items: center; gap: 8px; }
    .inline label { margin: 0; font-weight: 400; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
    button { cursor: pointer; color: #061522; background: var(--accent); border: 0; border-radius: 7px; font-weight: 700; padding: 10px 15px; }
    button.secondary { color: var(--text); background: transparent; border: 1px solid var(--border); }
    button:disabled { cursor: wait; opacity: .65; }
    details { margin-top: 18px; border-top: 1px solid var(--border); padding-top: 14px; }
    summary { cursor: pointer; color: var(--muted); }
    .advanced-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0 12px; }
    .error { display: none; margin: 16px 0 0; color: #fff; background: #742e36; border: 1px solid var(--bad); border-radius: 7px; padding: 10px; }
    .error.show { display: block; }
    .hint { color: var(--muted); font-size: .85rem; margin: 5px 0 0; }
    .audio-results { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); margin-top: 16px; }
    .audio-result { border: 1px solid var(--border); background: var(--panel-2); border-radius: 9px; padding: 13px; }
    audio { display: block; width: 100%; margin: 10px 0; }
    pre { max-height: 200px; overflow: auto; margin: 8px 0 0; padding: 10px; white-space: pre-wrap; background: #0e141c; border-radius: 6px; font: .8rem/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .hidden { display: none; }
    @media (max-width: 560px) { main { padding: 20px 14px 36px; } .card { padding: 14px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Aegis TTS</h1>
      <p class="subtitle">Teste local do clone de voz Qwen3-TTS Base.</p>
    </header>

    <section class="grid" aria-label="Estado do serviço">
      <div class="card">
        <h2>Serviço</h2>
        <p class="status"><span id="status-dot" class="dot loading"></span><span id="service-status">Consultando…</span></p>
        <dl id="service-meta" class="meta"></dl>
      </div>
      <div class="card">
        <h2>GPU</h2>
        <dl id="gpu-meta" class="meta"><dt>Estado</dt><dd>Consultando…</dd></dl>
      </div>
    </section>

    <section class="card" style="margin-top:16px">
      <h2>Síntese</h2>
      <label for="api-key">Chave de API (opcional)</label>
      <input id="api-key" type="password" autocomplete="off" spellcheck="false" placeholder="Enviada somente nesta sessão do navegador">
      <p class="hint">A chave fica apenas no armazenamento de sessão deste navegador e não é enviada em consultas de status.</p>

      <div class="grid">
        <div>
          <label for="voice">Voz <span id="default-voice" class="hint"></span></label>
          <select id="voice" aria-describedby="voice-hint"></select>
          <p id="voice-hint" class="hint">Carregando vozes…</p>
        </div>
        <div>
          <label for="seed">Seed (opcional)</label>
          <input id="seed" type="number" step="1" inputmode="numeric" value="41">
          <p class="hint">Padrão: 41. Use a mesma seed para comparações reproduzíveis.</p>
        </div>
      </div>

      <label for="text">Texto</label>
      <textarea id="text" maxlength="1000" required>Olá, Pedro. O serviço de voz está funcionando corretamente.</textarea>
      <p id="character-count" class="hint">0 / 1000 caracteres</p>

      <details>
        <summary>Parâmetros avançados</summary>
        <p class="hint">Vazio = usar a configuração nativa do modelo.</p>
        <div class="advanced-grid">
          <div><label for="temperature">Temperature</label><input id="temperature" type="number" min="0" max="2" step="0.05" placeholder="nativo"></div>
          <div><label for="top-p">Top P</label><input id="top-p" type="number" min="0" max="1" step="0.05" placeholder="nativo"></div>
          <div><label for="max-new-tokens">Máx. tokens</label><input id="max-new-tokens" type="number" min="1" max="4096" step="1" placeholder="nativo"></div>
          <div><label for="do-sample">Amostragem</label><select id="do-sample"><option value="">Nativo</option><option value="true">Ativada</option><option value="false">Desativada</option></select></div>
        </div>
      </details>

      <div class="controls">
        <button id="generate" type="button">Gerar</button>
        <button id="compare" type="button" class="secondary">Comparar Giseli 10s × Giseli 34s</button>
      </div>
      <div id="error" class="error" role="alert" aria-live="assertive"></div>
      <div id="results" class="audio-results" aria-live="polite"></div>
    </section>
  </main>

  <script>
  (() => {
    'use strict';
    const $ = (id) => document.getElementById(id);
    const el = {
      apiKey: $('api-key'), voice: $('voice'), defaultVoice: $('default-voice'), voiceHint: $('voice-hint'),
      text: $('text'), count: $('character-count'), seed: $('seed'), temperature: $('temperature'),
      topP: $('top-p'), maxNewTokens: $('max-new-tokens'), doSample: $('do-sample'), generate: $('generate'),
      compare: $('compare'), error: $('error'), results: $('results'), status: $('service-status'),
      statusDot: $('status-dot'), serviceMeta: $('service-meta'), gpuMeta: $('gpu-meta')
    };
    let defaultVoice = null;
    let activeUrls = [];

    function text(value) { return value === undefined || value === null || value === '' ? '—' : String(value); }
    function setList(target, entries) {
      target.replaceChildren();
      for (const [name, value] of entries) {
        const dt = document.createElement('dt'); dt.textContent = name;
        const dd = document.createElement('dd'); dd.textContent = text(value);
        target.append(dt, dd);
      }
    }
    function setError(message) {
      el.error.textContent = message || '';
      el.error.classList.toggle('show', Boolean(message));
    }
    function setBusy(busy, label) {
      el.generate.disabled = busy;
      el.compare.disabled = busy || !hasComparisonVoices();
      el.generate.textContent = busy ? (label || 'Gerando…') : 'Gerar';
    }
    function hasComparisonVoices() {
      const ids = [...el.voice.options].filter((option) => !option.disabled).map((option) => option.value);
      return ids.includes('giseli-10s') && ids.includes('giseli-34s');
    }
    function authHeaders() {
      const key = el.apiKey.value.trim();
      return key ? {'X-API-Key': key} : {};
    }
    function updateCount() { el.count.textContent = `${el.text.value.length} / 1000 caracteres`; }
    function apiError(response, body) {
      if (body && typeof body.detail === 'string') return body.detail;
      if (body && Array.isArray(body.detail)) return body.detail.map((item) => item.msg || 'Entrada inválida').join('; ');
      return `A solicitação falhou (HTTP ${response.status}).`;
    }
    async function responseError(response) {
      let body = null;
      try { body = await response.json(); } catch (_) { /* no JSON error body */ }
      return apiError(response, body);
    }
    function currentPayload(voice) {
      const source = el.text.value.trim();
      if (!source) throw new Error('Informe um texto para sintetizar.');
      if (source.length > 1000) throw new Error('O texto excede o limite de 1.000 caracteres.');
      const generation_parameters = {};
      if (el.temperature.value !== '') generation_parameters.temperature = Number(el.temperature.value);
      if (el.topP.value !== '') generation_parameters.top_p = Number(el.topP.value);
      if (el.maxNewTokens.value !== '') generation_parameters.max_new_tokens = Number(el.maxNewTokens.value);
      if (el.doSample.value !== '') generation_parameters.do_sample = el.doSample.value === 'true';
      const payload = { text: source, voice, language: 'pt-BR', generation_parameters };
      if (el.seed.value.trim() !== '') payload.seed = Number(el.seed.value);
      return payload;
    }
    function clearResults() {
      for (const url of activeUrls) URL.revokeObjectURL(url);
      activeUrls = [];
      el.results.replaceChildren();
    }
    function metricHeaders(headers) {
      const pairs = [
        ['Geração', headers.get('X-Aegis-TTS-Generation-Seconds'), ' s'],
        ['Áudio', headers.get('X-Aegis-TTS-Audio-Duration-Seconds'), ' s'],
        ['RTF', headers.get('X-Aegis-TTS-RTF'), ''],
        ['Fila', headers.get('X-Aegis-TTS-Queue-Wait-Seconds'), ' s'],
        ['Sample rate', headers.get('X-Aegis-TTS-Sample-Rate'), ' Hz'],
        ['Seed', headers.get('X-Aegis-TTS-Seed'), ''],
        ['VRAM antes', headers.get('X-Aegis-TTS-VRAM-Before-MiB'), ' MiB'],
        ['VRAM depois', headers.get('X-Aegis-TTS-VRAM-After-MiB'), ' MiB'],
        ['VRAM pico', headers.get('X-Aegis-TTS-VRAM-Peak-MiB'), ' MiB']
      ];
      return pairs.filter(([, value]) => value !== null).map(([name, value, suffix]) => [name, `${value}${suffix}`]);
    }
    function addResult(label, blob, headers) {
      const card = document.createElement('article'); card.className = 'audio-result';
      const heading = document.createElement('strong'); heading.textContent = label;
      const player = document.createElement('audio'); player.controls = true;
      const url = URL.createObjectURL(blob); activeUrls.push(url); player.src = url;
      const metrics = document.createElement('pre');
      metrics.textContent = metricHeaders(headers).map(([key, value]) => `${key}: ${value}`).join('\n') || 'Métricas não informadas.';
      card.append(heading, player, metrics); el.results.append(card);
    }
    async function synthesize(payload, label) {
      const response = await fetch('/v1/synthesize', {
        method: 'POST', headers: {'Content-Type': 'application/json', ...authHeaders()}, body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error(await responseError(response));
      const contentType = response.headers.get('content-type') || '';
      if (!contentType.toLowerCase().includes('audio/wav')) throw new Error('O serviço não retornou um arquivo WAV.');
      addResult(label, await response.blob(), response.headers);
    }
    async function generate() {
      try {
        setError(''); clearResults(); setBusy(true);
        const voice = el.voice.value;
        if (!voice) throw new Error('Nenhuma voz pronta está disponível.');
        await synthesize(currentPayload(voice), el.voice.selectedOptions[0]?.textContent || voice);
      } catch (error) {
        setError(error instanceof Error ? error.message : 'Erro desconhecido durante a síntese.');
      } finally { setBusy(false); }
    }
    async function compare() {
      try {
        setError(''); clearResults();
        if (!hasComparisonVoices()) throw new Error('As vozes giseli-10s e giseli-34s precisam estar disponíveis para comparar.');
        // Capture once so both voices always use exactly the same text, seed,
        // and generation parameters, even if somebody edits a field mid-run.
        const shortPayload = currentPayload('giseli-10s');
        const longPayload = {...shortPayload, voice: 'giseli-34s'};
        setBusy(true, 'Gerando 10s…');
        // Intentional sequential calls: the service has a single CUDA inference queue.
        await synthesize(shortPayload, 'Giseli — referência de 10 segundos');
        setBusy(true, 'Gerando 34s…');
        await synthesize(longPayload, 'Giseli — referência de 34 segundos');
      } catch (error) {
        setError(error instanceof Error ? error.message : 'Erro desconhecido durante a comparação.');
      } finally { setBusy(false); }
    }
    async function loadVoices() {
      try {
        const response = await fetch('/v1/voices');
        if (!response.ok) throw new Error(await responseError(response));
        const data = await response.json();
        defaultVoice = data.default_voice || null;
        el.voice.replaceChildren();
        for (const voice of (data.voices || [])) {
          const option = document.createElement('option'); option.value = voice.id;
          option.textContent = `${voice.display_name || voice.id}${voice.id === defaultVoice ? ' (padrão)' : ''}`;
          option.disabled = voice.ready === false;
          if (voice.id === defaultVoice) option.selected = true;
          el.voice.append(option);
        }
        el.defaultVoice.textContent = defaultVoice ? `(padrão: ${defaultVoice})` : '';
        const ready = (data.voices || []).filter((voice) => voice.ready).length;
        el.voiceHint.textContent = ready ? `${ready} voz(es) pronta(s).` : 'Nenhuma voz está pronta; aguarde o carregamento.';
      } catch (error) {
        el.voiceHint.textContent = 'Não foi possível carregar as vozes.';
      } finally { setBusy(false); }
    }
    async function loadHealth() {
      try {
        const response = await fetch('/health');
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        const status = data.status || 'unknown';
        el.status.textContent = status;
        el.statusDot.className = `dot ${status === 'ready' ? 'ready' : (status === 'error' || status === 'degraded' ? 'error' : 'loading')}`;
        setList(el.serviceMeta, [
          ['Modelo', data.model], ['Carregado', data.model_loaded ? 'sim' : 'não'], ['Dispositivo', data.device],
          ['Dtype', data.dtype], ['Atenção', data.attention], ['Carga', data.load_seconds === undefined ? null : `${data.load_seconds} s`]
        ]);
        const gpu = data.gpu || {};
        const gpuEntries = Object.keys(gpu).length ? Object.entries(gpu).map(([key, value]) => [key, typeof value === 'object' ? JSON.stringify(value) : value]) : [['Estado', data.cuda_available === false ? 'CUDA indisponível' : 'Sem métricas disponíveis']];
        setList(el.gpuMeta, gpuEntries);
      } catch (_) {
        el.status.textContent = 'indisponível'; el.statusDot.className = 'dot error';
        setList(el.serviceMeta, [['Estado', 'Não foi possível consultar /health.']]);
      }
    }
    el.apiKey.value = window.sessionStorage.getItem('aegis-tts-api-key') || '';
    el.apiKey.addEventListener('input', () => window.sessionStorage.setItem('aegis-tts-api-key', el.apiKey.value));
    el.text.addEventListener('input', updateCount);
    el.generate.addEventListener('click', generate);
    el.compare.addEventListener('click', compare);
    updateCount(); setBusy(true);
    Promise.all([loadHealth(), loadVoices()]).then(() => loadHealth());
    window.setInterval(loadHealth, 5000);
    window.addEventListener('beforeunload', () => activeUrls.forEach((url) => URL.revokeObjectURL(url)));
  })();
  </script>
</body>
</html>
"""


def render_ui() -> str:
    """Return the static, dependency-free manual test page."""

    return HTML_PAGE

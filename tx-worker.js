/* Speech-to-text worker.
   Loads Whisper through Transformers.js and transcribes 16 kHz mono PCM.
   Everything runs on this device; nothing is uploaded. */

let asr = null;
let loading = null;

async function load(model, device, dtype) {
  const { pipeline, env } = await import(
    'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js'
  );
  env.allowLocalModels = false;
  return pipeline('automatic-speech-recognition', model, {
    dtype: dtype || 'q8',
    device: device,
    progress_callback: (p) => {
      if (p && p.status === 'progress' && p.total) {
        self.postMessage({ type: 'progress', pct: Math.round((p.loaded / p.total) * 100), file: p.file });
      } else if (p && p.status) {
        self.postMessage({ type: 'stage', stage: p.status, file: p.file });
      }
    }
  });
}

self.onmessage = async (e) => {
  const m = e.data || {};

  if (m.type === 'load') {
    if (asr) { self.postMessage({ type: 'ready', device: 'cached' }); return; }
    if (loading) return;
    loading = (async () => {
      /* Try the GPU path when the browser has one, fall back to WASM on any failure. */
      const order = (typeof navigator !== 'undefined' && navigator.gpu) ? ['webgpu', 'wasm'] : ['wasm'];
      let lastErr = null;
      for (const dev of order) {
        try {
          asr = await load(m.model, dev, m.dtype);
          self.postMessage({ type: 'ready', device: dev });
          return;
        } catch (err) {
          lastErr = err;
          self.postMessage({ type: 'stage', stage: dev + ' failed, trying fallback' });
        }
      }
      self.postMessage({ type: 'error', where: 'load', msg: String((lastErr && lastErr.message) || lastErr) });
    })();
    await loading;
    loading = null;
    return;
  }

  if (m.type === 'run') {
    if (!asr) { self.postMessage({ type: 'error', where: 'run', id: m.id, msg: 'model not loaded' }); return; }
    try {
      /* English-only (.en) checkpoints reject language/task, so only send them
         to multilingual models. Everything else guards against runaway loops. */
      const opts = {
        return_timestamps: false,
        max_new_tokens: 48,
        no_repeat_ngram_size: 3,
        repetition_penalty: 1.2,
        num_beams: 1,
        do_sample: false
      };
      if (!/\.en$/.test(String(m.model || ''))) { opts.language = 'en'; opts.task = 'transcribe'; }

      let out;
      try {
        out = await asr(m.audio, opts);
      } catch (inner) {
        /* An option this build of the library dislikes must not kill the feature.
           Retry once with nothing but the bare minimum. */
        self.postMessage({ type: 'stage', stage: 'retrying without options: ' + ((inner && inner.message) || inner) });
        out = await asr(m.audio, { return_timestamps: false });
      }
      const text = ((out && out.text) || '').trim();
      self.postMessage({ type: 'result', id: m.id, text: text });
    } catch (err) {
      self.postMessage({ type: 'error', where: 'run', id: m.id, msg: String((err && err.message) || err) });
    }
  }
};

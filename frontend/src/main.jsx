import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function predict(event) {
    event.preventDefault(); setBusy(true); setError(''); setResult(null);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch('/api/predict', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), signal: controller.signal });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check your text and try again.');
      setResult(data);
    } catch (e) { setError(e.name === 'AbortError' ? 'The request timed out. Please try again.' : e.message); }
    finally { clearTimeout(timeout); setBusy(false); }
  }
  return <main>
    <nav><a href="#" className="brand"><span className="mark">N</span> NITW<span className="muted"> / Dell</span></a><span className="tag">INTERVIEW PROJECT · REVAMPED</span></nav>
    <header><div className="eyebrow">FROM AN IDEA TO INFERENCE</div><h1>Your words.<br/><span>A new perspective.</span></h1><p>A Dell interview project, reimagined as a scalable text classification experience. Explore patterns in writing with a 16-class Keras model.</p></header>
    <section className="workspace" aria-label="Text classifier">
      <div className="card-heading"><div><span className="eyebrow">TRY THE MODEL</span><h2>What’s on your mind?</h2></div><span className="pill">16 personality types</span></div>
      <form onSubmit={predict}><label htmlFor="text">Your writing sample</label><textarea id="text" value={text} onChange={e => { setText(e.target.value); setResult(null); setError(''); }} maxLength={10000} minLength={3} required disabled={busy} placeholder="Write about how you approach a new challenge, spend your free time, or make a decision…"/><div className="form-footer"><span>{text.length.toLocaleString()} / 10,000 characters</span><button disabled={busy || !text.trim()}>{busy ? 'Analyzing…' : 'Explore my text ↗'}</button></div></form>
      <div aria-live="polite" aria-busy={busy}>{error && <p className="error" role="alert">{error}</p>}{busy && <p className="loading">Finding patterns in your words…</p>}{result && <div className="result"><div className="result-top"><div><span className="eyebrow">PREDICTED CLASS</span><h3>{result.label}</h3></div><div className="confidence">{(result.confidence * 100).toFixed(1)}%<small>model probability</small></div></div><p className="muted">The closest matches in this writing sample</p>{result.scores.slice(0, 3).map(score => <div className="score" key={score.label}><span>{score.label}</span><div className="track"><div style={{ width: `${score.probability * 100}%` }}/></div><span>{(score.probability * 100).toFixed(1)}%</span></div>)}<div className="result-meta">Inference {result.latency_ms.toFixed(0)} ms · Model {result.model_version}</div></div>}</div>
      <p className="note">An educational experiment using MBTI-labeled forum posts. Model probabilities are not calibrated confidence or a psychological assessment. Longer writing samples better resemble the training data.</p>
    </section>
    <footer><span>BUILT TO LEARN. DESIGNED TO SCALE.</span><span>React · FastAPI · Keras · Kubernetes</span></footer>
  </main>;
}
createRoot(document.getElementById('root')).render(<App/>);

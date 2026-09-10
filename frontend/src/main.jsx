import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "";

function App() {
  const [page, setPage] = useState("home");
  const [query, setQuery] = useState("");
  const [images, setImages] = useState([]);
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const go = (next) => {
    setPage(next);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  async function submitQuery(event) {
    event.preventDefault();
    setError("");
    if (!images.length || !query.trim()) {
      setError("Add at least one image and a question before running the analysis.");
      return;
    }
    setBusy(true);
    try {
      if (!API_URL) {
        throw new Error("The original Python/Gradio application is the active analysis service. Start it with `python app.py`.");
      }
      const form = new FormData();
      form.append("query", query.trim());
      images.forEach((image) => form.append("images", image));
      const response = await fetch(API_URL, { method: "POST", body: form });
      if (!response.ok) throw new Error(`Analysis request failed (${response.status}).`);
      const data = await response.json();
      setAnswer(data.answer || data.error || "The service returned no answer.");
    } catch (requestError) {
      setError(`${requestError.message} The React interface is presentation-only until an API endpoint is configured with VITE_API_URL.`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => go("home")} aria-label="SatQuery home">
          <span className="brand-mark">SQ</span>
          <span>SatQuery</span>
        </button>
        <nav aria-label="Main navigation">
          {[
            ["console", "Query Console"],
            ["evaluation", "Evaluation"],
            ["about", "About This Build"],
          ].map(([id, label]) => (
            <button className={page === id ? "nav-link active" : "nav-link"} key={id} onClick={() => go(id)}>
              {label}
            </button>
          ))}
        </nav>
        <span className="link-state"><i /> Local analysis ready</span>
      </header>

      {page === "home" && <Landing onStart={() => go("console")} />}
      {page === "console" && (
        <main className="page narrow">
          <SectionIntro eyebrow="Query console" title="Ask clear questions about Earth observation imagery."
            copy="Upload one image for description, or two aligned images to examine change. Answers stay grounded in the selected imagery." />
          <form className="console-grid" onSubmit={submitQuery}>
            <section className="panel input-panel">
              <div className="panel-heading"><span className="eyebrow">01 / imagery</span><span className="mono">{images.length}/2 loaded</span></div>
              <label className="dropzone">
                <input type="file" accept="image/*,.tif,.tiff" multiple onChange={(e) => setImages(Array.from(e.target.files || []).slice(0, 2))} />
                <strong>Drop satellite images here</strong>
                <span>PNG, JPEG, or GeoTIFF · one or two files</span>
              </label>
              {images.map((image) => <div className="file-row" key={image.name}><span className="file-dot" />{image.name}<span className="mono">{Math.round(image.size / 1024)} KB</span></div>)}
              <label className="field-label" htmlFor="query">Natural-language question</label>
              <textarea id="query" value={query} onChange={(e) => setQuery(e.target.value)}
                placeholder="What changed between these two dates, and where did the change occur?" rows="5" />
              {error && <p className="error" role="alert">{error}</p>}
              <button className="button primary" disabled={busy}>{busy ? "Running analysis..." : "Run analysis"}</button>
            </section>
            <section className="panel answer-panel">
              <div className="panel-heading"><span className="eyebrow">02 / response</span><span className="status"><i /> specialist pipeline</span></div>
              {answer ? <div className="answer">{answer}</div> : <div className="empty-state"><div className="crosshair">+</div><strong>Your analysis will appear here</strong><span>Upload imagery and ask a question to begin.</span></div>}
            </section>
          </form>
        </main>
      )}
      {page === "evaluation" && <Evaluation />}
      {page === "about" && <About />}
      <footer><span>SatQuery</span><span>Earth observation analysis for informed decisions.</span><span className="mono">v1.0 / LOCAL PIPELINE</span></footer>
    </div>
  );
}

function Landing({ onStart }) {
  return <main className="landing page">
    <div className="hero-copy">
      <span className="eyebrow">Earth observation intelligence</span>
      <h1>Turn satellite imagery into a clear next step.</h1>
      <p>SatQuery helps you describe land cover, compare dates, locate features, and combine optical and SAR imagery without needing a remote-sensing background.</p>
      <div className="hero-actions"><button className="button primary" onClick={onStart}>Open query console</button><button className="button secondary" onClick={() => document.getElementById("capabilities").scrollIntoView({ behavior: "smooth" })}>View capabilities</button></div>
    </div>
    <div className="hero-visual" aria-label="Satellite analysis preview"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="earth-grid"><span>LAND COVER</span><strong>ACTIVE</strong><small className="mono">OPTICAL / SAR / CHANGE</small></div></div>
    <section id="capabilities" className="capabilities"><SectionIntro eyebrow="Capabilities" title="One console for the questions that matter." copy="The interface keeps the workflow simple while the backend selects the right specialist." /><div className="capability-grid">{[["01", "Describe", "Summarize visible land cover and major objects in plain language."], ["02", "Compare", "Identify meaningful changes between two dates and explain where they occur."], ["03", "Locate", "Highlight water, vegetation, built-up areas, or bare soil on the image."], ["04", "Combine", "Use optical and SAR inputs together for a more informed scene view."]].map(([n, t, c]) => <article className="capability" key={n}><span className="mono">{n}</span><h3>{t}</h3><p>{c}</p></article>)}</div></section>
  </main>;
}

function SectionIntro({ eyebrow, title, copy }) { return <div className="section-intro"><span className="eyebrow">{eyebrow}</span><h2>{title}</h2><p>{copy}</p></div>; }
function Evaluation() { return <main className="page narrow"><SectionIntro eyebrow="Evaluation" title="Understand what is running." copy="Use this space to review the active analysis paths before you interpret a result." /><div className="evaluation-grid">{[["Nominal", "Local specialist pipeline", "The default path runs without external model downloads."], ["Optional", "Qwen3-VL adapter", "Available for single-image VQA and captions when configured."], ["Evidence", "Visual overlays", "Responses can include an evidence image for spatial context."]].map(([tag, title, copy]) => <article className="panel info-card" key={title}><span className="status"><i /> {tag}</span><h3>{title}</h3><p>{copy}</p></article>)}</div></main>; }
function About() { return <main className="page narrow"><SectionIntro eyebrow="About this build" title="Designed for careful, explainable analysis." copy="SatQuery combines transparent spectral rules with optional learned models. It separates image understanding from the specialist that performs the task." /><div className="about-layout"><div className="panel"><span className="eyebrow">System principles</span><ul className="principles"><li><strong>Plain language</strong><span>Results are written for people who are new to machine learning.</span></li><li><strong>Auditable work</strong><span>Each request records the selected task and analysis path.</span></li><li><strong>Honest limits</strong><span>RGB proxies and model confidence are labeled instead of hidden.</span></li></ul></div><div className="panel"><span className="eyebrow">Supported workflows</span><div className="tag-list"><span>Captioning</span><span>VQA</span><span>Grounding</span><span>Change analysis</span><span>Optical / SAR fusion</span><span>Land cover</span></div></div></div></main>; }

createRoot(document.getElementById("root")).render(<App />);

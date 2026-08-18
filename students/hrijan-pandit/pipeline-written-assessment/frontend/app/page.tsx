"use client";

import { useState, useRef, useCallback } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────
interface GradeResult {
  transcription: string;
  assessment: string;
  score: string | null;
  feedback: string;
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function Home() {
  const [file, setFile]         = useState<File | null>(null);
  const [preview, setPreview]   = useState<string | null>(null);
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState<GradeResult | null>(null);
  const [error, setError]       = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);

  const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? "";

  // ── File handling ──────────────────────────────────────────────────────────
  const applyFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file (JPG, PNG, WEBP, PDF scan, etc.).");
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) applyFile(dropped);
  }, [applyFile]);

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  // ── Submit ─────────────────────────────────────────────────────────────────
  const grade = async () => {
    if (!file || !BACKEND) return;
    setLoading(true);
    setResult(null);
    setError(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${BACKEND}/grade`, { method: "POST", body: form });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail?.detail ?? `Server error ${res.status}`);
      }
      const data: GradeResult = await res.json();
      setResult(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="page">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-logo">📝</div>
        <span className="header-title">AnswerLens</span>
        <span className="header-tag">AI Assessment</span>
      </header>

      {/* ── Workspace ── */}
      <main className="workspace">

        {/* ── LEFT: upload panel ── */}
        <aside className="left-panel">
          <span className="panel-label">Upload Answer Sheet</span>

          {!preview ? (
            <div
              className={`drop-zone ${dragging ? "drag-over" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <div className="drop-zone-icon">🖼️</div>
              <p className="drop-zone-text">
                <strong>Click to browse</strong> or drag & drop<br />
                JPG, PNG, WEBP — up to 20 MB
              </p>
            </div>
          ) : (
            <div className="preview-wrap">
              <img src={preview} alt="Answer sheet preview" />
              <button className="preview-clear" onClick={clearFile} title="Remove image">✕</button>
            </div>
          )}

          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) applyFile(f); }}
          />

          {error && <div className="error-banner">⚠️ {error}</div>}

          <button
            className="btn-grade"
            disabled={!file || loading}
            onClick={grade}
          >
            {loading ? "Analysing…" : "Grade Answer Sheet"}
          </button>

          {BACKEND === "" && (
            <p style={{ fontSize: 12, color: "var(--danger)" }}>
              ⚠️ <code>NEXT_PUBLIC_BACKEND_URL</code> is not set. Add it to <code>.env.local</code>.
            </p>
          )}
        </aside>

        {/* ── RIGHT: results panel ── */}
        <section className="right-panel">
          {loading && (
            <div className="spinner-wrap">
              <div className="spinner" />
              <span className="spinner-label">Reading handwriting and generating assessment…</span>
            </div>
          )}

          {!loading && !result && (
            <div className="empty-state">
              <div className="empty-icon">🔍</div>
              <p className="empty-title">Results will appear here</p>
              <p style={{ fontSize: 13 }}>Upload an answer sheet and click <strong>Grade</strong>.</p>
            </div>
          )}

          {!loading && result && (
            <div className="result-grid">

              {/* Transcription */}
              <div className="card">
                <div className="card-head">
                  <span className="card-head-icon">📄</span>
                  <span className="card-head-title">Transcription</span>
                </div>
                <div className="card-body">
                  <pre className="transcription-text">{result.transcription}</pre>
                </div>
              </div>

              {/* Score */}
              {result.score && (
                <div className="card">
                  <div className="card-head">
                    <span className="card-head-icon">🏅</span>
                    <span className="card-head-title">Score</span>
                  </div>
                  <div className="card-body">
                    <div className="score-row">
                      <span className="score-badge">{result.score.split(" ")[0]}</span>
                      <span className="score-detail">
                        {result.score.split(" ").slice(1).join(" ")}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Assessment */}
              <div className="card">
                <div className="card-head">
                  <span className="card-head-icon">📊</span>
                  <span className="card-head-title">Assessment</span>
                </div>
                <div className="card-body">
                  <p className="assessment-text">{result.assessment}</p>
                </div>
              </div>

              {/* Feedback */}
              {result.feedback && (
                <div className="card">
                  <div className="card-head">
                    <span className="card-head-icon">💬</span>
                    <span className="card-head-title">Feedback for Student</span>
                  </div>
                  <div className="card-body">
                    <p className="feedback-text">{result.feedback}</p>
                  </div>
                </div>
              )}

            </div>
          )}
        </section>
      </main>
    </div>
  );
}

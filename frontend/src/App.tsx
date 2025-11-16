import React, { useState } from "react";

type Message = {
  role: "doctor" | "patient";
  content: string;
};

type BehavioralLinguisticProfile = Record<string, unknown>;
type PatientProfile = Record<string, unknown>;

type AlignmentScore = {
  score: number;
  label: string;
  explanation: string;
};

type TurnCritique = {
  turn_index: number;
  doctor_utterance: string;
  patient_reply: string;
  notes: string;
  issues: string[];
};

type CritiqueResult = {
  overall_comment: string;
  clinical_alignment: AlignmentScore;
  persona_alignment: AlignmentScore;
  safety_flags: string[];
  suggested_improvements: string[];
  per_turn: TurnCritique[];
};

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const App: React.FC = () => {
  const [rawTranscript, setRawTranscript] = useState("");
  const [rawCase, setRawCase] = useState("");
  const [doctorInput, setDoctorInput] = useState("");
  const [conversation, setConversation] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [blpReady, setBlpReady] = useState(false);
  const [patientReady, setPatientReady] = useState(false);

  const [blpObj, setBlpObj] = useState<BehavioralLinguisticProfile | null>(
    null,
  );
  const [patientObj, setPatientObj] = useState<PatientProfile | null>(null);

  const [busyBLP, setBusyBLP] = useState(false);
  const [blpAbortController, setBlpAbortController] = useState<AbortController | null>(null);
  const [busyProfile, setBusyProfile] = useState(false);
  const [patientAbortController, setPatientAbortController] = useState<AbortController | null>(null);
  const [busyTurn, setBusyTurn] = useState(false);
  const [busyCritique, setBusyCritique] = useState(false);
  const [busyOptimize, setBusyOptimize] = useState(false);
  const [optimizeJobId, setOptimizeJobId] = useState<string | null>(null);
  const [optimizePercent, setOptimizePercent] = useState<number>(0);
  const [optimizeStatus, setOptimizeStatus] = useState<string | null>(null);
  const [optimizeTimer, setOptimizeTimer] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [critique, setCritique] = useState<CritiqueResult | null>(null);
  const [originalPrompt, setOriginalPrompt] = useState<string | null>(null);
  const [optimizedPrompt, setOptimizedPrompt] = useState<string | null>(null);
  const [busyReset, setBusyReset] = useState(false);

  // Frontend-adjustable models and token budgets
  const [blpModel, setBlpModel] = useState("gpt-4.1-mini");
  const [blpMaxTokens, setBlpMaxTokens] = useState<number>(2048);
  const [patientModel, setPatientModel] = useState("gpt-4.1-mini");
  const [patientMaxTokens, setPatientMaxTokens] = useState<number>(1024);
  const [critiqueModel, setCritiqueModel] = useState("gpt-4.1-mini");
  const [critiqueMaxTokens, setCritiqueMaxTokens] = useState<number>(2048);
  // GEPA knobs
  const [gepaModel, setGepaModel] = useState("gpt-4.1-mini");
  const [gepaReflectionMinibatchSize, setGepaReflectionMinibatchSize] = useState<number>(1);
  const [gepaCandidateSelection, setGepaCandidateSelection] = useState("pareto");
  const [gepaMaxMetricCalls, setGepaMaxMetricCalls] = useState<number>(5);
  const [gepaUseMerge, setGepaUseMerge] = useState<boolean>(false);
  const [gepaTrackStats, setGepaTrackStats] = useState<boolean>(true);

  const handleExtractBLP = async () => {
    if (!rawTranscript.trim()) return;
    setBusyBLP(true);
    setError(null);
    const controller = new AbortController();
    setBlpAbortController(controller);

    try {
      const res = await fetch(`${API_BASE}/api/blp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: rawTranscript,
          model: blpModel,
          max_tokens: blpMaxTokens,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`BLP API error (${res.status}): ${text}`);
      }

      const data: {
        anonymized_transcript: string;
        blp: BehavioralLinguisticProfile;
      } = await res.json();

      setBlpObj(data.blp);
      setBlpReady(true);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        setError(null);
        return;
      }
      setError(
        e instanceof Error ? e.message : "Failed to extract BLP from backend.",
      );
    } finally {
      setBusyBLP(false);
      setBlpAbortController(null);
    }
  };

  const handleCancelBLP = () => {
    try {
      blpAbortController?.abort();
    } catch {
      // ignore
    } finally {
      setBusyBLP(false);
      setBlpAbortController(null);
    }
  };

  const handleEndAndOptimize = async () => {
    if (!rawTranscript.trim() || !rawCase.trim()) {
      setError("Raw transcript and raw case are required to optimize.");
      return;
    }
    if (conversation.length === 0) {
      setError("Have at least one exchange before optimizing.");
      return;
    }
    setBusyOptimize(true);
    setError(null);
    setOriginalPrompt(null);
    setOptimizedPrompt(null);
    setOptimizePercent(0);
    setOptimizeStatus("queued");

    try {
      const payload = {
        raw_transcript: rawTranscript,
        raw_case: rawCase,
        conversation: conversation.map((m) => ({
          role: m.role,
          content: m.content,
        })),
        // GEPA knobs from UI
        reflection_minibatch_size: gepaReflectionMinibatchSize,
        candidate_selection_strategy: gepaCandidateSelection,
        max_metric_calls: gepaMaxMetricCalls,
        use_merge: gepaUseMerge,
        track_stats: gepaTrackStats,
        model: gepaModel,
      };
      const res = await fetch(`${API_BASE}/api/optimize/blp-prompt/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Optimize start error (${res.status}): ${text}`);
      }
      const data: { job_id: string } = await res.json();
      setOptimizeJobId(data.job_id);
      // begin polling
      const id = window.setInterval(async () => {
        try {
          const pRes = await fetch(
            `${API_BASE}/api/optimize/blp-prompt/progress/${encodeURIComponent(
              data.job_id,
            )}`,
          );
          if (!pRes.ok) {
            if (pRes.status === 404) {
              window.clearInterval(id);
              setOptimizeTimer(null);
              setBusyOptimize(false);
              setOptimizeStatus("error");
            }
            return;
          }
          const pData: {
            status: string;
            percent: number;
            metric_calls?: number | null;
            max_metric_calls?: number | null;
            latest_score?: number | null;
            best_score?: number | null;
          } = await pRes.json();
          setOptimizeStatus(pData.status);
          setOptimizePercent(pData.percent ?? 0);
          if (pData.status === "cancelled" || pData.status === "cancelling") {
            window.clearInterval(id);
            setOptimizeTimer(null);
            setBusyOptimize(false);
            setOptimizeJobId(null);
            return;
          }
          if (pData.status === "complete") {
            const rRes = await fetch(
              `${API_BASE}/api/optimize/blp-prompt/result/${encodeURIComponent(
                data.job_id,
              )}`,
            );
            if (rRes.ok) {
              const rData: { original_prompt: string; optimized_prompt: string } =
                await rRes.json();
              setOriginalPrompt(rData.original_prompt);
              setOptimizedPrompt(rData.optimized_prompt);
            }
            window.clearInterval(id);
            setOptimizeTimer(null);
            setBusyOptimize(false);
            setOptimizeJobId(null);
          }
          if (pData.status === "error") {
            window.clearInterval(id);
            setOptimizeTimer(null);
            setBusyOptimize(false);
            setOptimizeJobId(null);
            setError("Optimization job failed. Check backend logs for details.");
          }
        } catch {
          // ignore transient errors
        }
      }, 1000);
      setOptimizeTimer(id);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Failed to optimize BLP prompt.",
      );
      setBusyOptimize(false);
      setOptimizeJobId(null);
    } finally {
      // busy flag cleared on completion by poller
    }
  };

  const handleCancelOptimize = async () => {
    try {
      if (optimizeTimer !== null) {
        window.clearInterval(optimizeTimer);
        setOptimizeTimer(null);
      }
      if (optimizeJobId) {
        await fetch(
          `${API_BASE}/api/optimize/blp-prompt/cancel/${encodeURIComponent(
            optimizeJobId,
          )}`,
          { method: "POST" },
        );
      }
    } catch {
      // best-effort cancel
    } finally {
      setBusyOptimize(false);
      setOptimizeJobId(null);
      setOptimizeStatus("cancelled");
    }
  };

  const handleBuildPatientProfile = async () => {
    if (!rawCase.trim()) return;
    setBusyProfile(true);
    setError(null);
    const controller = new AbortController();
    setPatientAbortController(controller);

    try {
      const res = await fetch(`${API_BASE}/api/patient-profile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          raw_case: rawCase,
          model: patientModel,
          max_tokens: patientMaxTokens,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Patient profile API error (${res.status}): ${text}`);
      }

      const data: { patient_profile: PatientProfile } = await res.json();
      setPatientObj(data.patient_profile);
      setPatientReady(true);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        setError(null);
        return;
      }
      setError(
        e instanceof Error
          ? e.message
          : "Failed to build patient profile from backend.",
      );
    } finally {
      setBusyProfile(false);
      setPatientAbortController(null);
    }
  };

  const handleCancelPatientProfile = () => {
    try {
      patientAbortController?.abort();
    } catch {
      // ignore
    } finally {
      setBusyProfile(false);
      setPatientAbortController(null);
    }
  };

  const ensureSession = async (): Promise<string> => {
    if (sessionId) return sessionId;
    if (!blpObj || !patientObj) {
      throw new Error("You need a BLP and Patient Profile before starting a session.");
    }

    const res = await fetch(`${API_BASE}/api/simulated-patient/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        blp: blpObj,
        patient_profile: patientObj,
      }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Session init error (${res.status}): ${text}`);
    }

    const data: { session_id: string } = await res.json();
    setSessionId(data.session_id);
    return data.session_id;
  };

  const handleSendTurn = async () => {
    const text = doctorInput.trim();
    if (!text) return;

    setBusyTurn(true);
    setError(null);

    const doctorMsg: Message = { role: "doctor", content: text };
    setConversation((prev) => [...prev, doctorMsg]);
    setDoctorInput("");

    try {
      const sid = await ensureSession();

      const res = await fetch(
        `${API_BASE}/api/simulated-patient/${encodeURIComponent(sid)}/turn`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ doctor_utterance: text }),
        },
      );

      if (!res.ok) {
        const t = await res.text();
        throw new Error(`Turn API error (${res.status}): ${t}`);
      }

      const data: { patient_reply: string } = await res.json();
      const patientMsg: Message = { role: "patient", content: data.patient_reply };
      setConversation((prev) => [...prev, patientMsg]);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Failed to get patient reply from backend.",
      );
    } finally {
      setBusyTurn(false);
    }
  };

  const handleResetChat = async () => {
    setError(null);
    setBusyReset(true);
    try {
      if (sessionId) {
        const res = await fetch(
          `${API_BASE}/api/simulated-patient/${encodeURIComponent(
            sessionId,
          )}/reset`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
          },
        );
        if (!res.ok) {
          const t = await res.text();
          throw new Error(`Reset error (${res.status}): ${t}`);
        }
      }
      setConversation([]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to reset chat.");
    } finally {
      setBusyReset(false);
    }
  };

  const handleRunCritique = async () => {
    if (!blpObj || !patientObj) {
      setError("You need a BLP and Patient Profile before running a critique.");
      return;
    }

    if (!rawTranscript.trim() || !rawCase.trim()) {
      setError("Raw transcript and raw case are required for the critique agent.");
      return;
    }

    if (conversation.length === 0) {
      setError("You need at least one doctor–patient exchange before running a critique.");
      return;
    }

    setBusyCritique(true);
    setError(null);

    try {
      const payload = {
        blp: blpObj,
        patient_profile: patientObj,
        raw_transcript: rawTranscript,
        raw_case: rawCase,
        conversation: conversation.map((msg) => ({
          role: msg.role,
          content: msg.content,
        })),
        model: critiqueModel,
        max_tokens: critiqueMaxTokens,
      };

      const res = await fetch(`${API_BASE}/api/critique`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(`Critique API error (${res.status}): ${t}`);
      }

      const data: { critique: CritiqueResult } = await res.json();
      setCritique(data.critique);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Failed to get critique from backend.",
      );
    } finally {
      setBusyCritique(false);
    }
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <div>
          <h1>Persona Generation Lab</h1>
          <p>
            Prototype UI that mirrors your architecture: transcript → BLP,
            case → patient profile, both feeding a simulated patient.
          </p>
        </div>
        <span className="badge">Node · React · Vite</span>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className="pipeline-grid">
        {/* Top-left: Transcript → BLP */}
        <section className="panel">
          <h2>Behavioral &amp; Linguistic Profile</h2>
          <p className="panel-subtitle">
            Raw interview transcript &rarr; anonymized transcript &rarr; BLP.
          </p>

          <label className="field-label" htmlFor="transcript">
            Raw interview transcript
          </label>
          <textarea
            id="transcript"
            className="field-textarea"
            placeholder="Paste the raw transcript here..."
            value={rawTranscript}
            onChange={(e) => {
              setRawTranscript(e.target.value);
              setBlpReady(false);
            }}
          />
          <div className="field-row">
            <div className="field-column">
              <label className="field-label" htmlFor="blp-model">
                BLP model
              </label>
              <input
                id="blp-model"
                className="conversation-input"
                placeholder="e.g. gpt-4.1-mini"
                value={blpModel}
                onChange={(e) => setBlpModel(e.target.value)}
              />
            </div>
            <div className="field-column">
              <label className="field-label" htmlFor="blp-max-tokens">
                BLP max tokens
              </label>
              <input
                id="blp-max-tokens"
                type="number"
                className="conversation-input"
                placeholder="2048"
                value={blpMaxTokens}
                onChange={(e) => setBlpMaxTokens(Number(e.target.value) || 0)}
              />
            </div>
          </div>

          <button
            className="primary-button"
            onClick={handleExtractBLP}
            disabled={busyBLP}
          >
            {busyBLP ? "Processing..." : "Anonymize & Extract BLP"}
          </button>
          {busyBLP && (
            <button
              className="secondary-button"
              onClick={handleCancelBLP}
              style={{ marginLeft: 8 }}
              disabled={!blpAbortController}
            >
              Stop
            </button>
          )}
          {blpReady && (
            <span className="badge" style={{ marginLeft: 8 }}>BLP ready</span>
          )}

        </section>

        {/* Bottom-left: Raw case → Patient profile */}
        <section className="panel">
          <h2>Patient Profile</h2>
          <p className="panel-subtitle">
            Raw case (structured + unstructured) &rarr; structured patient
            profile.
          </p>

          <label className="field-label" htmlFor="case">
            Raw case description
          </label>
          <textarea
            id="case"
            className="field-textarea"
            placeholder="Paste the case description or structured data here..."
            value={rawCase}
            onChange={(e) => {
              setRawCase(e.target.value);
              setPatientReady(false);
            }}
          />
          <div className="field-row">
            <div className="field-column">
              <label className="field-label" htmlFor="patient-model">
                Patient profile model
              </label>
              <input
                id="patient-model"
                className="conversation-input"
                placeholder="e.g. gpt-4.1-mini"
                value={patientModel}
                onChange={(e) => setPatientModel(e.target.value)}
              />
            </div>
            <div className="field-column">
              <label className="field-label" htmlFor="patient-max-tokens">
                Patient profile max tokens
              </label>
              <input
                id="patient-max-tokens"
                type="number"
                className="conversation-input"
                placeholder="1024"
                value={patientMaxTokens}
                onChange={(e) => setPatientMaxTokens(Number(e.target.value) || 0)}
              />
            </div>
          </div>

          <button
            className="primary-button"
            onClick={handleBuildPatientProfile}
            disabled={busyProfile}
          >
            {busyProfile ? "Structuring..." : "Structure data & build profile"}
          </button>
          {busyProfile && (
            <button
              className="secondary-button"
              onClick={handleCancelPatientProfile}
              style={{ marginLeft: 8 }}
              disabled={!patientAbortController}
            >
              Stop
            </button>
          )}
          {patientReady && (
            <span className="badge" style={{ marginLeft: 8 }}>Patient profile ready</span>
          )}

        </section>

        {/* Right: Simulated patient conversation */}
        <section className="panel panel-span">
          <h2>Simulated Patient</h2>
          <p className="panel-subtitle">
            BLP + Patient profile + constraints &rarr; in-role patient behavior.
          </p>

          <div className="conversation-card">
            <div className="conversation-header">
              <div>
                <div className="conversation-title">Training encounter</div>
                <div className="conversation-caption">
                  Talk as the doctor. The agent stays maximally faithful to the
                  profiles you provide.
                </div>
              </div>
              <span className="badge badge-soft">
                {sessionId ? "Session active" : "Awaiting profiles"}
              </span>
              <button
                className="secondary-button"
                onClick={handleResetChat}
                disabled={busyReset || busyTurn}
                style={{ marginLeft: 12 }}
              >
                {busyReset ? "Resetting..." : "Reset chat"}
              </button>
            </div>

            <div className="conversation-body">
              {conversation.length === 0 ? (
                <div className="placeholder">
                  Start typing as the clinician. Once the backend is wired up,
                  you&apos;ll see realistic patient turns here driven by the BLP
                  and patient profile.
                </div>
              ) : (
                conversation.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`bubble bubble-${msg.role}`}
                  >
                    <div className="bubble-role">
                      {msg.role === "doctor" ? "Doctor" : "Patient"}
                    </div>
                    <div className="bubble-text">{msg.content}</div>
                  </div>
                ))
              )}
            </div>

            <div className="conversation-input-row">
              <input
                type="text"
                className="conversation-input"
                placeholder="Ask the patient something as the doctor..."
                value={doctorInput}
                onChange={(e) => setDoctorInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendTurn();
                  }
                }}
              />
              <button
                className="primary-button"
                onClick={handleSendTurn}
                disabled={busyTurn}
              >
                {busyTurn ? "Thinking..." : "Send"}
              </button>
            </div>
          </div>

          <div className="critique-controls">
            <div className="field-row" style={{ marginBottom: 8 }}>
              <div className="field-column">
                <label className="field-label" htmlFor="gepa-model">
                  GEPA model
                </label>
                <input
                  id="gepa-model"
                  className="conversation-input"
                  placeholder="e.g. gpt-4.1-mini"
                  value={gepaModel}
                  onChange={(e) => setGepaModel(e.target.value)}
                />
              </div>
              <div className="field-column">
                <label className="field-label" htmlFor="gepa-reflection-minibatch">
                  Reflection minibatch size
                </label>
                <input
                  id="gepa-reflection-minibatch"
                  type="number"
                  className="conversation-input"
                  placeholder="e.g. 3"
                  value={gepaReflectionMinibatchSize}
                  onChange={(e) =>
                    setGepaReflectionMinibatchSize(Number(e.target.value) || 0)
                  }
                />
              </div>
              <div className="field-column">
                <label className="field-label" htmlFor="gepa-max-metric-calls">
                  Max metric calls
                </label>
                <input
                  id="gepa-max-metric-calls"
                  type="number"
                  className="conversation-input"
                  placeholder="e.g. 200"
                  value={gepaMaxMetricCalls}
                  onChange={(e) =>
                    setGepaMaxMetricCalls(Number(e.target.value) || 0)
                  }
                />
              </div>
            </div>
            <div className="field-row" style={{ marginBottom: 8 }}>
              <div className="field-column">
                <label className="field-label" htmlFor="gepa-candidate-selection">
                  Candidate selection strategy
                </label>
                <input
                  id="gepa-candidate-selection"
                  className="conversation-input"
                  placeholder="e.g. pareto"
                  value={gepaCandidateSelection}
                  onChange={(e) => setGepaCandidateSelection(e.target.value)}
                />
              </div>
              <div className="field-column">
                <span className="field-label">GEPA options</span>
                <div style={{ display: "flex", gap: 12, marginTop: 4 }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <input
                      type="checkbox"
                      checked={gepaUseMerge}
                      onChange={(e) => setGepaUseMerge(e.target.checked)}
                    />
                    use_merge
                  </label>
                  <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <input
                      type="checkbox"
                      checked={gepaTrackStats}
                      onChange={(e) => setGepaTrackStats(e.target.checked)}
                    />
                    track_stats
                  </label>
                </div>
              </div>
            </div>
            <div className="field-row" style={{ marginBottom: 8 }}>
              <div className="field-column">
                <label className="field-label" htmlFor="critique-model">
                  Critique model
                </label>
                <input
                  id="critique-model"
                  className="conversation-input"
                  placeholder="e.g. gpt-4.1-mini"
                  value={critiqueModel}
                  onChange={(e) => setCritiqueModel(e.target.value)}
                />
              </div>
              <div className="field-column">
                <label className="field-label" htmlFor="critique-max-tokens">
                  Critique max tokens
                </label>
                <input
                  id="critique-max-tokens"
                  type="number"
                  className="conversation-input"
                  placeholder="e.g. 2048"
                  value={critiqueMaxTokens}
                  onChange={(e) =>
                    setCritiqueMaxTokens(Number(e.target.value) || 0)
                  }
                />
              </div>
            </div>
            <button
              className="secondary-button"
              onClick={handleRunCritique}
              disabled={busyCritique}
            >
              {busyCritique ? "Critiquing..." : "Run critique on conversation"}
            </button>
            <button
              className="primary-button"
              onClick={handleEndAndOptimize}
              disabled={busyOptimize}
              style={{ marginLeft: 12 }}
            >
              {busyOptimize ? "Optimizing..." : "End & Optimize BLP Prompt"}
            </button>
            {busyOptimize && optimizeJobId && (
              <button
                className="secondary-button"
                onClick={handleCancelOptimize}
                style={{ marginLeft: 8 }}
              >
                Stop Optimize
              </button>
            )}
            {busyOptimize && (
              <div style={{ marginTop: 8 }}>
                <div className="progress">
                  <div
                    className="progress-bar"
                    style={{ width: `${optimizePercent}%` }}
                  />
                </div>
                <div className="progress-label">
                  {optimizeStatus ?? "running"} · {optimizePercent}%
                </div>
              </div>
            )}
          </div>

          {critique && (
            <div className="panel-output">
              <div className="panel-output-header">Critique</div>
              <div className="panel-output-body">
                <p>{critique.overall_comment}</p>
                <p>
                  <strong>Clinical alignment:</strong>{" "}
                  {(critique.clinical_alignment.score * 100).toFixed(0)}% –{" "}
                  {critique.clinical_alignment.label}
                </p>
                <p>
                  <strong>Persona alignment:</strong>{" "}
                  {(critique.persona_alignment.score * 100).toFixed(0)}% –{" "}
                  {critique.persona_alignment.label}
                </p>
                {critique.safety_flags.length > 0 && (
                  <p>
                    <strong>Safety flags:</strong> {critique.safety_flags.join("; ")}
                  </p>
                )}
                {critique.suggested_improvements.length > 0 && (
                  <div>
                    <strong>Suggestions:</strong>
                    <ul>
                      {critique.suggested_improvements.map((s, idx) => (
                        <li key={idx}>{s}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
          {(originalPrompt || optimizedPrompt) && (
            <div className="panel-output">
              <div className="panel-output-header">
                BLP Prompt: Original vs Optimized
              </div>
              <div className="panel-output-body blp-prompt-grid">
                <div>
                  <div className="panel-output-subheader">Original</div>
                  <pre className="panel-pre">{originalPrompt ?? "—"}</pre>
                </div>
                <div>
                  <div className="panel-output-subheader">Optimized</div>
                  <pre className="panel-pre">{optimizedPrompt ?? "—"}</pre>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default App;



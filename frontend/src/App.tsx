import React, { useEffect, useRef, useState } from "react";

type Message = {
  role: "doctor" | "patient";
  content: string;
};

type BehavioralLinguisticProfile = Record<string, unknown>;
type PatientProfile = {
  extra_attributes?: Record<string, string>;
  id?: string;
  primary_diagnoses?: string[];
  [key: string]: unknown;
};

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

type SimulationTrace = {
  session_id: string;
  case_id: string;
  transcript: { role: "doctor" | "patient"; content: string }[];
  doctor_diagnosis: string;
  ground_truth_diagnosis: string[];
  turns_count: number;
};

type DoctorCritiqueResult = {
  overall_score: number;
  diagnostic_accuracy: number;
  process_score: number;
  critique_text: string;
  successful_diagnosis: boolean;
};

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const App: React.FC = () => {
  // Mode Toggle

  // Frontend-adjustable models and token budgets
  const defaultModel = "openai/gpt-5.1-2025-11-13";

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

  // Doctor Mode State
  const [busyDoctorSim, setBusyDoctorSim] = useState(false);
  const [doctorTrace, setDoctorTrace] = useState<SimulationTrace | null>(null);
  const [doctorCritique, setDoctorCritique] = useState<DoctorCritiqueResult | null>(null);
  const [busyTraining, setBusyTraining] = useState(false);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);

  const [optimizeJobId, setOptimizeJobId] = useState<string | null>(null);
  const [optimizePercent, setOptimizePercent] = useState<number>(0);
  const [optimizeStatus, setOptimizeStatus] = useState<string | null>(null);
  const [optimizeTimer, setOptimizeTimer] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [critique, setCritique] = useState<CritiqueResult | null>(null);
  const [originalPrompt, setOriginalPrompt] = useState<string | null>(null);
  const [optimizedPrompt, setOptimizedPrompt] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [busyReset, setBusyReset] = useState(false);
  const [doctorSimModel, setDoctorSimModel] = useState(defaultModel);
  const [doctorPatientModel, setDoctorPatientModel] = useState(defaultModel);
  const [simPatientModel, setSimPatientModel] = useState(defaultModel);
  const extraAttributes = patientObj?.extra_attributes;
  const hasExtraAttributes =
    extraAttributes && Object.keys(extraAttributes).length > 0;
  const conversationEndRef = useRef<HTMLDivElement | null>(null);
  const [blpModel, setBlpModel] = useState(defaultModel);
  const [blpMaxTokens, setBlpMaxTokens] = useState<number>(2048);
  const [patientModel, setPatientModel] = useState(defaultModel);
  const [patientMaxTokens, setPatientMaxTokens] = useState<number>(1024);
  const [critiqueModel, setCritiqueModel] = useState(defaultModel);
  const [critiqueMaxTokens, setCritiqueMaxTokens] = useState<number>(2048);
  // GEPA knobs
  const [gepaModel, setGepaModel] = useState(defaultModel);
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
          api_key: apiKey || undefined,
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
        api_key: apiKey || undefined,
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
            setError(`Optimization job failed: ${pData.message || "Check backend logs for details."}`);
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
          api_key: apiKey || undefined,
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
        model: simPatientModel,
        api_key: apiKey || undefined,
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

  useEffect(() => {
    if (conversationEndRef.current) {
      conversationEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [conversation, doctorTrace]);

  const handleResetChat = async () => {
    if (conversation.length > 0) {
      const ok = window.confirm(
        "Reset chat and clear the conversation history?"
      );
      if (!ok) return;
    }
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
      setDoctorTrace(null);
      setDoctorCritique(null);
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
        api_key: apiKey || undefined,
      };

      const res = await fetch(`${API_BASE}/api/critique`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const t = await res.text();
        // Try to parse JSON error if possible, otherwise use text
        let msg = t;
        try {
            const jsonErr = JSON.parse(t);
            if (jsonErr.detail) msg = jsonErr.detail;
        } catch {}
        throw new Error(`Critique API error (${res.status}): ${msg}`);
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

  // Doctor Simulation Handlers
  const handleRunDoctorSim = async () => {
    if (!blpObj || !patientObj) {
      setError("You need a BLP and Patient Profile first.");
      return;
    }
    setBusyDoctorSim(true);
    setError(null);
    setDoctorTrace(null);
    setDoctorCritique(null);

    try {
        const res = await fetch(`${API_BASE}/api/simulate/doctor-session`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                blp: blpObj,
                patient_profile: patientObj,
                max_turns: 10,
                doctor_model: doctorSimModel,
                patient_model: doctorPatientModel,
                api_key: apiKey || undefined,
            }),
        });
        if (!res.ok) {
            const text = await res.text();
            throw new Error(`Simulation failed: ${text}`);
        }
        const data = await res.json();
        setDoctorTrace(data.trace);
        setDoctorCritique(data.critique);
    } catch(e) {
        setError(e instanceof Error ? e.message : "Simulation failed");
    } finally {
        setBusyDoctorSim(false);
    }
  };

  const handleTrainDoctor = async () => {
    if (!blpObj || !patientObj) {
      setError("You need a BLP and Patient Profile first.");
      return;
    }
    setBusyTraining(true);
    try {
        const res = await fetch(`${API_BASE}/api/train/doctor`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                blp: blpObj,
                patient_profile: patientObj,
                iterations: 3,
                case_id: patientObj.id || "manual_case",
                api_key: apiKey || undefined,
            }),
        });
        const data = await res.json();
        setTrainingJobId(data.job_id);
        alert("Training job started! Check server logs for JSONL trace output.");
    } catch(e) {
        setError("Failed to start training");
    } finally {
        setBusyTraining(false);
    }
  };

  const handleTrainFromData = async () => {
    setBusyTraining(true);
    try {
        const res = await fetch(`${API_BASE}/api/train/doctor/from-data`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                num_cases: 10,
                iterations: 2,
                api_key: apiKey || undefined,
            }),
        });
        const data = await res.json();
        setTrainingJobId(data.job_id);
        alert("Training from data files started! Loading 10 cases from parquet + transcripts.");
    } catch(e) {
        setError("Failed to start data training");
    } finally {
        setBusyTraining(false);
    }
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <div style={{ flex: 1 }}>
          <h1>Persona Generation Lab</h1>
          <div style={{ marginTop: "0.5rem", fontSize: "0.9rem" }}>
            Build BLP + Patient Profile, chat with the simulated patient, then run the Simulated Doctor & batch training on the same page.
          </div>
        </div>
        <span className="badge">Node · React · Vite</span>
      </header>

      <section className="panel" style={{ marginBottom: "1rem", display: "flex", gap: "1rem", alignItems: "center" }}>
        <div style={{ flex: 1 }}>
          <label className="field-label" htmlFor="api-key">API key (OpenAI / Gemini)</label>
          <input
            id="api-key"
            type="password"
            className="conversation-input"
            placeholder="Enter API key to use for requests..."
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
        </div>
        <span className="badge badge-soft">Used for all backend requests</span>
      </section>

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
              {/* Force style display block to ensure visibility */}
              <select
                id="blp-model"
                className="conversation-input"
                style={{ display: 'block', width: '100%', marginBottom: '4px' }}
                value={blpModel}
                onChange={(e) => setBlpModel(e.target.value)}
              >
                <option value="gemini/gemini-3-pro-preview">Gemini 3 Pro Preview</option>
                <option value="openai/gpt-5.1-2025-11-13">GPT-5.1 (Preview)</option>
                <option value="openai/gpt-4o">GPT-4o</option>
              </select>
              {/* Fallback text input for custom models */}
              <input 
                 className="conversation-input" 
                 style={{marginTop: 4}}
                 placeholder="Or type provider/model (e.g. anthropic/claude-3-opus)..." 
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
            disabled={busyBLP || !rawTranscript.trim()}
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
              <select
                id="patient-model"
                className="conversation-input"
                style={{ display: 'block', width: '100%', marginBottom: '4px' }}
                value={patientModel}
                onChange={(e) => setPatientModel(e.target.value)}
              >
                <option value="gemini/gemini-3-pro-preview">Gemini 3 Pro Preview</option>
                <option value="openai/gpt-5.1-2025-11-13">GPT-5.1 (Preview)</option>
                <option value="openai/gpt-4o">GPT-4o</option>
              </select>
               <input 
                 className="conversation-input" 
                 style={{marginTop: 4}}
                 placeholder="Or type provider/model..." 
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
            disabled={busyProfile || !rawCase.trim()}
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
          {blpObj && (
            <div className="panel-output">
              <div className="panel-output-header">BLP (read-only)</div>
              <div className="panel-output-body">
                <pre className="panel-pre small-pre">
                  {JSON.stringify(blpObj, null, 2)}
                </pre>
              </div>
            </div>
          )}
          {patientObj && (
            <div className="panel-output">
              <div className="panel-output-header">Patient Profile (read-only)</div>
              <div className="panel-output-body">
                <pre className="panel-pre small-pre">
                  {JSON.stringify(patientObj, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {hasExtraAttributes && extraAttributes && (
            <div className="panel-output">
              <div className="panel-output-header">Extra attributes</div>
              <div className="panel-output-body">
                <ul className="extra-attributes-list">
                  {Object.entries(extraAttributes).map(([key, value]) => (
                    <li key={key}>
                      <span className="extra-attr-key">{key}</span>
                      <span className="extra-attr-value">{value}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

        </section>

        {/* Right: Chat + Doctor Training on one page */}
        <section className="panel panel-span">
          <>
            <h2>Simulated Patient (interactive)</h2>
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
                <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                  <select
                    className="conversation-input"
                    value={simPatientModel}
                    onChange={(e) => setSimPatientModel(e.target.value)}
                    style={{ minWidth: 200 }}
                  >
                    <option value={defaultModel}>Sim patient model: GPT-5.1</option>
                    <option value="gemini/gemini-3-pro-preview">Gemini 3 Pro Preview</option>
                    <option value="openai/gpt-4o">GPT-4o</option>
                  </select>
                </div>
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
                  <>
                    {conversation.map((msg, idx) => (
                      <div
                        key={idx}
                        className={`bubble bubble-${msg.role}`}
                      >
                        <div className="bubble-role">
                          {msg.role === "doctor" ? "Doctor" : "Patient"}
                        </div>
                        <div className="bubble-text">{msg.content}</div>
                      </div>
                    ))}
                    <div ref={conversationEndRef} />
                  </>
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
                    if (
                      e.key === "Enter" &&
                      (e.ctrlKey || e.metaKey || !e.shiftKey)
                    ) {
                      e.preventDefault();
                      handleSendTurn();
                    }
                  }}
                />
                <button
                  className="primary-button"
                  onClick={handleSendTurn}
                  disabled={busyTurn || !doctorInput.trim()}
                >
                  {busyTurn ? "Thinking..." : "Send"}
                </button>
              </div>
            </div>

            <div className="critique-controls">
              <div className="field-row" style={{ marginBottom: 8 }}>
                <div className="field-column">
                  <label className="field-label">Critique model</label>
                  <select
                    className="conversation-input"
                    style={{ display: "block", width: "100%", marginBottom: 4 }}
                    value={critiqueModel}
                    onChange={(e) => setCritiqueModel(e.target.value)}
                  >
                    <option value="openai/gpt-5.1-2025-11-13">GPT-5.1 (default)</option>
                    <option value="gemini/gemini-3-pro-preview">Gemini 3 Pro Preview</option>
                    <option value="openai/gpt-4o">GPT-4o</option>
                  </select>
                </div>
              </div>
            <button
              className="secondary-button"
              onClick={handleRunCritique}
              disabled={busyCritique}
            >
              {busyCritique ? "Critiquing..." : "Run critique on conversation"}
            </button>
            {busyCritique && (
                <div style={{ marginTop: "0.5rem" }}>
                    <div className="progress">
                        <div
                            className="progress-bar"
                            style={{ width: "100%", animation: "indeterminate 1.5s infinite linear", background: "linear-gradient(90deg, #3b82f6, #8b5cf6)" }}
                        />
                        <style>{`
                            @keyframes indeterminate {
                                0% { transform: translateX(-100%); }
                                100% { transform: translateX(100%); }
                            }
                        `}</style>
                    </div>
                    <div className="progress-label"> Analyzing conversation... </div>
                </div>
            )}
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
                  <div style={{ marginTop: "1rem" }}>
                    <strong>Safety Flags:</strong> {critique.safety_flags.length > 0 ? critique.safety_flags.join(", ") : "None"}
                  </div>
                  <div style={{ marginTop: "0.5rem" }}>
                    <strong>Suggestions:</strong>
                    <ul style={{ margin: "0.5rem 0 0 1.2rem" }}>
                      {critique.suggested_improvements.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {(originalPrompt || optimizedPrompt) && (
              <div className="panel-output">
                <div className="panel-output-header">Optimization Result</div>
                <div className="panel-output-body">
                  <div style={{ marginBottom: "1rem" }}>
                    <strong>Original Prompt:</strong>
                    <pre className="panel-pre small-pre">{originalPrompt}</pre>
                  </div>
                  <div>
                    <strong>Optimized Prompt:</strong>
                    <pre
                      className="panel-pre small-pre"
                      style={{ border: "1px solid #4caf50" }}
                    >
                      {optimizedPrompt}
                    </pre>
                  </div>
                  <p style={{ fontSize: "0.85rem", color: "#666", marginTop: "0.5rem" }}>
                    The optimized prompt has been saved to <code>prompts.yaml</code>.
                  </p>
                </div>
              </div>
            )}

            {/* Doctor Training Section (always visible) */}
              <div style={{ marginTop: "2rem", borderTop: "1px solid #eee", paddingTop: "1.5rem" }}>
                <h2>Simulated Doctor Agent</h2>
                <p className="panel-subtitle">
                    Automated Doctor vs Patient loop to generate training traces.
                </p>

              <div className="conversation-card">
                  <div className="conversation-header" style={{ alignItems: "center" }}>
                      <div style={{ maxWidth: "65%" }}>
                          <div className="conversation-title">Simulation View</div>
                          <div className="conversation-caption">
                              Watch the Simulated Doctor interview the Patient autonomously.
                          </div>
                          <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem" }}>
                            <select
                              className="conversation-input"
                              style={{ fontSize: "0.85rem", padding: "4px 8px" }}
                              value={doctorSimModel}
                              onChange={(e) => setDoctorSimModel(e.target.value)}
                            >
                              <option value={defaultModel}>Dr Model: GPT-5.1</option>
                              <option value="gemini/gemini-3-pro-preview">Gemini 3 Pro Preview</option>
                              <option value="openai/gpt-4o">GPT-4o</option>
                            </select>
                            <select
                              className="conversation-input"
                              style={{ fontSize: "0.85rem", padding: "4px 8px" }}
                              value={doctorPatientModel}
                              onChange={(e) => setDoctorPatientModel(e.target.value)}
                            >
                              <option value={defaultModel}>Pat Model: GPT-5.1</option>
                              <option value="gemini/gemini-3-pro-preview">Gemini 3 Pro Preview</option>
                              <option value="openai/gpt-4o">GPT-4o</option>
                            </select>
                          </div>
                      </div>
                      <button 
                          className="primary-button"
                          onClick={handleRunDoctorSim}
                          disabled={busyDoctorSim}
                          style={{ minWidth: 160 }}
                      >
                          {busyDoctorSim ? "Running Simulation..." : "Run Auto-Sim"}
                      </button>
                  </div>

                  <div className="conversation-body" style={{ minHeight: 180 }}>
                      { busyDoctorSim ? (
                          <div className="placeholder" style={{ textAlign: "center", marginTop: "2rem" }}>
                              <div className="spinner" style={{ 
                                  width: "24px", 
                                  height: "24px", 
                                  border: "3px solid rgba(255,255,255,0.3)", 
                                  borderTopColor: "#fff", 
                                  borderRadius: "50%", 
                                  animation: "spin 1s linear infinite",
                                  margin: "0 auto 1rem" 
                              }} />
                              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                              <div>Running automated simulation...</div>
                          </div>
                      ) : !doctorTrace ? (
                          <div className="placeholder">
                              Click "Run Auto-Sim" to start an automated session.
                          </div>
                      ) : (
                          <>
                              {doctorTrace.transcript.map((msg, idx) => (
                                  <div key={idx} className={`bubble bubble-${msg.role}`}>
                                      <div className="bubble-role">
                                          {msg.role === "doctor" ? "Sim Doctor" : "Sim Patient"}
                                      </div>
                                      <div className="bubble-text">{msg.content}</div>
                                  </div>
                              ))}
                              <div className="bubble bubble-system">
                                  <div className="bubble-role">SYSTEM</div>
                                  <div className="bubble-text">
                                      <strong>Final Diagnosis:</strong> {doctorTrace.doctor_diagnosis}
                                  </div>
                              </div>
                              <div ref={conversationEndRef} />
                          </>
                      )}
                  </div>
              </div>

              { doctorCritique && (
                  <div className="panel-output">
                      <div className="panel-output-header">
                          Reward Signal (Score: {(doctorCritique.overall_score * 100).toFixed(0)}%)
                      </div>
                      <div className="panel-output-body">
                          <div style={{display: 'flex', gap: '1rem', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap'}}>
                              <div>
                                  <strong>Diagnostic Accuracy:</strong> {(doctorCritique.diagnostic_accuracy * 100).toFixed(0)}%
                              </div>
                              <div>
                                  <strong>Process Score:</strong> {(doctorCritique.process_score * 100).toFixed(0)}%
                              </div>
                          </div>
                          <p>{doctorCritique.critique_text}</p>
                      </div>
                  </div>
              )}

              <div className="critique-controls" style={{marginTop: '1rem', borderTop: '1px solid #eee', paddingTop: '1rem'}}>
                  <h3>Batch Training</h3>
                  <p>Run multiple simulations to collect traces for GRPO.</p>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                      <button
                          className="primary-button"
                          onClick={handleTrainDoctor}
                          disabled={busyTraining}
                      >
                          {busyTraining ? "Job Queued..." : "Train from Current Case"}
                      </button>
                      <button
                          className="secondary-button"
                          onClick={handleTrainFromData}
                          disabled={busyTraining}
                      >
                          {busyTraining ? "Job Queued..." : "Train from Data Files"}
                      </button>
                  </div>
                  <p style={{ fontSize: '0.85rem', marginTop: '0.5rem', color: '#666' }}>
                      "Train from Data Files" loads 10 cases from data/clinical_cases/cases.parquet and data/transcripts/
                  </p>
              </div>
            </div>
          </>
        </section>
      </main>
    </div>
  );
};

export default App;

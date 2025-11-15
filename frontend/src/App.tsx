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
  const [blpPreview, setBlpPreview] = useState<string | null>(null);
  const [patientPreview, setPatientPreview] = useState<string | null>(null);
  const [doctorInput, setDoctorInput] = useState("");
  const [conversation, setConversation] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const [blpObj, setBlpObj] = useState<BehavioralLinguisticProfile | null>(
    null,
  );
  const [patientObj, setPatientObj] = useState<PatientProfile | null>(null);

  const [busyBLP, setBusyBLP] = useState(false);
  const [busyProfile, setBusyProfile] = useState(false);
  const [busyTurn, setBusyTurn] = useState(false);
  const [busyCritique, setBusyCritique] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [critique, setCritique] = useState<CritiqueResult | null>(null);

  const handleExtractBLP = async () => {
    if (!rawTranscript.trim()) return;
    setBusyBLP(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/blp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: rawTranscript }),
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
      setBlpPreview(JSON.stringify(data.blp, null, 2));
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Failed to extract BLP from backend.",
      );
    } finally {
      setBusyBLP(false);
    }
  };

  const handleBuildPatientProfile = async () => {
    if (!rawCase.trim()) return;
    setBusyProfile(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/patient-profile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw_case: rawCase }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Patient profile API error (${res.status}): ${text}`);
      }

      const data: { patient_profile: PatientProfile } = await res.json();
      setPatientObj(data.patient_profile);
      setPatientPreview(JSON.stringify(data.patient_profile, null, 2));
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Failed to build patient profile from backend.",
      );
    } finally {
      setBusyProfile(false);
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
            onChange={(e) => setRawTranscript(e.target.value)}
          />

          <button
            className="primary-button"
            onClick={handleExtractBLP}
            disabled={busyBLP}
          >
            {busyBLP ? "Processing..." : "Anonymize & Extract BLP"}
          </button>

          <div className="panel-output">
            <div className="panel-output-header">BLP preview</div>
            <div className="panel-output-body">
              {blpPreview ?? (
                <span className="placeholder">
                  Once connected, this will show the structured Behavioral &amp;
                  Linguistic Profile.
                </span>
              )}
            </div>
          </div>
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
            onChange={(e) => setRawCase(e.target.value)}
          />

          <button
            className="primary-button"
            onClick={handleBuildPatientProfile}
            disabled={busyProfile}
          >
            {busyProfile ? "Structuring..." : "Structure data & build profile"}
          </button>

          <div className="panel-output">
            <div className="panel-output-header">Patient profile preview</div>
            <div className="panel-output-body">
              {patientPreview ?? (
                <span className="placeholder">
                  Once connected, this will show the structured Patient Profile.
                </span>
              )}
            </div>
          </div>
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
            <button
              className="secondary-button"
              onClick={handleRunCritique}
              disabled={busyCritique}
            >
              {busyCritique ? "Critiquing..." : "Run critique on conversation"}
            </button>
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
        </section>
      </main>
    </div>
  );
};

export default App;



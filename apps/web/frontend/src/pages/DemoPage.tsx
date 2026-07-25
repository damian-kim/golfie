import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import type { PreviousSessionSummary, TrajectoryPayload } from "../lib/types";
import { ShotSimulatorView } from "../components/ShotSimulatorView";
import { DEFAULT_SAMPLE, makeSamplePayload, type SampleParameters } from "../lib/sampleTrajectory";
import "../styles/forms.css";
import "../styles/demo.css";

const SAMPLE_SESSION_ID = "sample";

const processedAtFormatter = new Intl.DateTimeFormat(undefined, {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  second: "2-digit",
  timeZoneName: "short",
});

function formatProcessedAt(value: string): string {
  return processedAtFormatter.format(new Date(value));
}

export function DemoPage() {
  const navigate = useNavigate();
  const [parameters, setParameters] = useState<SampleParameters>(DEFAULT_SAMPLE);
  const [replayKey, setReplayKey] = useState(0);
  const [payload, setPayload] = useState<TrajectoryPayload | null>(() => makeSamplePayload(DEFAULT_SAMPLE));
  const [history, setHistory] = useState<PreviousSessionSummary[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState(SAMPLE_SESSION_ID);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const selectedSession = useMemo(
    () => history.find((session) => session.session_id === selectedSessionId) ?? null,
    [history, selectedSessionId]
  );

  useEffect(() => {
    api.listPreviousSessions()
      .then(setHistory)
      .catch(() => setHistory([]))
      .finally(() => setIsLoading(false));
  }, []);

  async function selectSession(sessionId: string) {
    setSelectedSessionId(sessionId);
    setIsLoading(true);
    setError(null);

    try {
      const nextPayload =
        sessionId === SAMPLE_SESSION_ID
          ? makeSamplePayload(parameters)
          : await api.getTrajectory(sessionId);
      setPayload(nextPayload);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Failed to load that simulation session."
      );
    } finally {
      setIsLoading(false);
    }
  }

  const title = selectedSession ? "Previous Shot Replay" : "Demo Driving Range";
  const subtitle = selectedSession
    ? `${selectedSession.upload_identifier} - Processed ${formatProcessedAt(selectedSession.processed_at)}`
    : "Synthetic flight profile utilized for WebGL rendering validation.";

  return (
    <div className="page page--full demo-page">
      <button
        type="button"
        className="primary-button demo-back-button"
        onClick={() => navigate("/")}
      >
        &larr; Back to Dashboard
      </button>

      <section className="demo-session-picker" aria-label="Past simulation sessions">
        <label htmlFor="demo-session-select">Past sessions</label>
        <select
          id="demo-session-select"
          value={selectedSessionId}
          onChange={(event) => void selectSession(event.target.value)}
          disabled={isLoading}
        >
          <option value={SAMPLE_SESSION_ID}>Sample simulation</option>
          {history.map((session) => (
            <option key={session.session_id} value={session.session_id}>
              {formatProcessedAt(session.processed_at)} - {session.upload_identifier}
            </option>
          ))}
        </select>
        <div className="demo-session-picker__meta" title={selectedSession?.session_uid}>
          {selectedSession ? `UID: ${selectedSession.session_uid}` : "Synthetic sample"}
        </div>
      </section>

      {error && <div className="error-banner demo-error-banner">{error}</div>}
      {isLoading && <div className="demo-loading">Loading simulation...</div>}

      {selectedSessionId === SAMPLE_SESSION_ID && <section className="demo-parameters" aria-label="Sample shot parameters"><header><div><strong>Sample shot parameters</strong><small>Synthetic physics - no swing attached</small></div><button type="button" onClick={() => { setParameters(DEFAULT_SAMPLE); setPayload(makeSamplePayload(DEFAULT_SAMPLE)); }}>Reset</button></header><div>{([
        ['velocity','Velocity',20,75,1,'m/s'],['angle','Angle',5,35,1,'deg'],['direction','Direction',-15,15,1,'deg'],['backspin','Backspin',0,6500,100,'rpm'],['sidespin','Sidespin',-2500,2500,100,'rpm'],['drag','Drag coefficient',.1,.5,.01,'Cd'],['gravity','Gravity',2,15,.1,'m/s2'],
      ] as const).map(([key,label,min,max,step,unit]) => <label key={key}><span>{label}<b>{parameters[key]} {unit}</b></span><input type="range" min={min} max={max} step={step} value={parameters[key]} onChange={(event) => { const next = { ...parameters, [key]: Number(event.target.value) }; setParameters(next); setPayload(makeSamplePayload(next)); }} /></label>)}</div><button type="button" className="primary-button" onClick={() => setReplayKey((value) => value + 1)}>Replay customized trajectory</button></section>}

      {payload && (
        <ShotSimulatorView key={`${payload.session_id}-${replayKey}`} payload={payload} title={title} subtitle={subtitle} />
      )}
    </div>
  );
}

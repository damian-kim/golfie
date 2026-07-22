import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, apiUrl } from "../lib/api";
import type { TrimJobResult } from "../lib/types";
import "../styles/forms.css";
import "./AutoTrimPage.css";

const MAX_FILE_SIZE = 4 * 1024 * 1024 * 1024;
const MIN_SELECTION_SECONDS = 0.05;

function formatSeconds(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds - minutes * 60;
  return `${minutes}:${remainder.toFixed(2).padStart(5, "0")}`;
}

export function AutoTrimPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    mode: "start" | "end" | "window" | "playhead";
    pointerId: number;
    originTime: number;
    originStart: number;
    originEnd: number;
  } | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState(0);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [thumbnails, setThumbnails] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TrimJobResult | null>(null);

  const sourceUrl = useMemo(() => file ? URL.createObjectURL(file) : null, [file]);
  useEffect(() => {
    return () => {
      if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    };
  }, [sourceUrl]);

  useEffect(() => {
    if (!sourceUrl || duration <= 0) {
      return;
    }
    let cancelled = false;
    const thumbnailVideo = document.createElement("video");
    thumbnailVideo.muted = true;
    thumbnailVideo.preload = "auto";
    thumbnailVideo.src = sourceUrl;

    async function generateThumbnails() {
      if (thumbnailVideo.readyState < 2) {
        await new Promise<void>((resolve, reject) => {
          thumbnailVideo.addEventListener("loadeddata", () => resolve(), { once: true });
          thumbnailVideo.addEventListener("error", () => reject(new Error("preview failed")), { once: true });
        });
      }
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      if (!context) return;
      canvas.width = 160;
      canvas.height = 90;
      const frameCount = 10;
      const frames: string[] = [];
      for (let index = 0; index < frameCount && !cancelled; index += 1) {
        const target = Math.min(duration - 0.01, (duration * index) / frameCount);
        if (Math.abs(thumbnailVideo.currentTime - target) > 0.005) {
          await new Promise<void>((resolve) => {
            thumbnailVideo.addEventListener("seeked", () => resolve(), { once: true });
            thumbnailVideo.currentTime = Math.max(0, target);
          });
        }
        const scale = Math.max(
          canvas.width / thumbnailVideo.videoWidth,
          canvas.height / thumbnailVideo.videoHeight
        );
        const width = thumbnailVideo.videoWidth * scale;
        const height = thumbnailVideo.videoHeight * scale;
        context.drawImage(
          thumbnailVideo,
          (canvas.width - width) / 2,
          (canvas.height - height) / 2,
          width,
          height
        );
        frames.push(canvas.toDataURL("image/jpeg", 0.66));
      }
      if (!cancelled) setThumbnails(frames);
    }

    generateThumbnails().catch(() => {
      if (!cancelled) setThumbnails([]);
    });
    return () => {
      cancelled = true;
      thumbnailVideo.removeAttribute("src");
      thumbnailVideo.load();
    };
  }, [duration, sourceUrl]);

  const selectionDuration = Math.max(0, trimEnd - trimStart);
  const selectionLeft = duration > 0 ? (trimStart / duration) * 100 : 0;
  const selectionWidth = duration > 0 ? (selectionDuration / duration) * 100 : 0;
  const excludedRight = duration > 0 ? Math.max(0, 100 - (trimEnd / duration) * 100) : 0;
  const playheadPosition = duration > 0
    ? Math.min(100, Math.max(0, (currentTime / duration) * 100))
    : 0;
  const canTrim = Boolean(
    file && duration > 0 && selectionDuration >= MIN_SELECTION_SECONDS && !busy
  );

  function chooseFile(nextFile: File | null) {
    setResult(null);
    setError(null);
    setDuration(0);
    setTrimStart(0);
    setTrimEnd(0);
    setCurrentTime(0);
    setThumbnails([]);
    if (!nextFile) {
      setFile(null);
      return;
    }
    if (!nextFile.type.startsWith("video/") && !nextFile.name.match(/\.(mp4|mov|m4v|mkv|webm)$/i)) {
      setError("Choose a video file such as MP4, MOV, M4V, MKV, or WebM.");
      return;
    }
    if (nextFile.size > MAX_FILE_SIZE) {
      setError("This video is larger than the current 4 GB upload limit.");
      return;
    }
    setFile(nextFile);
  }

  function setStart(value: number, seek = true) {
    if (!Number.isFinite(value)) return;
    const next = Math.min(Math.max(0, value), Math.max(0, trimEnd - MIN_SELECTION_SECONDS));
    setTrimStart(next);
    if (seek && videoRef.current) videoRef.current.currentTime = next;
  }

  function setEnd(value: number, seek = true) {
    if (!Number.isFinite(value)) return;
    const next = Math.max(
      trimStart + MIN_SELECTION_SECONDS,
      Math.min(duration, value)
    );
    setTrimEnd(next);
    if (seek && videoRef.current) videoRef.current.currentTime = next;
  }

  function timeFromPointer(clientX: number): number {
    const rect = timelineRef.current?.getBoundingClientRect();
    if (!rect || duration <= 0) return 0;
    return Math.min(duration, Math.max(0, ((clientX - rect.left) / rect.width) * duration));
  }

  function beginTimelineDrag(
    mode: "start" | "end" | "window" | "playhead",
    event: React.PointerEvent
  ) {
    if (busy || duration <= 0) return;
    event.preventDefault();
    event.stopPropagation();
    const pointerTime = timeFromPointer(event.clientX);
    dragRef.current = {
      mode,
      pointerId: event.pointerId,
      originTime: pointerTime,
      originStart: trimStart,
      originEnd: trimEnd,
    };
    timelineRef.current?.setPointerCapture(event.pointerId);
    if (mode === "playhead") {
      setCurrentTime(pointerTime);
      if (videoRef.current) videoRef.current.currentTime = pointerTime;
    }
  }

  function moveTimelineDrag(event: React.PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    const pointerTime = timeFromPointer(event.clientX);
    if (drag.mode === "start") {
      setStart(pointerTime);
    } else if (drag.mode === "end") {
      setEnd(pointerTime);
    } else if (drag.mode === "window") {
      const length = drag.originEnd - drag.originStart;
      const nextStart = Math.min(
        Math.max(0, drag.originStart + pointerTime - drag.originTime),
        duration - length
      );
      setTrimStart(nextStart);
      setTrimEnd(nextStart + length);
      setCurrentTime(nextStart);
      if (videoRef.current) videoRef.current.currentTime = nextStart;
    } else {
      setCurrentTime(pointerTime);
      if (videoRef.current) videoRef.current.currentTime = pointerTime;
    }
  }

  function endTimelineDrag(event: React.PointerEvent<HTMLDivElement>) {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    if (timelineRef.current?.hasPointerCapture(event.pointerId)) {
      timelineRef.current.releasePointerCapture(event.pointerId);
    }
  }

  async function playSelection() {
    if (!videoRef.current) return;
    videoRef.current.currentTime = trimStart;
    try {
      await videoRef.current.play();
    } catch {
      // Native media controls remain available when autoplay is restricted.
    }
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!file || !canTrim) return;
    setBusy(true);
    setError(null);
    setResult(null);
    videoRef.current?.pause();
    try {
      setResult(await api.trimVideoManually(file, trimStart, trimEnd));
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.message
          : "Golfie could not trim this selection."
      );
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setError(null);
    setDuration(0);
    setTrimStart(0);
    setTrimEnd(0);
    setCurrentTime(0);
    setThumbnails([]);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <main className="trim-page">
      <section className="trim-hero">
        <div className="trim-hero__eyebrow">
          <span className="trim-hero__pulse" />
          Lossless video editor
        </div>
        <h1>Mark the swing.<br /><span>Keep exactly what matters.</span></h1>
      </section>

      <section className="trim-workspace" aria-label="Manual golf video editor">
        <form onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="file"
            accept="video/*,.mp4,.mov,.m4v,.mkv,.webm"
            onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
            hidden
          />

          {!file || !sourceUrl ? (
            <div
              className={`trim-dropzone${dragging ? " trim-dropzone--dragging" : ""}`}
              onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => {
                event.preventDefault();
                if (event.currentTarget === event.target) setDragging(false);
              }}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                chooseFile(event.dataTransfer.files[0] ?? null);
              }}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
              }}
            >
              <div className="trim-dropzone__icon" aria-hidden="true">+</div>
              <strong>Drop your golf video here</strong>
              <span>or click to browse</span>
              <small>MP4, MOV, M4V, MKV, or WebM - up to 4 GB</small>
            </div>
          ) : (
            <div className="trim-editor">
              <div className="trim-editor__filebar">
                <button type="button" onClick={() => inputRef.current?.click()} disabled={busy}>
                  Change video
                </button>
              </div>

              <video
                ref={videoRef}
                className="trim-editor__video"
                src={sourceUrl}
                controls
                playsInline
                preload="metadata"
                onLoadedMetadata={(event) => {
                  const nextDuration = event.currentTarget.duration;
                  if (!Number.isFinite(nextDuration) || nextDuration <= 0) {
                    setError("This browser could not read the video duration.");
                    return;
                  }
                  setDuration(nextDuration);
                  setTrimStart(0);
                  setTrimEnd(nextDuration);
                  setCurrentTime(0);
                }}
                onTimeUpdate={(event) => {
                  setCurrentTime(event.currentTarget.currentTime);
                  if (trimEnd > trimStart && event.currentTarget.currentTime >= trimEnd) {
                    event.currentTarget.pause();
                    event.currentTarget.currentTime = trimEnd;
                  }
                }}
                onSeeked={(event) => setCurrentTime(event.currentTarget.currentTime)}
              />

              <div className="trim-selection">
                <div className="trim-selection__summary">
                  <span>Selected clip</span>
                  <strong>{formatSeconds(selectionDuration)}</strong>
                </div>
                <div
                  ref={timelineRef}
                  className="trim-filmstrip"
                  onPointerDown={(event) => beginTimelineDrag("playhead", event)}
                  onPointerMove={moveTimelineDrag}
                  onPointerUp={endTimelineDrag}
                  onPointerCancel={endTimelineDrag}
                >
                  <div className="trim-filmstrip__frames" aria-hidden="true">
                    {(thumbnails.length ? thumbnails : Array.from({ length: 10 }, () => "")).map((thumbnail, index) => (
                      thumbnail
                        ? <img key={index} src={thumbnail} alt="" draggable={false} />
                        : <span key={index} />
                    ))}
                  </div>
                  <div className="trim-filmstrip__excluded trim-filmstrip__excluded--left" style={{ width: `${selectionLeft}%` }} />
                  <div className="trim-filmstrip__excluded trim-filmstrip__excluded--right" style={{ width: `${excludedRight}%` }} />
                  <div
                    className="trim-filmstrip__selection"
                    style={{ left: `${selectionLeft}%`, width: `${selectionWidth}%` }}
                    onPointerDown={(event) => beginTimelineDrag("window", event)}
                  >
                    <button
                      type="button"
                      className="trim-filmstrip__handle trim-filmstrip__handle--start"
                      aria-label={`Trim start ${formatSeconds(trimStart)}`}
                      onPointerDown={(event) => beginTimelineDrag("start", event)}
                      onKeyDown={(event) => {
                        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
                          event.preventDefault();
                          const direction = event.key === "ArrowLeft" ? -1 : 1;
                          setStart(trimStart + direction * (event.shiftKey ? 0.1 : 0.01));
                        }
                      }}
                    ><i /><i /><i /></button>
                    <span className="trim-filmstrip__grab">Drag clip</span>
                    <button
                      type="button"
                      className="trim-filmstrip__handle trim-filmstrip__handle--end"
                      aria-label={`Trim end ${formatSeconds(trimEnd)}`}
                      onPointerDown={(event) => beginTimelineDrag("end", event)}
                      onKeyDown={(event) => {
                        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
                          event.preventDefault();
                          const direction = event.key === "ArrowLeft" ? -1 : 1;
                          setEnd(trimEnd + direction * (event.shiftKey ? 0.1 : 0.01));
                        }
                      }}
                    ><i /><i /><i /></button>
                  </div>
                  <div
                    className="trim-filmstrip__playhead"
                    style={{ left: `${playheadPosition}%` }}
                    aria-hidden="true"
                  ><span /></div>
                </div>

                <div className="trim-filmstrip__ruler">
                  <span>{formatSeconds(0)}</span>
                  <span>{formatSeconds(duration / 2)}</span>
                  <span>{formatSeconds(duration)}</span>
                </div>

                <div className="trim-editor__preview-actions">
                  <button type="button" onClick={() => setStart(videoRef.current?.currentTime ?? 0, false)}>
                    Set start to playhead
                  </button>
                  <button type="button" className="trim-editor__play" onClick={playSelection}>
                    Play selection
                  </button>
                  <button type="button" onClick={() => setEnd(videoRef.current?.currentTime ?? duration, false)}>
                    Set end to playhead
                  </button>
                </div>
              </div>
            </div>
          )}

          {error && <div className="error-banner" role="alert">{error}</div>}

          {file && (
            <button className="trim-submit" type="submit" disabled={!canTrim}>
              {busy ? (
                <><span className="trim-spinner" /> Trimming selection...</>
              ) : (
                <>Export lossless clip <span aria-hidden="true">-&gt;</span></>
              )}
            </button>
          )}
          {busy && (
            <p className="trim-processing-note">
              FFmpeg is copying the selected source packets without re-encoding.
            </p>
          )}
        </form>
      </section>

      {result && (
        <section className="trim-result" aria-live="polite">
          <div className="trim-result__header">
            <div>
              <span className="trim-result__status">Export complete</span>
              <h2>Your trimmed shot is ready</h2>
            </div>
            <span className="trim-result__lossless">100% stream copy</span>
          </div>

          <video
            className="trim-result__video"
            src={apiUrl(result.preview_url)}
            controls
            playsInline
            preload="metadata"
          />

          <dl className="trim-stats">
            <div><dt>Clip length</dt><dd>{formatSeconds(result.output_duration_seconds)}</dd></div>
            <div><dt>Frame rate</dt><dd>{result.fps.toFixed(3).replace(/\.0+$/, "")} fps</dd></div>
            <div><dt>Resolution</dt><dd>{result.width} x {result.height}</dd></div>
            <div><dt>Aspect</dt><dd>{result.display_aspect_ratio || result.sample_aspect_ratio}</dd></div>
            <div><dt>Quality</dt><dd>No re-encode</dd></div>
          </dl>

          {!result.keyframe_aligned && (
            <p className="trim-keyframe-note">
              You selected {formatSeconds(result.requested_start_seconds)}. The downloaded clip
              begins at {formatSeconds(result.actual_start_seconds)}, the nearest earlier source
              keyframe. That small safety margin is required for a clean lossless video.
            </p>
          )}

          <div className="trim-result__actions">
            <a className="trim-download" href={apiUrl(result.download_url)}>
              Download trimmed video <span aria-hidden="true">v</span>
            </a>
            <button type="button" onClick={reset}>Edit another video</button>
          </div>
        </section>
      )}

    </main>
  );
}

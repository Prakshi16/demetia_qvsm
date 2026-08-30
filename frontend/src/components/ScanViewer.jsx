/**
 * In-browser MRI viewer, wrapping Niivue (WebGL2). A `.nii`/`.nii.gz`/`.dcm`
 * file doesn't open in anything a clinician has by default, so the Visit Detail
 * screen mounts this instead of just handing over a download.
 *
 * Lazy-loaded from VisitDetail (`React.lazy`) so Niivue's ~1-2 MB never reaches
 * the receptionists and admins who don't view scans.
 *
 * `url` is a short-lived Supabase signed URL; Niivue fetches it directly. If it
 * has expired or CORS blocks it, `loadVolumes` rejects and we show a retry hint.
 */
import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";

export default function ScanViewer({ url, filename }) {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const [status, setStatus] = useState("loading"); // loading | ready | error

  useEffect(() => {
    let cancelled = false;
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    setStatus("loading");
    const nv = new Niivue({
      backColor: [0.06, 0.08, 0.11, 1],
      show3Dcrosshair: true,
      dragAndDropEnabled: false,
    });
    nvRef.current = nv;

    async function load() {
      try {
        nv.attachToCanvas(canvas);
        await nv.loadVolumes([{ url, name: filename || "scan.nii.gz" }]);
        if (!cancelled) setStatus("ready");
      } catch {
        if (!cancelled) setStatus("error");
      }
    }
    load();

    return () => {
      cancelled = true;
      // Niivue has no public dispose; drop the reference and let the canvas be
      // GC'd when this component unmounts.
      nvRef.current = null;
    };
  }, [url, filename]);

  return (
    <div className="scan-viewer no-print">
      <canvas ref={canvasRef} className="scan-viewer__canvas" />
      {status === "loading" ? (
        <p className="visit-note scan-viewer__overlay">Loading scan…</p>
      ) : null}
      {status === "error" ? (
        <p className="visit-note scan-viewer__overlay">
          Couldn’t load the scan — the link may have expired. Close and reopen
          the viewer to refresh it.
        </p>
      ) : null}
    </div>
  );
}

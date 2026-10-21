import { useRef, useState } from "react";

// Pinned pdf.js (UMD build exposes window.pdfjsLib). Loaded on demand so the
// bundle stays small and only PDF uploads pay the cost.
const PDFJS_VERSION = "3.11.174";
const PDFJS_SRC = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${PDFJS_VERSION}/pdf.min.js`;
const PDFJS_WORKER = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${PDFJS_VERSION}/pdf.worker.min.js`;

function loadPdfJs(): Promise<any> {
  return new Promise((resolve, reject) => {
    const w = window as any;
    if (w.pdfjsLib) {
      w.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
      return resolve(w.pdfjsLib);
    }
    const s = document.createElement("script");
    s.src = PDFJS_SRC;
    s.onload = () => {
      if (!w.pdfjsLib) return reject(new Error("pdf.js failed to load"));
      w.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
      resolve(w.pdfjsLib);
    };
    s.onerror = () => reject(new Error("Couldn't load the PDF reader"));
    document.head.appendChild(s);
  });
}

async function extractPdfText(file: File): Promise<string> {
  const pdfjsLib = await loadPdfJs();
  const data = new Uint8Array(await file.arrayBuffer());
  const pdf = await pdfjsLib.getDocument({ data }).promise;
  const pages: string[] = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    pages.push(content.items.map((it: any) => it.str).join(" "));
  }
  return pages.join("\n\n").replace(/[ \t]+/g, " ").trim();
}

async function extractText(file: File): Promise<string> {
  if (file.name.toLowerCase().endsWith(".pdf") || file.type === "application/pdf") {
    return extractPdfText(file);
  }
  // .txt / .md / .docx-as-text and anything else readable as text.
  return (await file.text()).trim();
}

export default function DocsModal({
  onClose,
  onSave,
}: {
  onClose: () => void;
  onSave: (resume: string, jd: string) => Promise<void>;
}) {
  const [resume, setResume] = useState("");
  const [jd, setJd] = useState("");
  const [saving, setSaving] = useState(false);
  const [busyField, setBusyField] = useState<"" | "resume" | "jd">("");
  const [error, setError] = useState("");
  const resumeFileRef = useRef<HTMLInputElement>(null);
  const jdFileRef = useRef<HTMLInputElement>(null);

  async function handleFile(field: "resume" | "jd", file: File | undefined) {
    if (!file) return;
    setError("");
    setBusyField(field);
    try {
      const text = await extractText(file);
      if (!text) {
        setError(`"${file.name}" had no readable text (a scanned/image PDF won't extract — paste it instead).`);
      } else if (field === "resume") {
        setResume(text);
      } else {
        setJd(text);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't read that file — paste the text instead.");
    } finally {
      setBusyField("");
    }
  }

  async function save() {
    setSaving(true);
    try {
      await onSave(resume, jd);
      onClose();
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Use your own resume</h2>
        <p className="modal-sub">
          Upload or paste your résumé and a target job description. The Knowledge agent will analyse{" "}
          <em>your</em> data — match and gaps — instead of the demo profile. Stays in this session only.
        </p>

        {error && <div className="modal-error">{error}</div>}

        <div className="modal-field-head">
          <label className="modal-label">Your résumé</label>
          <button
            type="button"
            className="btn-upload"
            onClick={() => resumeFileRef.current?.click()}
            disabled={busyField === "resume"}
          >
            {busyField === "resume" ? "Reading…" : "⬆ Upload file"}
          </button>
          <input
            ref={resumeFileRef}
            type="file"
            accept=".pdf,.txt,.md,.markdown,.docx,.doc,text/plain"
            hidden
            onChange={(e) => {
              handleFile("resume", e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </div>
        <textarea value={resume} onChange={(e) => setResume(e.target.value)} placeholder="Upload a PDF/TXT or paste your résumé text…" rows={7} />

        <div className="modal-field-head">
          <label className="modal-label">Target job description</label>
          <button
            type="button"
            className="btn-upload"
            onClick={() => jdFileRef.current?.click()}
            disabled={busyField === "jd"}
          >
            {busyField === "jd" ? "Reading…" : "⬆ Upload file"}
          </button>
          <input
            ref={jdFileRef}
            type="file"
            accept=".pdf,.txt,.md,.markdown,.docx,.doc,text/plain"
            hidden
            onChange={(e) => {
              handleFile("jd", e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </div>
        <textarea value={jd} onChange={(e) => setJd(e.target.value)} placeholder="Upload a PDF/TXT or paste the job description…" rows={6} />

        <div className="modal-actions">
          <button className="btn-ghost" onClick={onClose} disabled={saving}>Cancel</button>
          <button className="btn-primary" onClick={save} disabled={saving || (!resume.trim() && !jd.trim())}>
            {saving ? "Saving…" : "Use these"}
          </button>
        </div>
      </div>
    </div>
  );
}

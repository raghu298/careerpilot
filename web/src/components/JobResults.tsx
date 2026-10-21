import { useState } from "react";
import { addApplication } from "@/lib/api";

export type JobCard = {
  title: string;
  company: string;
  location: string;
  url: string;
  salary: string;
  stack: string;
};

/**
 * Renders live job postings as cards with a one-click "Save to tracker" button, so a
 * seeker can go search → save → analyse without retyping. Save writes a real row into
 * the session's SQLite tracker via the API (deduped server-side); the Analytics agent
 * then counts it immediately.
 */
export default function JobResults({ jobs, token }: { jobs: JobCard[]; token: string }) {
  const [saved, setSaved] = useState<Record<number, "saving" | "saved" | "error">>({});

  async function save(i: number, job: JobCard) {
    if (saved[i] === "saving" || saved[i] === "saved") return;
    setSaved((s) => ({ ...s, [i]: "saving" }));
    try {
      await addApplication(token, {
        company: job.company,
        role: job.title,
        status: "Applied",
        location: job.location,
        source: "Job search",
      });
      setSaved((s) => ({ ...s, [i]: "saved" }));
    } catch {
      setSaved((s) => ({ ...s, [i]: "error" }));
    }
  }

  if (!jobs.length) return null;
  return (
    <div className="job-cards">
      {jobs.map((job, i) => (
        <div className="job-card" key={`${job.url}-${i}`}>
          <div className="job-card-main">
            <div className="job-card-title">
              {job.url ? (
                <a href={job.url} target="_blank" rel="noopener noreferrer">{job.title}</a>
              ) : (
                job.title
              )}
            </div>
            <div className="job-card-meta">
              {job.company}
              {job.location ? ` · ${job.location}` : ""}
              {job.salary ? ` · ${job.salary}` : ""}
            </div>
            {job.stack && <div className="job-card-stack">{job.stack}</div>}
          </div>
          <button
            className={`job-save ${saved[i] === "saved" ? "done" : ""}`}
            onClick={() => save(i, job)}
            disabled={saved[i] === "saving" || saved[i] === "saved"}
            title="Save this job to your application tracker"
          >
            {saved[i] === "saved"
              ? "✓ Saved"
              : saved[i] === "saving"
                ? "Saving…"
                : saved[i] === "error"
                  ? "Retry"
                  : "+ Save to tracker"}
          </button>
        </div>
      ))}
    </div>
  );
}

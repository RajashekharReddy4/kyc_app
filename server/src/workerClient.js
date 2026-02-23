import fetch from "node-fetch";
import FormData from "form-data";
import fs from "fs";

export async function callWorkerAnalyze({
  workerUrl,
  caseId,
  lang,
  files,
  videoPath,
  mode
}) {
  // mode: "all" (default), "documents", "images", "audios", "video"

  if (mode === "video") {
    const fd = new FormData();
    fd.append("case_id", caseId);
    fd.append("file", fs.createReadStream(videoPath));

    const res = await fetch(`${workerUrl}/analyze/video`, {
      method: "POST",
      body: fd,
      headers: fd.getHeaders()
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`Video worker error ${res.status}: ${txt}`);
    }
    return await res.json();
  }

  const fd = new FormData();
  fd.append("case_id", caseId);
  if (lang) fd.append("lang", lang);

  const safeFiles = files || {};

  if (mode === "documents") {
    for (const p of (safeFiles.docs || [])) fd.append("docs", fs.createReadStream(p));
    return await postForm(`${workerUrl}/analyze/documents`, fd);
  }

  if (mode === "images") {
    if (safeFiles.idFront) fd.append("id_front", fs.createReadStream(safeFiles.idFront));
    if (safeFiles.selfie) fd.append("selfie", fs.createReadStream(safeFiles.selfie));
    return await postForm(`${workerUrl}/analyze/images`, fd);
  }

  if (mode === "audios") {
    for (const p of (safeFiles.audios || [])) fd.append("audios", fs.createReadStream(p));
    return await postForm(`${workerUrl}/analyze/audios`, fd);
  }

  // default: all
  for (const p of (safeFiles.docs || [])) fd.append("docs", fs.createReadStream(p));
  if (safeFiles.idFront) fd.append("id_front", fs.createReadStream(safeFiles.idFront));
  if (safeFiles.selfie) fd.append("selfie", fs.createReadStream(safeFiles.selfie));
  for (const p of (safeFiles.audios || [])) fd.append("audios", fs.createReadStream(p));
  if (safeFiles.video) fd.append("video", fs.createReadStream(safeFiles.video));

  return await postForm(`${workerUrl}/analyze`, fd);
}

async function postForm(url, fd) {
  const res = await fetch(url, { method: "POST", body: fd, headers: fd.getHeaders() });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Worker error ${res.status}: ${txt}`);
  }
  return await res.json();
}
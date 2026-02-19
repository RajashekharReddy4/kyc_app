import fetch from "node-fetch";
import FormData from "form-data";
import fs from "fs";

export async function callWorkerAnalyze({ workerUrl, caseId, lang, files }) {
  const fd = new FormData();
  fd.append("case_id", caseId);
  fd.append("lang", lang || "eng");

  for (const p of (files.docs || [])) {
    fd.append("docs", fs.createReadStream(p));
  }
  if (files.idFront) fd.append("id_front", fs.createReadStream(files.idFront));
  if (files.idBack) fd.append("id_back", fs.createReadStream(files.idBack));
  if (files.selfie) fd.append("selfie", fs.createReadStream(files.selfie));
  for (const p of (files.audios || [])) {
    fd.append("audios", fs.createReadStream(p));
  }

  const res = await fetch(`${workerUrl}/analyze`, { method: "POST", body: fd, headers: fd.getHeaders() });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Worker error ${res.status}: ${txt}`);
  }
  return await res.json();
}

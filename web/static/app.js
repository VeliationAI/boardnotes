(() => {
  const fileInput = document.getElementById("file");
  const dropzone = document.getElementById("dropzone");
  const fileName = document.getElementById("file-name");
  const statusEl = document.getElementById("status");
  const errorEl = document.getElementById("error");
  const results = document.getElementById("results");
  const btnSample = document.getElementById("btn-sample");
  const imgOriginal = document.getElementById("img-original");
  const imgRectified = document.getElementById("img-rectified");
  const markdownEl = document.getElementById("markdown");
  const metaEl = document.getElementById("meta");
  const dlMd = document.getElementById("dl-md");
  const dlPdf = document.getElementById("dl-pdf");

  function setError(msg) {
    if (!msg) {
      errorEl.classList.add("hidden");
      errorEl.textContent = "";
      return;
    }
    errorEl.textContent = msg;
    errorEl.classList.remove("hidden");
  }

  function setStatus(msg) {
    statusEl.textContent = msg || "";
  }

  async function convert(formData) {
    setError("");
    setStatus("Working — warp, cleanup, OCR…");
    results.classList.add("hidden");

    const res = await fetch("/api/convert", { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setStatus("");
      setError(data.error || "Conversion failed.");
      return;
    }

    imgOriginal.src = data.original_url;
    imgRectified.src = data.rectified_url;
    markdownEl.textContent = data.markdown || "";
    dlMd.href = data.markdown_url;
    dlPdf.href = data.pdf_url;
    metaEl.textContent = [
      `OCR: ${data.ocr_backend}`,
      data.warped ? "perspective corrected" : "no warp",
      `${data.line_count || 0} lines`,
      `${data.diagram_count || 0} diagrams`,
      (data.notes || []).join(" · "),
    ]
      .filter(Boolean)
      .join(" · ");

    results.classList.remove("hidden");
    setStatus("Done — download Markdown or PDF below.");
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function onFile(file) {
    if (!file) return;
    fileName.textContent = file.name;
    fileName.classList.remove("hidden");
    const fd = new FormData();
    fd.append("file", file);
    await convert(fd);
  }

  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    onFile(f);
  });

  btnSample.addEventListener("click", async () => {
    fileName.textContent = "whiteboard_sample.jpg";
    fileName.classList.remove("hidden");
    const fd = new FormData();
    fd.append("sample", "whiteboard");
    await convert(fd);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    onFile(f);
  });
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
})();

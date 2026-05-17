/* Path-landscape agent UI client */
(() => {
  const form = document.getElementById("analyze-form");
  const submitBtn = document.getElementById("submit-btn");
  const progressSection = document.getElementById("progress");
  const bar = document.getElementById("bar");
  const barLabel = document.getElementById("bar-label");
  const log = document.getElementById("event-log");

  // example-button quick fill
  document.querySelectorAll(".examples button[data-example]").forEach(b => {
    b.addEventListener("click", () => {
      document.getElementById("phenomenon").value = b.dataset.example;
    });
  });

  function logEvent(event, opts = {}) {
    // remove "current" class from old entries
    log.querySelectorAll("li.current").forEach(li => li.classList.remove("current"));
    const li = document.createElement("li");
    if (opts.error) li.classList.add("error");
    else li.classList.add("current");
    const tag = document.createElement("span");
    tag.className = "step-tag";
    tag.textContent = `[${event.step}]`;
    li.appendChild(tag);
    li.appendChild(document.createTextNode(" " + event.message));
    log.appendChild(li);
    li.scrollIntoView({ block: "end", behavior: "smooth" });
  }

  function setBar(percent) {
    bar.style.width = Math.max(0, Math.min(100, percent)) + "%";
    barLabel.textContent = percent + "%";
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const phenomenon = document.getElementById("phenomenon").value.trim();
    const n_paths = parseInt(document.getElementById("n_paths").value, 10) || 1500;
    const eps = parseFloat(document.getElementById("eps").value) || 0.45;
    if (!phenomenon) return;

    submitBtn.disabled = true;
    submitBtn.textContent = "Running…";
    progressSection.hidden = false;
    log.innerHTML = "";
    setBar(0);

    let jobId;
    try {
      const resp = await fetch("/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phenomenon, n_paths, eps }),
      });
      const j = await resp.json();
      if (!resp.ok || j.error) throw new Error(j.error || `HTTP ${resp.status}`);
      jobId = j.job_id;
    } catch (err) {
      logEvent({ step: "error", message: String(err.message || err) }, { error: true });
      submitBtn.disabled = false;
      submitBtn.textContent = "Analyze";
      return;
    }

    const evt = new EventSource(`/stream/${jobId}`);
    evt.onmessage = (e) => {
      let data;
      try { data = JSON.parse(e.data); } catch { return; }
      logEvent(data, { error: data.step === "error" });
      if (typeof data.percent === "number") setBar(data.percent);
      if (data.step === "done") {
        evt.close();
        setTimeout(() => { window.location.href = `/result/${jobId}`; }, 500);
      } else if (data.step === "error") {
        evt.close();
        submitBtn.disabled = false;
        submitBtn.textContent = "Analyze";
      }
    };
    evt.onerror = () => {
      logEvent({ step: "stream", message: "stream closed; reconnecting…" });
    };
  });
})();

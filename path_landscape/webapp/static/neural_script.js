/* 4-agent neural-circuit pipeline UI client.
 * Same SSE machinery as the emergence pipeline, redirects to /neural/result.
 */
(() => {
  const form = document.getElementById("neural-form");
  if (!form) return;
  const submitBtn = document.getElementById("submit-btn");
  const progressSection = document.getElementById("progress");
  const bar = document.getElementById("bar");
  const barLabel = document.getElementById("bar-label");
  const log = document.getElementById("event-log");

  document.querySelectorAll(".examples button[data-example]").forEach(b => {
    b.addEventListener("click", () => {
      document.getElementById("phenomenon").value = b.dataset.example;
    });
  });

  // Map pipeline `step` strings to an agent-tag pill for the event log.
  function agentTagFor(step) {
    if (step.startsWith("agent1")) return { num: 1, label: "Agent 1" };
    if (step.startsWith("agent2")) return { num: 2, label: "Agent 2" };
    if (step.startsWith("agent3")) return { num: 3, label: "Agent 3" };
    if (step.startsWith("agent4")) return { num: 4, label: "Agent 4" };
    return null;
  }

  function logEvent(event, opts = {}) {
    log.querySelectorAll("li.current").forEach(li => li.classList.remove("current"));
    const li = document.createElement("li");
    if (opts.error) li.classList.add("error");
    else li.classList.add("current");
    const agent = agentTagFor(event.step || "");
    if (agent) {
      const tag = document.createElement("span");
      tag.className = `agent-tag agent-${agent.num}`;
      tag.textContent = agent.label;
      li.appendChild(tag);
    }
    const stepTag = document.createElement("span");
    stepTag.className = "step-tag";
    stepTag.textContent = `[${event.step}]`;
    li.appendChild(stepTag);
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
    if (!phenomenon) return;

    submitBtn.disabled = true;
    submitBtn.textContent = "Running…";
    progressSection.hidden = false;
    log.innerHTML = "";
    setBar(0);

    let jobId;
    try {
      const resp = await fetch("/neural/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phenomenon, n_paths }),
      });
      const j = await resp.json();
      if (!resp.ok || j.error) throw new Error(j.error || `HTTP ${resp.status}`);
      jobId = j.job_id;
    } catch (err) {
      logEvent({ step: "error", message: String(err.message || err) }, { error: true });
      submitBtn.disabled = false;
      submitBtn.textContent = "Analyze neural circuit";
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
        setTimeout(() => { window.location.href = `/neural/result/${jobId}`; }, 500);
      } else if (data.step === "error") {
        evt.close();
        submitBtn.disabled = false;
        submitBtn.textContent = "Analyze neural circuit";
      }
    };
    evt.onerror = () => {
      logEvent({ step: "stream", message: "stream closed; reconnecting…" });
    };
  });
})();

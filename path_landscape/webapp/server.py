"""Flask server for the path-landscape agent UI."""
from __future__ import annotations

# Force a non-interactive matplotlib backend BEFORE any other module pulls in
# pyplot. The pipeline runs in a background thread, and the default macOS
# backend (`MacOSX`) raises if used outside the main thread.
import matplotlib
matplotlib.use("Agg", force=True)

import json
import os
import queue
import secrets
import threading
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from flask import (
        Flask, Response, jsonify, redirect, render_template,
        request, send_from_directory, url_for,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "path_landscape.webapp requires Flask. Install with `pip install flask markdown`."
    ) from exc

try:
    import anthropic
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "path_landscape.webapp requires the `anthropic` package."
    ) from exc

from ..agent.pipeline import analyze_emergence
from ..agent.neural_pipeline import analyze_neural_circuit
from ..agent.schemas import SystemSpec

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("PATH_LANDSCAPE_OUT", "./emergence_analysis_web")
)


def _markdown_to_html(text: str) -> str:
    try:
        import markdown
        return markdown.markdown(text, extensions=["extra", "sane_lists"])
    except Exception:
        # very basic fallback: preserve paragraph breaks
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        return "\n".join(f"<p>{p}</p>" for p in paras)


def create_app(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )
    output_root = output_root.resolve()   # send_from_directory needs absolute path
    output_root.mkdir(parents=True, exist_ok=True)
    app.config["OUTPUT_ROOT"] = output_root
    app.config["JOBS"] = {}            # job_id -> job dict

    # ---------------------------------------------------------------- routes

    @app.route("/")
    def index():
        # Neural-circuit mode is now the primary analysis; redirect to it.
        return redirect(url_for("neural_index"))

    @app.route("/classic")
    def classic_index():
        """The original general emergence-analysis pipeline."""
        api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
        return render_template("index.html", api_key_set=api_key_set)

    @app.route("/start", methods=["POST"])
    def start():
        data = request.get_json(force=True) or {}
        phenomenon = (data.get("phenomenon") or "").strip()
        if not phenomenon:
            return jsonify({"error": "phenomenon is required"}), 400
        try:
            n_paths = int(data.get("n_paths") or 1500)
            eps = float(data.get("eps") or 0.45)
        except (TypeError, ValueError):
            return jsonify({"error": "invalid n_paths or eps"}), 400
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return jsonify(
                {"error": "ANTHROPIC_API_KEY environment variable is not set"}
            ), 400

        job_id = secrets.token_hex(8)
        q: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        job = {
            "id": job_id,
            "status": "pending",
            "queue": q,
            "phenomenon": phenomenon,
            "n_paths": n_paths,
            "eps": eps,
            "result": None,
            "error": None,
            "events": [],
            "started_at": time.time(),
        }
        app.config["JOBS"][job_id] = job
        thread = threading.Thread(
            target=_run_job,
            args=(app, job_id),
            daemon=True,
            name=f"agent-{job_id}",
        )
        thread.start()
        return jsonify({"job_id": job_id})

    @app.route("/stream/<job_id>")
    def stream(job_id):
        job = app.config["JOBS"].get(job_id)
        if not job:
            return jsonify({"error": "unknown job"}), 404

        def gen():
            q = job["queue"]
            # First flush any events that already happened
            for ev in list(job["events"]):
                yield f"data: {json.dumps(ev)}\n\n"
            while True:
                try:
                    ev = q.get(timeout=20)
                except queue.Empty:
                    yield ": heartbeat\n\n"
                    continue
                if ev is None:
                    break
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("step") in ("done", "error"):
                    break

        return Response(
            gen(),
            mimetype="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    @app.route("/result/<job_id>")
    def result(job_id):
        job = app.config["JOBS"].get(job_id)
        if not job:
            return "Job not found.", 404
        if job["status"] == "error":
            return render_template("error.html", job=job,
                                   error=job.get("error") or "unknown error")
        if job["status"] != "done":
            return render_template("pending.html", job_id=job_id)
        res = job["result"]
        interpretation_html = _markdown_to_html(res["interpretation"])
        return render_template(
            "result.html",
            job_id=job_id,
            phenomenon=res["phenomenon_name"],
            phenomenon_summary=res["phenomenon_summary"],
            interpretation_html=interpretation_html,
            metrics=res["metrics"],
            spec=res["spec"],
            system_summary=res["system_summary"],
            landscape_summary=res["landscape_summary"],
            partial=res.get("partial", False),
            partial_error=res.get("error"),
        )

    @app.route("/file/<job_id>/<path:filename>")
    def file(job_id, filename):
        out_dir = app.config["OUTPUT_ROOT"] / job_id
        return send_from_directory(str(out_dir), filename)

    # ============================================================ neural circuit
    #
    # The 4-agent neural-circuit pipeline lives alongside the existing
    # emergence pipeline. Same job-queue + SSE machinery; different worker
    # and a different result template centred on the path representation.

    @app.route("/neural")
    def neural_index():
        api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
        return render_template("neural_index.html", api_key_set=api_key_set)

    @app.route("/neural/start", methods=["POST"])
    def neural_start():
        data = request.get_json(force=True) or {}
        phenomenon = (data.get("phenomenon") or "").strip()
        if not phenomenon:
            return jsonify({"error": "phenomenon is required"}), 400
        try:
            n_paths = int(data.get("n_paths") or 1500)
        except (TypeError, ValueError):
            return jsonify({"error": "invalid n_paths"}), 400
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return jsonify(
                {"error": "ANTHROPIC_API_KEY environment variable is not set"}
            ), 400

        job_id = secrets.token_hex(8)
        q: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        job = {
            "id": job_id,
            "kind": "neural",
            "status": "pending",
            "queue": q,
            "phenomenon": phenomenon,
            "n_paths": n_paths,
            "result": None,
            "error": None,
            "events": [],
            "started_at": time.time(),
        }
        app.config["JOBS"][job_id] = job
        thread = threading.Thread(
            target=_run_neural_job,
            args=(app, job_id),
            daemon=True,
            name=f"neural-{job_id}",
        )
        thread.start()
        return jsonify({"job_id": job_id})

    @app.route("/neural/result/<job_id>")
    def neural_result(job_id):
        job = app.config["JOBS"].get(job_id)
        if not job:
            return "Job not found.", 404
        if job["status"] == "error":
            return render_template("error.html", job=job,
                                   error=job.get("error") or "unknown error")
        if job["status"] != "done":
            return render_template("pending.html", job_id=job_id)
        res = job["result"]
        description_html = _markdown_to_html(res.get("description") or "")
        interp = res.get("interpretation")  # may be None if Agent 5 failed
        mechanism_html = _markdown_to_html(interp.get("mechanism", "")) \
            if interp else ""
        # Check whether the paper circuit schematic was generated.
        out_dir = output_root / job_id
        has_paper_circuit = (out_dir / "paper_circuit.png").exists()
        return render_template(
            "neural_result.html",
            job_id=job_id,
            phenomenon=res["phenomenon_name"],
            phenomenon_input=res["phenomenon_input"],
            paper=res["paper"],
            neural_system=res["neural_system"],
            spec=res["spec"],
            path_rep=res["path_representation"],
            description_html=description_html,
            interpretation=interp,
            mechanism_html=mechanism_html,
            has_paper_circuit=has_paper_circuit,
        )

    return app


def _run_job(app: Flask, job_id: str) -> None:
    job = app.config["JOBS"][job_id]
    q = job["queue"]
    output_root: Path = app.config["OUTPUT_ROOT"]
    out_dir = output_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def emit(step: str, percent: int, message: str) -> None:
        ev = {
            "step": step,
            "percent": int(percent),
            "message": message,
            "t": round(time.time() - job["started_at"], 2),
        }
        job["events"].append(ev)
        q.put(ev)

    try:
        emit("queued", 1, f"job queued for {job['phenomenon']!r}")
        job["status"] = "running"
        client = anthropic.Anthropic()

        result = analyze_emergence(
            job["phenomenon"],
            out_dir=str(out_dir),
            n_paths=job["n_paths"],
            eps=job["eps"],
            client=client,
            verbose=False,
            on_progress=emit,
        )

        spec: SystemSpec = result["spec"]
        partial = bool(result.get("partial"))
        L = result.get("landscape")
        job["result"] = {
            "phenomenon_name": spec.phenomenon_name,
            "phenomenon_summary": spec.phenomenon_summary,
            "interpretation": result["interpretation"],
            "metrics": result["metrics"],     # may be None on partial
            "spec": spec.to_dict(),
            "landscape_summary": (L.describe() if L is not None
                                  else "path landscape NOT computed "
                                       "(path extraction failed; see figure)"),
            "system_summary": result["system"].summary(),
            "partial": partial,
            "error": result.get("error"),
        }
        job["status"] = "done"
        # Final event was already emitted by analyze_emergence as "done".
        q.put(None)
    except Exception as exc:
        tb = traceback.format_exc(limit=4)
        job["status"] = "error"
        job["error"] = f"{exc}\n{tb}"
        emit("error", 100, f"failed: {exc}")
        q.put(None)


def _run_neural_job(app: Flask, job_id: str) -> None:
    """Worker for the 4-agent neural-circuit pipeline."""
    job = app.config["JOBS"][job_id]
    q = job["queue"]
    output_root: Path = app.config["OUTPUT_ROOT"]
    out_dir = output_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def emit(step: str, percent: int, message: str) -> None:
        ev = {
            "step": step,
            "percent": int(percent),
            "message": message,
            "t": round(time.time() - job["started_at"], 2),
        }
        job["events"].append(ev)
        q.put(ev)

    try:
        emit("queued", 1, f"neural-circuit analysis queued for "
                          f"{job['phenomenon']!r}")
        job["status"] = "running"
        client = anthropic.Anthropic()

        result = analyze_neural_circuit(
            job["phenomenon"],
            out_dir=str(out_dir),
            n_paths=job["n_paths"],
            client=client,
            verbose=False,
            on_progress=emit,
        )

        spec_dict = result["spec"]
        job["result"] = {
            "phenomenon_name": spec_dict.get("phenomenon_name",
                                             job["phenomenon"]),
            "phenomenon_input": job["phenomenon"],
            "paper": result["paper"],
            "neural_system": result["neural_system"],
            "spec": spec_dict,
            "path_representation": result["path_representation"],
            "description": result["description"],
            "interpretation": result["interpretation"],
        }
        job["status"] = "done"
        q.put(None)
    except Exception as exc:
        tb = traceback.format_exc(limit=4)
        job["status"] = "error"
        job["error"] = f"{exc}\n{tb}"
        emit("error", 100, f"failed: {exc}")
        q.put(None)


def run_server(host: str = "127.0.0.1", port: int = 5174,
               output_root: Path = DEFAULT_OUTPUT_ROOT) -> None:
    """Entry point for `python -m path_landscape.webapp`."""
    app = create_app(output_root=output_root)
    print(f"  path-landscape agent UI: http://{host}:{port}")
    print(f"  output directory       : {output_root.resolve()}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  WARNING: ANTHROPIC_API_KEY is not set; runs will fail until it is.")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    run_server()

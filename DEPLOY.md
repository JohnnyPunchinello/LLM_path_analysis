# Deploying the path-landscape webapp

The repo ships three deploy paths, sorted from quickest to most durable.

## 1. Local + public link (no account, ephemeral)

For demos. The link dies when you close the terminal.

```bash
# terminal 1
export ANTHROPIC_API_KEY=sk-ant-...
python serve_agent.py --port 5174

# terminal 2 (any of the three works — no signup, no API key)
ssh -R 80:localhost:5174 nokey@localhost.run         # prints https://<id>.lhr.life
# or
ssh -R 80:localhost:5174 serveo.net                  # prints https://<id>.serveo.net
# or
cloudflared tunnel --url http://localhost:5174       # prints https://<id>.trycloudflare.com
```

## 2. Render (one-click, free or $7/month)

The simplest durable option.

```bash
# 1. push the repo to GitHub if you haven't
git push origin main

# 2. go to https://dashboard.render.com -> New + -> Blueprint -> pick your repo
#    Render reads render.yaml automatically.

# 3. once the service is created, set the API key in the dashboard:
#    Service -> Environment -> Add Environment Variable
#       ANTHROPIC_API_KEY = sk-ant-...
```

Your public URL is `https://<service-name>.onrender.com`.

## 3. Fly.io (more control, pay-as-you-go)

```bash
brew install flyctl
fly auth signup
fly launch --no-deploy --copy-config           # reads fly.toml
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly deploy
```

Your public URL is `https://path-landscape.fly.dev`.

## 4. Bare Docker (any host)

```bash
docker build -t path-landscape .
docker run -d -p 8080:8080 \
    -e ANTHROPIC_API_KEY=sk-ant-... \
    -v $(pwd)/runs:/data/runs \
    path-landscape
```

---

## Env vars

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | required for the analysis pipeline | unset (UI loads, analysis fails) |
| `PATH_LANDSCAPE_OUT` | where per-job artifacts get written | `./emergence_analysis_web` (local) / `/data/runs` (docker) |
| `PORT` | port gunicorn binds to | `8080` |

## Disk layout

Each analysis job writes ~1–3 MB of artifacts (figures + JSON) to a job-specific subdirectory of `PATH_LANDSCAPE_OUT`. The 1 GB volume in `render.yaml` / `fly.toml` holds ~300–1000 jobs. Increase if you plan to keep more history.

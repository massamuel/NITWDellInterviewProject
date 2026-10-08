# NITW Dell — interview project, revamped

A Dell interview NLP project turned into a React demo and a containerized Keras inference service. The repository dataset is **MBTI personality classification (16 classes)**, rather than sentiment classification. This is an educational classifier, not a psychological assessment. The historical scripts remain for reference; use `backend/` for the new application.

## Architecture

Browser → React/Nginx → FastAPI → saved Keras model. Each API pod loads one immutable model at startup; readiness waits for load and warmup. Kubernetes distributes requests across pods. KEDA manages a single HPA with CPU demand and a scheduled capacity floor. The model uses TF-IDF plus a small dense neural network for CPU inference, replacing the original LSTM experiment.

## Train and run locally

Requires Python 3.11 or 3.12 and Node 22+. No cloud account required.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
python -m backend.train --epochs 15
pytest -q
uvicorn backend.app:app --port 8000
# In another terminal:
cd frontend
npm ci
npm run dev
```

Open http://localhost:5173. Interactive API docs: http://localhost:8000/docs.

Training removes URLs and explicit MBTI labels, deduplicates cleaned documents, and stratifies into 64% training / 16% validation / 20% testing. Vocabulary and IDF weights are learned only from training data. Early stopping uses validation loss; class weights address imbalance. Held-out accuracy, macro F1, per-class scores, majority baseline, split sizes, dataset hash and model version are saved in `artifacts/metadata.json`. The exported `.keras` file includes vectorization; serving applies the same shared cleaning function. Test data is evaluated only after training. These labels reflect self-reported forum users, and a short input differs from the long training posts; high performance or calibrated probabilities are not assumed.

Artifacts are ignored by Git. Train before building the API image. The deployment must use the same versioned model artifact on every pod. Neither raw training posts nor submitted text are included in API logs or metrics.

## Containers

```sh
# Optionally train in Docker instead of a local Python environment:
docker build --target trainer -t nitwdell-trainer .
docker run --rm -v "$PWD/artifacts:/app/artifacts" nitwdell-trainer --epochs 15
# Build and run API + UI with the trained artifact:
docker compose up --build
```

Open http://localhost:8080. The UI proxies `/api` to FastAPI through Nginx; no browser CORS configuration is needed for this path. For direct cross-origin API clients set `CORS_ORIGINS` to explicit allowed origins. Run one process per API pod to avoid duplicating model memory; inference is serialized inside each pod, with a bounded Uvicorn concurrency limit that returns 503 when saturated. Scale pods for concurrency.

## Kubernetes and cloud configuration

The Helm chart targets **existing** AKS, EKS, or GKE clusters. Cloud values specify registry patterns and a LoadBalancer service; they do not provision accounts, clusters, registries, or IAM. Replace every `YOUR_*` placeholder, push both images, and grant the cluster image-pull access. Use a dedicated namespace (the chart uses fixed service names).

Prerequisites: Kubernetes, Helm 3, Metrics Server, and KEDA installed in the cluster. Enable node autoscaling and enough quota to host up to 20 API pods plus the frontend; pod autoscaling alone does not add node capacity. Install the relevant cloud load balancer controller when required by your cluster.

```sh
docker build --target api -t YOUR_API_IMAGE .
docker build -t YOUR_WEB_IMAGE frontend
# Push to ACR / ECR / Artifact Registry after cloud CLI login.
docker push YOUR_API_IMAGE
docker push YOUR_WEB_IMAGE
helm lint deploy/chart
helm upgrade --install nitwdell deploy/chart --namespace nitwdell --create-namespace   -f deploy/clouds/azure.yaml
# Substitute aws.yaml or gcp.yaml for EKS / GKE.
kubectl -n nitwdell get pods,svc,scaledobject,hpa
```

Use immutable image tags or digests. CPU scaling targets 60% of the 500m CPU request, between 2 and 20 pods. Example scheduled scaling maintains at least 6 pods weekdays 09:00–18:00 in `America/Los_Angeles`; CPU demand may raise replicas further. Change `scaling.schedule` in Helm values for launch windows and prewarm before the event. Scheduled values and resource requests are starting points requiring measurement. Scale-down has a 5-minute stabilization window. Do not create a second HPA for the API. For a public deployment configure ingress with a real hostname and TLS secret, and add authentication/rate limits and cloud edge protection before inviting customers.

## Load testing and capacity

```sh
BASE_URL=https://YOUR_HOST k6 run --summary-export=load-summary.json load-tests/predict.js
kubectl -n nitwdell get hpa,pods -w
```

The scenario ramps to **1,000 concurrent virtual users**, sustains them for 5 minutes, then ramps down. Each user sends a request and waits one second; this is not a fixed 1,000 requests/second test. Initial acceptance goals: failure rate below 1%, p95 under 500 ms, p99 under 1 s. These are goals, not measured results. Run against the deployed frontend endpoint from a separate load generator with enough CPU/network capacity. Compare cold and prewarmed runs, capture replicas and node capacity, and monitor CPU/memory, ingress errors, model latency and end-to-end latency. API Prometheus metrics at `/metrics` expose successful/invalid/failed predictions and inference durations; install a Prometheus scraper to collect them. The histogram excludes proxy and request queue delay; k6 measures end-to-end latency. Tune concurrency limits, CPU requests, minimum/maximum replicas and prewarming from observed results.

Docker, Kubernetes and k6 are required to validate deployment and the full 1,000-user target. Local unit tests do not establish cluster capacity.

## Validation

`pytest -q` checks input validation, prediction shape, failure handling, metrics and health. With artifacts present it also reloads the real saved model and calls the API. `npm run build` checks the React production bundle. CI validates API unit tests, the frontend build and cloud Helm rendering; full training and cloud load testing are separate, deliberate jobs.

## Initial local results (October 8, 2026)

Training stopped after 12 epochs and restored the best validation-loss weights. On 1,735 held-out rows: accuracy **35.33%**, macro F1 **0.2145**, majority baseline **21.10%**. This is a starter baseline with weak minority-class performance; inspect `docs/model-evaluation.json` before presenting model quality. Artifacts were trained locally and remain outside Git.

For a sequential in-process latency sanity check run `python -m scripts.benchmark_local`. This excludes network, ingress and concurrent demand; results are saved to `docs/local-latency.json`. The separate k6 scenario is the deployment capacity test.

Scaling references: [KEDA CPU scaler](https://keda.sh/docs/2.21/scalers/cpu/) and [KEDA scheduled scaling](https://keda.sh/docs/2.21/scalers/cron/). Model export follows [Keras model serialization](https://keras.io/guides/serialization_and_saving/).

## Future Python cloud infrastructure

`infra/` contains Python starter modules for Azure, AWS and GCP. Provisioning is intentionally unimplemented: there are no cloud credentials, resource definitions or account configurations. See [the starter guide](infra/README.md). This scaffolding is for a later deployment decision; no local VM is required.

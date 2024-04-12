# Docker Compose Deployment

## Quick Start

This is a simple example that shows you how to connect `nm-vllm` with production monitoring via Prometheus and Grafana via Docker Compose.

## Install

Make sure you have Docker and Docker Compose installed.
- [`docker`](https://docs.docker.com/engine/install/)
- [`docker compose`](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

> Note: `nm-vllm` requires a GPU. Make sure you have installed the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Deploy

### Launch

Launch your services:

```bash
docker compose up
```

### Submit Sample Workload

Submit some sample requests to the server:
```bash
# download sample dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# install client reqs
python3 -m venv client-env
source client-env/bin/activate
pip install -r client/requirements.txt

# submit sample workload
python3 client/client.py --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 3.0
```

### Monitor Metrics in Grafana

Navigate to the Grafana Client [`http://localhost:3000`](http://localhost:3000) and log in with the default username (`admin`) and password (`admin`).

#### Add Prometheus Data Source

There are three steps.
1. Navigate to [`http://localhost:3000/connections/datasources/new`](http://localhost:3000/connections/datasources/new) and select Prometheus. 

2. On Prometheus configuration page, add `http://prometheus:9090` as the `Prometheus Server URL` in `Connection`.

3. Click `Save & Test`. You should get a green check saying "Successfully queried the Prometheus API."

#### Setup Dashboard

Navigate to [`http://localhost:3000/dashboard/import`](http://localhost:3000/dashboard/import), upload `grafana.json` for overview metrics, selecting the `prometheus` datasource.

## Prometheus Metric Definitions

The Prometheus Metrics are exposed at the `/metrics` endpoint.

### System State

The following system state information is exposed:

| Metric | Type | Definition |
|--------|------|------------|
| `vllm:num_requests_running` | `gauge` | Number of requests in the `RUNNING` state (in decode "generation" phase)    | 
| `vllm:num_requests_waiting` | `gauge` | Number of requests in the `WAITING` state (not get in "generation" phase)   |
| `vllm:num_requests_swapped` | `gauge` | Number of requests in the `SWAPPED` state (evicted from "generation" phase) |
| `vllm:gpu_cache_usage_perc` | `gauge` | Percentage of GPU KV cache memory that is currently utilized |
| `vllm:cpu_cache_usage_perc` | `gauge` | Percentage of CPU KV cache memory that is currently utilized |

### Token Metrics

The following token metrics information is exposed:
| Metric | Type | Definition |
|--------|------|------------|
| `vllm:prompt_tokens_total`        | `counter`    | Number of prompt tokens processed |
| `vllm:generation_tokens_total`    | `counter`    | Number of generation tokens processed | 


### Request Metrics

| Metric | Type | Definition |
|--------|------|------------|
| `vllm:time_to_first_token_seconds`    | `histogram`   | Histogram of first token latency in seconds (often called TTFT) |
| `vllm:time_per_output_token_seconds`  | `histogram`   | Histogram of next token latency in seconds (often called TPOT) |
| `vllm:e2e_request_latency_seconds`    | `histogram`   | Histogram of end-to-end request latency in seconds |
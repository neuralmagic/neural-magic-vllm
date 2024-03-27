# vLLM + Prometheus/Grafana 

## Log Frequency and General Info
By default, FastAPI server that vLLM is hosted on logs at a defined frequency `t`, described in `vllm/entrypoints/openai/api_server.py` (currently set at 10 s). The async call will invoke both python logger logs and prometheus logs. 
Metric info per `t` interval are shown in both. Python logger shows instataneous info, and prometheus + graphana can show its history with more custom user-defined metrics (eg. queries per second). 

Example python log:
```
INFO 03-07 21:49:49 metrics.py:211] Avg prompt throughput: 603.9 tokens/s, Avg generation throughput: 377.9 tokens/s, Running: 17 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 1.8%, CPU KV cache usage: 0.0%
```

Example promethus + graphana log visualization:
![Grafana Dashboard Image](./assets/overview.png)


## Quick Start

This is a simple example that shows you how to connect vLLM metric logging to the Prometheus/Grafana stack. For this example, we launch Prometheus and Grafana via Docker. You can checkout other methods through [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/) websites. 

Install: 
- [`docker`](https://docs.docker.com/engine/install/)
- [`docker compose`](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

### Comamnds

Prometheus metric logging is enabled by default in the OpenAI-compatible server. Launch via the entrypoint:
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-v0.1 \
    --max-model-len 2048 \
    --disable-log-requests
```

Launch Prometheus and Grafana servers with `docker compose`:
```bash
cd examples/production_monitoring/
docker compose up
```

Submit some sample requests to the server:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 benchmarks/benchmark_serving.py \
    --model mistralai/Mistral-7B-v0.1 \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --endpoint /v1/completions \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 3.0 \
    --backend openai

```

Navigating to [`http://localhost:8000/metrics`](http://localhost:8000/metrics) will show the raw Prometheus metrics being exposed by vLLM.
Note: If using a remote machine, some servers may not be tunneled into the local machine. Run 
`ssh {user}@{ip} -L {port}:localhost:{port}`

## Grafana Dashboard

Navigate to [`http://localhost:3000`](http://localhost:3000). Log in with the default username (`admin`) and password (`admin`).

### Add Prometheus Data Source

Navigate to [`http://localhost:3000/connections/datasources/new`](http://localhost:3000/connections/datasources/new) and select Prometheus. 

On Prometheus configuration page, we need to add the `Prometheus Server URL` in `Connection`. For this setup, Grafana and Prometheus are running in separate containers, but Docker creates DNS name for each containers. You can just use `http://prometheus:9090`.

Click `Save & Test`. You should get a green check saying "Successfully queried the Prometheus API."

### Dashboard Setup - Importing from json

Dashboards can be imported from `json` config files. Navigate to [`http://localhost:3000/dashboard/import`](http://localhost:3000/dashboard/import), upload `vllm-metrics-overview.json` for overview metrics; `vllm-metrics-realtime.json` for real-time metrics and select the `prometheus` datasource. 

## Dashboards
Two dashboards - overview and real-time - can be imported from 'json'. Overview dashboard shows metrics over the course of some interval (default set to 24 hours), and real-time shows by default the last 5 mins. 

### Overview Metrics 
Metrics are all time-series, where the x-axis is time (by default `t` minus 24 hours to current) and y-axis values varies by dashboard.
Some time series use percentile and average values - 99th, 95th, 90th, 50th. This will be referred as default percentiles
 
- E2E request latency 
    * Shows default percentiles for the duration from the start to the end of the request to vLLM over time
- Token Throughout
    * Shows the prompt_token/s and generation_token/s over time 
- Time Per Output Token Latency
    * Shows the default percentiles for the time it takes to output any token as a part of the response
- Scheduler State
    * vLLM uses scheduler to manage jobs, each job has its own state. Simplest states are `running`, `swapped`, and `waiting`. The dashboard here shows its counts over time
- Time to First Token Latency
    * First tokens is a good time estimate to see the first streamed response token from the server. Default percentiles are uses over time. 
- Cache Utilization 
    * Shows both the CPU and GPU cache usage over time
- System GPU Utilization
    * Shows the percent of GPU resources used, analogous to `nvidia-smi` output for each recognizable GPU(s)


### Real-Time Metrics 
Consists of two dropdowns - Inference Request Metrics and Hardware Infrastructure Metrics. The logged metrics are shown as time-series or histogram. Time-series by default shows the history of the last 5 mins. Histogram shows the distribution of last 5 mins entries. 

#### Infernece Requesst Metrics
- Average Prompt Throughput
    * The number of completed prompts average over a defined frequency (usually 10s, set in code leve).
- Average Generation Throughput
    * THe number of generation
- Prompt Tokens
    * The rate of the number of prompt tokens over time
-  Generation Tokens
    * The rate of the number of generated over time
- Scheduler Running/Swapped/Waiting
    * The number of running/swapped/waiting jobs
- Inference Compute Duration
    * Histogram showing time for a job to in the `RUNNING` state over the default of 5 min
- Inference Queue Duration
    * Histogram showing the 
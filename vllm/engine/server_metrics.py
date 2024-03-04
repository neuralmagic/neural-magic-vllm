from aioprometheus import Counter, Histogram

INFERENCE_REQUEST_SUCCESS_COUNTER = Counter('vllm:inference_request_success', 'Number of successful inference calls')
INFERENCE_REQUEST_ABORTED_COUNTER = Counter('vllm:inference_request_aborted', 'Number of aborted inference calls')

INFERENCE_STATE_DURATION_HISTOGRAM = Histogram("vllm:inference_queue_duration", "Histogram of time in SequenceStatus")
 
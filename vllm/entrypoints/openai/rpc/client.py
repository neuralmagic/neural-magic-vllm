import pickle
from typing import Any, AsyncIterator, Mapping, Optional

import zmq
import zmq.asyncio

from vllm.config import (DecodingConfig, ModelConfig, ParallelConfig,
                         LoRAConfig, SchedulerConfig)
from vllm.entrypoints.openai.rpc import (
    RPC_REQUEST_TYPE, VLLM_RPC_HEALTHY_STR, VLLM_RPC_SUCCESS_STR, 
    RPCAbortRequest, RPCGenerateRequest, RPCUtilityRequest)
from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import _init_tokenizer_from_configs

class RPCClient:
    def __init__(self, port: int):
        self.context = zmq.asyncio.Context()
        self.path = f"tcp://localhost:{port}"
    
    async def setup(self):
        """Setup the client before it starts sending server requests."""
        
        # Wait until server is ready.
        await self.wait_for_server()

        # Get the configs.
        self.model_config = await self.get_model_config()
        self.decoding_config = await self.get_decoding_config()

        # Create the tokenizer group.
        self.tokenizer_group = _init_tokenizer_from_configs(
            model_config=self.model_config,
            scheduler_config=(await self.get_scheduler_config),
            parallel_config=(await self.get_parallel_config()),
            enable_lora=bool(await self.get_lora_config),
        )


    def close(self):
        """Destroy the ZeroMQ Context."""
        self.context.destroy()

    async def _send_get_data_rpc_request(self, request: RPCUtilityRequest,
                                         expected_type: Any,
                                         error_message: str) -> Any:
        """Send an RPC request that is expecting data back."""

        # Connect to socket.
        socket = self.context.socket(zmq.constants.DEALER)
        socket.connect(self.path)

        # Ping RPCServer with GET_MODEL_CONFIG request.
        await socket.send(pickle.dumps(request))

        # Await the MODEL_CONFIG from the Server.
        data = pickle.loads(await socket.recv())

        if not isinstance(data, expected_type):
            socket.close()
            raise ValueError(error_message)

        socket.close()

        return data

    async def _send_one_way_rpc_request(self, request: RPC_REQUEST_TYPE,
                                        error_message: str):
        """Send one-way RPC request to trigger an action."""

        # Connect to socket.
        socket = self.context.socket(zmq.constants.DEALER)
        socket.connect(self.path)

        # Ping RPC Server with request.
        await socket.send(pickle.dumps(request, pickle.HIGHEST_PROTOCOL))

        # Await acknowledgement from RPCServer.
        response = pickle.loads(await socket.recv())

        if not isinstance(response, str) or response != VLLM_RPC_SUCCESS_STR:
            socket.close()
            raise ValueError(error_message)

        socket.close()

        return response

    async def get_tokenizer(self, lora_request: LoRARequest):
        # TODO: handle this via get data? - or avoid doing via RPC
        return self.tokenizer

    async def is_tracing_enabled(self):
        # TODO: what is this?
        return False

    async def wait_for_server(self):
        """Wait for the RPCServer to start up."""

        await self._send_one_way_rpc_request(
            request=RPCUtilityRequest.IS_SERVER_READY,
            error_message="Unable to start RPC Server.")

    async def get_model_config(self) -> ModelConfig:
        """Get the ModelConfig object from the RPC Server"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_MODEL_CONFIG,
            expected_type=ModelConfig,
            error_message="Could not get ModelConfig from RPC Server"
        )

    async def get_decoding_config(self):
        """Get DecodingConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_DECODING_CONFIG,
            expected_type=ModelConfig,
            error_message="Could not get DecodingConfig from RPC Server"
        )

    async def get_parallel_config(self):
        """Get ParallelConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_PARALLEL_CONFIG,
            expected_type=ModelConfig,
            error_message="Could not get ModelConfig from RPC Server"
        )
    
    async def get_scheduler_config(self):
        """Get SchedulerConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_SCHEDULER_CONFIG,
            expected_type=SchedulerConfig,
            error_message="Could not get SchedulerConfig from RPC Server"
        )

    async def get_lora_config(self):
        """Get LoRAConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_SCHEDULER_CONFIG,
            expected_type=LoRAConfig,
            error_message="Could not get LoRAConfig from RPC Server"
        )


    async def abort(self, request_id: str):
        """Send an ABORT_REQUEST signal to the RPC Server"""

        await self._send_one_way_rpc_request(
            request=RPCAbortRequest(request_id),
            error_message=f"RPCAbortRequest {request_id} failed")

    async def do_log_stats(self):
        """Send a DO_LOG_STATS signal to the RPC Server"""

        await self._send_one_way_rpc_request(
            request=RPCUtilityRequest.DO_LOG_STATS,
            error_message="RPCRequest DO_LOG_STATS failed.")

    async def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> AsyncIterator[RequestOutput]:
        """Send an RPCGenerateRequest to the RPCServer and stream responses."""

        # Connect to RPC socket for Request-Reply pattern,
        # Note that we use DEALER to enable asynchronous communication
        # to enable streaming.
        socket = self.context.socket(zmq.constants.DEALER)
        socket.connect(self.path)

        # Send RPCGenerateRequest to the RPCServer.
        await socket.send_multipart([
            pickle.dumps(
                RPCGenerateRequest(
                    inputs=inputs,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request),
                pickle.HIGHEST_PROTOCOL)
        ])

        # Stream back the results from the RPC Server.
        while True:
            message = await socket.recv()
            request_output = pickle.loads(message)

            if isinstance(request_output, Exception):
                socket.close()
                raise request_output

            if request_output.finished:
                break
            yield request_output

        yield request_output
        socket.close()

    async def check_health(self) -> None:
        """Raise if unhealthy"""

        # Connect to socket.
        socket = self.context.socket(zmq.constants.DEALER)
        socket.connect(self.path)

        # Ping RPCServer with CHECK_HEALTH request.
        await socket.send(pickle.dumps(RPCUtilityRequest.CHECK_HEALTH))

        # Await the reply from the server.
        # TODO: do we need an internal timeout here?
        # Or do we expect the external probe to timeout and let this chill?
        health_message = pickle.loads(await socket.recv())
        socket.close()

        if isinstance(health_message, Exception):
            raise health_message

        if health_message != VLLM_RPC_HEALTHY_STR:
            raise ValueError("Expected healthy response from backend but got "
                             "f{health_message}")

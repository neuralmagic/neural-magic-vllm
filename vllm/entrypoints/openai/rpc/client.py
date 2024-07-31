import pickle
from typing import AsyncIterator, Mapping, Optional

import zmq
import zmq.asyncio

from vllm.config import DecodingConfig, ModelConfig
from vllm.entrypoints.openai.rpc import (
    VLLM_RPC_PATH, VLLM_RPC_SUCCESS_STR, 
    RPCGenerateRequest, RPCAbortRequest, RPCUtilityRequest)
from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams


class RPCClient:

    # TODO: check if opening all these sockets is an antipattern?
    def __init__(self, tokenizer):
        # ZMQ context.
        self.context = zmq.asyncio.Context()

        # TODO: do the tokenizer properly.
        self.tokenizer = tokenizer
        self.decoding_config = DecodingConfig()        

    def close(self):
        """Destroy the ZeroMQ Context."""
        self.context.destroy()

    async def wait_for_server(self):
        """Wait for the RPCServer to start up."""

        # Connect to socket.
        socket = self.context.socket(zmq.constants.DEALER)
        socket.connect(VLLM_RPC_PATH)

        # Ping RPCServer with IS_SERVER_READY request.
        socket.send(pickle.dumps(RPCUtilityRequest.IS_SERVER_READY))

        # Await acknoledgement from RPCServer that it is ready.
        message = await socket.recv()
        response = pickle.loads(message)
        
        if (not isinstance(response, str) or 
            not response == VLLM_RPC_SUCCESS_STR):
            socket.close()
            raise ValueError(f"Unable to start RPC Server.")
    
        socket.close()
        
    async def get_model_config(self) -> ModelConfig:
        """Get the ModelConfig object from the RPC Server"""

        # Connect to socket.
        socket = self.context.socket(zmq.constants.DEALER)
        socket.connect(VLLM_RPC_PATH)

        # Ping RPCServer with GET_MODEL_CONFIG request.
        socket.send(pickle.dumps(RPCUtilityRequest.GET_MODEL_CONFIG))

        # Await the MODEL_CONFIG from the Server.
        model_config = await socket.recv()
        model_config = pickle.loads(model_config)

        if not isinstance(model_config, ModelConfig):
            socket.close()
            raise ValueError(
                "Expected ModelConfig object from RPC, but "
                f"got {model_config}")

        socket.close()

        return model_config

    async def get_tokenizer(self, lora_request: LoRARequest):
        # TODO: handle this via get data? - or avoid doing via RPC
        return self.tokenizer

    async def get_decoding_config(self):
        # TODO: handle this via get data? -  or avoid doing via RPC
        return self.decoding_config

    async def is_tracing_enabled(self):
        # TODO: what is this?
        return False

    async def abort(self, request_id: str):
        """Send an RPCAbortRequest to the RPC Server"""

        # Connect to socket.
        socket = self.context.socket(zmq.constants.DEALER)
        socket.connect(VLLM_RPC_PATH)

        # Ping RPCServer with RPCAbortRequest request.
        socket.send_multipart([
            pickle.dumps(RPCAbortRequest(request_id),
                         pickle.HIGHEST_PROTOCOL)
        ])

        # Await acknoledgement from RPCServer that it aborted.
        response = pickle.loads(await socket.recv())
        if (not isinstance(response, str) or 
            not response == VLLM_RPC_SUCCESS_STR):
            socket.close()
            raise ValueError(
                f"RPCAbortRequest of {request_id} failed.")

        socket.close()
        

    async def do_log_stats(self,):
        """Send a DO_LOG_STATS signal to the RPC Server"""

        # Connect to socket.
        socket = self.context.socket(zmq.constants.DEALER)
        socket.connect(VLLM_RPC_PATH)

        # Ping RPCServer with DO_LOG_STATS request.
        socket.send(pickle.dumps(RPCUtilityRequest.DO_LOG_STATS))

        # Await acknoledgement from RPCServer that it logged stats.
        response = pickle.loads(await socket.recv())
        if (not isinstance(response, str) or 
            not response == VLLM_RPC_SUCCESS_STR):
            socket.close()
            raise ValueError(f"Unable to start RPC Server.")
        
        socket.close()

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
        socket.connect(VLLM_RPC_PATH)


        # Send RPCGenerateRequest to the RPCServer.
        socket.send_multipart([
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

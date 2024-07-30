import asyncio
import pickle
import signal

import zmq
import zmq.asyncio

from vllm import AsyncLLMEngine
from vllm.entrypoints.openai.rpc import (VLLM_GENERATE_RPC_PATH,
                                         VLLM_GET_DATA_RPC_PATH,
                                         VLLM_IS_READY_RPC_PATH,
                                         VLLM_ABORT_RPC_PATH,
                                         VLLM_ABORT_RESPONSE_STR,
                                         VLLM_READY_RESPONSE_STR,
                                         VLLM_LOG_STATS_RPC_PATH,
                                         AbortRequest,
                                         RPCRequestType)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger('vllm.entrypoints.openai.rpc.server')


class RPCServer:

    # TODO: check if opening all these sockets is an antipattern.
    # Alternative, use a smaller number of sockets with conditioning on the 
    # data that is passed through the socket.
    def __init__(self, async_engine_args):
        # Initialize engine first.
        self.engine = AsyncLLMEngine.from_engine_args(
            async_engine_args, UsageContext.OPENAI_API_SERVER)

        # Initialize context.
        self.context = zmq.asyncio.Context()

        # Init socket for readiness state.
        self.is_ready_socket = self.context.socket(zmq.constants.REQ)
        self.is_ready_socket.bind(VLLM_IS_READY_RPC_PATH)

        # Init socket for generation.
        self.generate_socket = self.context.socket(zmq.constants.ROUTER)
        self.generate_socket.bind(VLLM_GENERATE_RPC_PATH)
        
        # Init socket for aborting requests.
        self.abort_socket = self.context.socket(zmq.constants.ROUTER)
        self.abort_socket.bind(VLLM_ABORT_RPC_PATH)

        # Init socket for do_log_stats requests.
        self.log_stats_socket = self.context.socket(zmq.constants.ROUTER)
        self.log_stats_socket.bind(VLLM_LOG_STATS_RPC_PATH) 

        # Init socket for simple data requests.
        self.get_data_socket = self.context.socket(zmq.constants.REP)
        self.get_data_socket.bind(VLLM_GET_DATA_RPC_PATH)

        # Setup polling so we can listen on both sockets.
        self.poller = zmq.asyncio.Poller()
        self.poller.register(self.generate_socket, zmq.constants.POLLIN)
        self.poller.register(self.abort_socket, zmq.constants.POLLIN)
        self.poller.register(self.get_data_socket, zmq.constants.POLLIN)
        self.poller.register(self.do_log_stats_socket, zmq.constants.POLLIN)

    def cleanup(self):
        """Shuts down the zmq context and closes all sockets"""
        self.context.destroy()
        del self.abort_socket
        del self.get_data_socket
        del self.generate_socket
        del self.log_stats_socket
        del self.is_ready_socket

    async def get_data(self, message):
        request_type = pickle.loads(message)

        if request_type == GetDataRequest.MODEL_CONFIG:
            data = await self.engine.get_model_config()
        else:
            raise ValueError(f"Unknown request type: {request_type}")

        await self.get_data_socket.send_multipart([
            pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
        ])
        
    async def do_log_stats(self, message):
        request_type = pickle.loads(message)

        if request_type == RPCRequestType.DO_LOG_STATS:
            await self.engine.do_log_stats()
        
        
    async def abort(self, identity, message):
        request = pickle.loads(message)

        # Abort the request in the llm engine.
        await self.engine.abort(request.request_id)

        # Send confirmation to the client.
        self.abort_socket.send_multipart([
            identity,
            pickle.dumps(VLLM_ABORT_RESPONSE_STR, pickle.HIGHEST_PROTOCOL),
        ])

    async def generate(self, identity, message):
        try:
            request = pickle.loads(message)

            results_generator = self.engine.generate(
                request.inputs,
                sampling_params=request.sampling_params,
                request_id=request.request_id)

            async for request_output in results_generator:
                self.generate_socket.send_multipart([
                    identity,
                    pickle.dumps(request_output, pickle.HIGHEST_PROTOCOL)
                ])
                
        except Exception as e:
            ### Notify client of all failures
            self.generate_socket.send_multipart(
                [identity, pickle.dumps(e, pickle.HIGHEST_PROTOCOL)])


    async def run_loop(self):
        """Inner RPC Server Loop"""

        # Notify the RPC Client that we are ready to receive requests.
        await self.is_ready_socket.send(pickle.dumps(VLLM_READY_RESPONSE_STR))
        self.is_ready_socket.close()

        # We need to keep around a strong reference to the task,
        # to avoid the task disappearing mid-execution as running tasks
        # can be GC'ed. Below is a common "fire-and-forget" tasks
        # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
        running_tasks = set()

        while True:
            # TODO: Why is this self?
            # TODO: Is it possible to have > 1 generate request per poll
            self.poll_future = self.poller.poll()
            socks = dict(await self.poll_future)

            # Handle generate request.
            if self.generate_socket in socks:
                identity, message = await self.generate_socket.recv_multipart()
                task = asyncio.create_task(self.generate(identity, message))
                running_tasks.add(task)
                task.add_done_callback(running_tasks.discard)

            # Handle abort request.
            if self.abort_socket in socks:
                identity, message = await self.abort_socket.recv_multipart()
                task = asyncio.create_task(self.abort(identity, message))
                running_tasks.add(task)
                task.add_done_callback(running_tasks.discard)

            # Handle get_data request.
            if self.get_data_socket in socks:
                message = await self.get_data_socket.recv()
                task = asyncio.create_task(self.get_data(message))
                running_tasks.add(task)
                task.add_done_callback(running_tasks.discard)

        # TODO: Do I need to close the generate / get_data sockets?

async def run_server(server: RPCServer):
    # Run with proper interrupt handling
    logger.info("Booting up vLLM zmq backend")

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.run_loop())

    def signal_handler() -> None:
        # Kill the server on interrupt / terminate
        server_task.cancel()

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("ZMQ Backend was interrupted")
    finally:
        # Clean up all the zmq resources before exiting
        server.cleanup()
    logger.info("vLLM ZMQ Backend shut down")


def run_rpc_server(async_engine_args):
    server = RPCServer(async_engine_args=async_engine_args)
    asyncio.run(run_server(server))

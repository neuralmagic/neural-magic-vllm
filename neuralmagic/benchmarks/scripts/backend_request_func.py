"""
Benchmark serving utilities for various end-points.

NOTE: This script is a version of benchmarks/backend_request_func.py from
 the upstream vllm repo at commit a4211a4dc.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    """
    Populate server_response instead of generated_text, if we don't
    want to burden the async process with decoding the server_response. This
    decoding can instead happen offline.
    """
    generated_text: str = None
    server_response: bytes = None
    success: bool = False
    latency: float = 0
    ttft: float = 0
    prompt_len: int = 0


def trim_prefix(text: str, prefix_str: str) -> str:
    assert len(text) >= len(prefix_str)
    assert text[:len(prefix_str)] == prefix_str
    return text[len(prefix_str):]


def trim_suffix(text: str, suffix_str: str) -> str:
    assert len(text) >= len(suffix_str)
    assert text[-1 * len(suffix_str):] == suffix_str
    return text[:-1 * len(suffix_str)]


async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        params = {
            "best_of": request_func_input.best_of,
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    data = None
                    async for part_data in response.content.iter_any():
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                        data = part_data
                    output.latency = time.perf_counter() - st

                    body = trim_prefix(data.decode("utf-8"), "data:")
                    output.generated_text = json.loads(body)["generated_text"]
                    output.success = True
                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

        if pbar:
            pbar.update(1)
        return output


class AsyncRequestVLLM:

    @staticmethod
    def stream_server_outputs():
        """
        This function is queried to set the `stream` option of the 
        server JSON payload.
        """
        return True

    @staticmethod
    def decode_server_response(server_response: bytes, prompt_len: int) -> str:
        """
        Decodes the server response and returns the text generated by
        the server in response to the prompt.
        """

        def try_json_decode(s: str) -> dict:
            try:
                return json.loads(s)
            except Exception as _:
                return None

        # When streaming, '\0' is appended to the end.
        assert (AsyncRequestVLLM.stream_server_outputs())
        body: str = trim_suffix(server_response.decode('utf-8'), "\0")

        # Most times we only have one JSON in the body.
        decoded_json = try_json_decode(body)
        if decoded_json is not None:
            return decoded_json["text"][0][prompt_len:]

        # Some times body contains more than one JSON.
        # These JSONs essentially contain the generated text and the
        # last of the JSONs has the entire generated text.
        json_starts = [m.start() for m in re.finditer('{\"text\":', body)]
        for json_start in reversed(json_starts):
            decoded_json = try_json_decode(body[json_start:])
            if decoded_json is not None:
                return decoded_json["text"][0][prompt_len:]

        raise ValueError(f"Cannot decode json body \n {body}")

    @staticmethod
    async def async_request_vllm(
        request_func_input: RequestFuncInput,
        pbar: Optional[tqdm] = None,
    ) -> RequestFuncOutput:
        api_url = request_func_input.api_url
        assert api_url.endswith("generate")

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            payload = {
                "prompt": request_func_input.prompt,
                "n": 1,
                "best_of": request_func_input.best_of,
                "use_beam_search": request_func_input.use_beam_search,
                # TODO (varun) : Make temperature configurable
                #"temperature": 0.0 if request_func_input.use_beam_search \
                #                   else 1.0,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": request_func_input.output_len,
                "ignore_eos": True,
                "stream": AsyncRequestVLLM.stream_server_outputs(),
            }
            output = RequestFuncOutput()
            output.prompt_len = len(request_func_input.prompt)

            ttft = 0
            st = time.perf_counter()
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        data = None
                        async for part_data in response.content.iter_any():
                            if ttft == 0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft
                            data = part_data

                        output.latency = time.perf_counter() - st
                        output.server_response = data
                        output.success = True

                    else:
                        output.success = False
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
                output.success = False

            if pbar:
                pbar.update(1)
            return output


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        assert request_func_input.best_of == 1
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload) as resp:
                if resp.status == 200:
                    data = None
                    async for part_data in resp.content.iter_any():
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                        data = part_data
                    output.latency = time.perf_counter() - st

                    body = trim_prefix(data.decode("utf-8"), "data:")
                    output.generated_text = json.loads(body)["text_output"]
                    output.success = True

                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert request_func_input.best_of == 1
        assert not request_func_input.use_beam_search

        payload = {
            "prompts": request_func_input.prompt,
            "max_new_tokens": request_func_input.output_len,
            "ignore_eos": True,
            "do_sample": True,
            "temperature":
            0.01,  # deepspeed-mii does not accept 0.0 temperature.
            "top_p": 1.0,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(url=request_func_input.api_url,
                                    json=payload) as resp:
                if resp.status == 200:
                    parsed_resp = await resp.json()
                    output.latency = time.perf_counter() - st
                    output.generated_text = parsed_resp[0]["generated_text"]
                    output.success = True
                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("v1/completions")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = trim_prefix(chunk.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            body = json.loads(chunk)
                            generated_text += body["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": AsyncRequestVLLM.async_request_vllm,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "tensorrt-llm": async_request_trt_llm,
}
def check_logprobs_close(outputs_0, outputs_1, name_0, name_1):
    num_tokens = max(len(outputs_0), len(len_outputs_1))

    for prompt_idx in range(num_tokens):
        output_ids_0, output_str_0, logprobs_0 = outputs_0[prompt_idx]
        output_ids_1, output_str_1, logprobs_1 = outputs_1[prompt_idx]

        for idx, (output_id_0, output_id_1) in enumerate(
                zip(output_ids_0, output_ids_1)):

            # If sequence is not an exact match ...
            if output_id_0 != output_id_1:
                # ... each predicted token must be in top N logprobs of the other's
                assert output_id_0 in logprobs_1[idx], (
                    f"Test{prompt_idx}:\n{name_0}:\t{output_str_0!r}\n{name_1}:\t{output_str_1!r}"
                )
                assert output_id_1 in logprobs_0[idx], (
                    f"Test{prompt_idx}:\n{name_0}:\t{output_str_0!r}\n{name_1}:\t{output_str_1!r}"
                )

                # Break out since sequences will now diverge.
                break
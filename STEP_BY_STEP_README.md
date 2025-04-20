## Introduction

Since you want to get your hands dirty, here's a quick guide on how to work do the verification flow step by step.

I'd also encourage you to check out [the verification README](/VERIFICATION_README.md) for more context.

## Setup

First, create a virtual environment at the root of the repository.

Activate the environment and install sglang:

```
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
pip install transformers==4.48.3
pip install datasets
```

You also need to set your `HF_TOKEN` environment variable to a token which has access to `meta-llama/3.1-8b-instruct`.  You can find mine in 1Password under Engineering.

```
export HF_TOKEN=...
```

## Example Script

Try running this script:

```
python toploc-scripts/minimal_example.py --disable-cuda-graph
```

**Note**: I've disabled CUDA graph because it introduces some kind of non-determinism in the prefill that makes verification occassionally fail (maybe 1 out of 6 times).  This is a new behavior compared to my testing from last week, so I'm hoping it's because I upgraded toploc to v0.1.4, and this is easily resolved.  I've got a ticket to look into it.

## How to do it "By Hand"

First, you have to start the server in `--toploc-verification` mode.

Here is the command you can run to start the server:
```
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 3001 --toploc-verification --toploc-verification-topk 128 --log-level debug --disable-cuda-graph
```

Now, you can send an inference request to the server, and you can see the fingerprint in the response:
```
import json
import openai

params = {
    "temperature": 0,
    "seed": 42,
}

client = openai.Client(base_url=f"http://127.0.0.1:3001/v1", api_key="None")

prompt = "What is the capital of Bulgaria?"
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": prompt},
    ],
    **params
)
response_dump = response.model_dump()
print("Response received:")
print(json.dumps(response_dump, indent=4))
```

The response will contain a `toploc_verification_fingerprints` array:
```json
{
    "choices": [
        {
            "message": {
                "content": "Sofia",
                "toploc_verification_fingerprints": ["...", "..."]
            }
        }
    ]
}
```

There are typically two.  We're only interested in the last one.

Now, we need to validate the fingerprint.  How do we do that?  By sending it to a verification instance running the same model, along with the original prompt and response.

To build the verification request, you have to:

1. Append to the messages array, so that it includes both the original prompt and the assistant's response:
```json
{
    "role": "user",
    "content": "What is the capital of Bulgaria?"
},
// This is the response ---v to this ----^
{
    "role": "assistant",
    "content": (the response)
}
```

2. Set `max_tokens` to 0. This is what makes it a prefill.

3. Set `toploc_verification_fingerprint_to_validate` to the last fingerprint in the `toploc_verification_fingerprints` array.

The verification instance will respond with a `toploc_verification_fingerprint_validation_result`, which will look something like this (but serialized as a string):

```json
{
    "exp_mismatches": 1,
    "mant_err_mean": 0.75,
    "mant_err_median": 0.75,
}
```

These error statistics are what is interpreted to determine if this is a verification pass or verification failure.


The implementation of this fork would have been much simpler if we had worked with the SGLang module directly in Python (i.e.; `import sglang`), but that would have entailed basically rewriting how our workers work.

So, unfortunately, I had to devote a lot of code to pass-thrus to/from the API layer of SGLang.

**Important Note On Prefill Replication**

I am prefilling the original prompt + response by appending an assistant message to the messages array.

This may not work in all cases: i.e.; tools in the request for example.

Another concern is fragility.  Suppose that SGLang changes the way it parses or generates responses, the model  updates its chat template, etc etc.  Then, the same messages array will not correspond to the same token ID inputs.

For both of these reasons, I've implemented two other features to make pre-fill more robust:
1. `return_input_ids` - returns the token IDs of the prompt if included in the request
2. `return_output_ids` - returns the token IDs of the response if included in the request

Then, pre-fill request will simply take:
`input_ids[:-1] + output_ids + EOT`, which is a far more reliable way to replicate prompt + response.

## How The Fork Works

It turns out the EAGLE speculative decoding has a lot in common with verification.

SGLang has an internal flag called `CaptureHiddenMode`, which has values of either `NONE`, `LAST`, or `FULL`.

These values refer to which of the hidden layers of the LLM should be "captured" so that when inference is complete, their values are accessible for use in EAGLE speculative sampling.

Ordinarily, `CaptureHiddenMode` is set to `NONE` unless some version of EAGLE is enabled.

I modified this logic so that verification is enabled, `CaptureHiddenMode` is set to at least `LAST`.

Then, after inference is complete, I move the last hidden layer to the CPU.

At this point, the code path diverges.

1. If we are performing inference, I use the hidden layer to generate the toploc fingerprint, and return it with the response.

2. If we are verifying a fingerprint, I compare the hidden layer with the toploc fingerprint, and return the result with the response.

The "core" logic of fingerprint verification and fingerprint generation are part of the toploc library, which I have added as a dependency.

I could have re-implemented it all from scratch because I understand the math, but that seemed like a wasteful exercise when we have a working implementation available.

## What Makes The Fork Tricky

SGLang takes requests and puts them into a general purpose task scheduler.

Then, SGLang attempts to take tasks of the same kind and group them into batches.

The batches store information in arrays, and in some cases the batch objects store nested data structures as flat arrays and then use array indices to set the boundaries between contiguous regions that represent individual items in the batch.  There are also a few different kinds of objects that are at the batch level (`ScheduleBatch`, `BatchTokenIDOut`, `LogitProcessorOutput`, etc.)

So, there is quite a bit of "glue" required to correctly assemble the various kinds of batches and then slice them back apart into requests once inference is complete.

Then, there's additional layers of pass-thru to the API layer of SGLang.

However, there was plenty of precedent for how to do this kind of stuff by looking at the EAGLE code.

So, you'll see a lot of code that is basically the same as EAGLE code that lives right next to it, if you explore up or down a few lines.  This is especially the case when it comes to handling `CaptureHiddenMode` and dealing with `hidden_states`.

This is also a divergent codepath for CUDA Graph Runner that needs to be properly handled.

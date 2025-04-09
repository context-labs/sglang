# Verification Rollout Phases

## Phase 1 (MVP)
Stand up a simple end-to-end verification system with a single verifier for `3.1-8b-instruct`.

Components of System:
* Benchmarker running fork of sglang inference engine
* Verification instance running fork of sglang inference engine
* apps/relay sends verification requests to verification instance
* verification instance performs verification
* results are collected and dashboarded for visibility
* automatic calculation of operator "probability you're a spoofer" score based on verification results

I'm fully anticipating we'll uncover problems along the way and will need to make refinements as we go.

## Phase 2 (MVP -> P)

Address shortcomings of the MVP implementation.  These would include:
* Switching to Token-ID based verification instead of using request and response prompt messages
* Rolling out verification on a few more models on a trial basis
* Attempting the MOE fingerprinting optimization to cut down on the cost of verifying models like Deepseek-v3

## Phase 3 (Decentralization)

* Allow highly trusted operators to run decentralization instances
 themselves
* This trust level can be dynamic or manual
* Create a "admin slash dashboard" (slashboard?) that lists operators, the likelihood they are a spoofer, and then allows admins to confirm and apply punitive measures (like delisting and/or slashing)

## Other Tasks
Unsure which phases these should be in:
* Forking VLLM and/or ollama?
* Collect error threshold measurements over batches, implement tooling to automate the process.
* Run some additional experiments with EAGLE speculative decoding
* Investigate and resolve a problem I noticed with CUDA graph

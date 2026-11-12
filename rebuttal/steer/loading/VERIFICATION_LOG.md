# Verification log: steer/loading (Qwen3.5-9B loading, pod pin set)

Entries for the lead to append to the project VERIFICATION_LOG.md. Scratch evidence lives under
/private/tmp/claude-501/-Users-unrulyabstractions-work-papers-breaking-character/d77c369f-47a9-4de4-8db2-b38a6d6a51dd/scratchpad/q35x/.

## 2026-11-12

1. persona_load.py (this folder). HOW: ran test_loading_tiny.py on CPU with torch 2.11.0,
   transformers 5.16.1, peft 0.20.0, accelerate 1.15.0. It builds tiny random Qwen3.5 and Gemma-3 models
   with the real configs' structure. I read all 16 PASS lines and `{"n_pass": 16, "n_checks": 16}`, exit 0.
   RESULT: VERIFIED on tiny models. UNVERIFIED on a GPU and with real base weights.
2. The six real Qwen3.5-9B adapter files on a full-size Qwen3_5ForCausalLM (8,953,803,264 params, built
   from the real unsloth config in uninitialised bf16 memory). HOW: scratchpad
   q35x/scripts/real_adapters_fullsize.py. I read out/real_adapters_fullsize.json.
   RESULT: VERIFIED. Every adapter tensor landed in the model and equals the file: bad-legal, bad-medical
   and extreme-sports 256/256 as they are; risky-financial and reward-hacks 256/256 after the remap; cheese
   352/352 after the remap. Without the remap, PEFT only warned "missing adapter keys" and verification raised.
3. requirements-pod.txt. HOW: uv pip compile for python 3.12 / x86_64-manylinux_2_28 with the cu128 extra
   index. I read the 66 resolved pins back from the file. RESULT: VERIFIED that it resolves. UNVERIFIED that
   it installs and imports on a RunPod pod.
4. Family compatibility, 4.57.6 vs 5.16.1. HOW: scratchpad q35x/scripts/fam_inspect.py, gen_cfg.py and
   tiny_gemma.py; I read the JSON outputs. RESULT: VERIFIED for the items checked. For every family the
   CausalLM class and parameter names match, the 8 rendered EM prompts give identical token ids, the
   effective sampling parameters match, and every LoRA module name exists. Gemma-3 legacy-key full
   fine-tunes load with zero missing keys and equal logits. The Gemma-3 decoder path changed and is handled
   in decoder_layers(). Real-weight generation is UNVERIFIED for every family.
5. Independent `verifier` agent: NOT RUN. This subagent has no tool to spawn agents. Status UNVERIFIED by
   an independent check.

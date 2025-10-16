from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

# Prepare the input to the model
prompt = "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Set to False to strictly disable thinking
)

# Generate outputs
llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
    speculative_config={
        "model": "Qwen/Qwen3-1.7B",
        "num_speculative_tokens": 8, # K
    },
    disable_log_stats=False, # Ensure stats/metrics are not disabled
)
# Generate outputs
outputs = llm.generate([text], sampling_params)

for output_idx, output in enumerate(outputs):
    print(f"--- Output for Prompt {output_idx + 1} ---")
    prompt_token_ids = output.prompt_token_ids
    completion_output = output.outputs[0]
    generated_token_ids = completion_output.token_ids
    generated_text = completion_output.text

    # Access the underlying Hugging Face tokenizer
    hf_tokenizer = llm.llm_engine.tokenizer.tokenizer 
    
    print(f"Original Prompt: {output.prompt!r}")
    # print(f"Prompt Token IDs: {prompt_token_ids}") # Can be verbose
    # raw_prompt_with_special_tokens = hf_tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
    # print(f"Raw Prompt (with special tokens): {raw_prompt_with_special_tokens!r}") # Can be verbose
    print(f"Generated text: {generated_text!r}\n")

    if output.metrics and hasattr(output.metrics, 'spec_token_acceptance_counts'):
        print(f"Metrics: {output.metrics}")
        spec_acceptance_counts = output.metrics.spec_token_acceptance_counts
        
        K_config = 0
        if llm.llm_engine.speculative_config:
            K_config = llm.llm_engine.speculative_config.num_speculative_tokens
        
        print(f"Configured num_speculative_tokens (K_config): {K_config}")
        print(f"Speculative Token Acceptance Counts per verification step: {spec_acceptance_counts}")

        token_origins = []
        current_gen_token_idx = 0 # Index into generated_token_ids

        for step_idx, num_accepted_in_this_verification_step in enumerate(spec_acceptance_counts):
            if current_gen_token_idx >= len(generated_token_ids):
                break

            # Mark accepted draft tokens
            for i in range(num_accepted_in_this_verification_step):
                if current_gen_token_idx < len(generated_token_ids):
                    token_origins.append(f"Draft (Accepted in v-step {step_idx + 1})")
                    current_gen_token_idx += 1
                else:
                    # This case implies spec_acceptance_counts suggests more tokens than were generated.
                    # Should ideally not happen if data is consistent.
                    print("Warning: spec_acceptance_counts mismatch with generated_token_ids length during draft token marking.")
                    break 
            
            if current_gen_token_idx >= len(generated_token_ids):
                break # All generated tokens accounted for or mismatch

            # The next token is the one chosen/verified by the target model.
            token_origins.append(f"Target (Verified/Generated post v-step {step_idx + 1})")
            current_gen_token_idx += 1
        
        # Any remaining tokens not covered by the loop are typically target-generated (e.g., if generation ends).
        while current_gen_token_idx < len(generated_token_ids):
            token_origins.append("Target (Trailing)")
            current_gen_token_idx += 1
        
        if len(token_origins) != len(generated_token_ids):
            print(f"Warning: Token origins list length ({len(token_origins)}) does not match generated tokens length ({len(generated_token_ids)}). This may indicate an issue in provenance tracking logic or unexpected metrics data.")


        print("\nToken Provenance:")
        for i, token_id in enumerate(generated_token_ids):
            token_text = hf_tokenizer.decode([token_id], skip_special_tokens=True)
            origin = token_origins[i] if i < len(token_origins) else "Error: Origin mapping failed (length mismatch)"
            print(f"  Token {i + 1}: ID={token_id}, Text={token_text!r}, Origin={origin}")
        print("-" * 30)

    elif output.metrics:
        print(f"Metrics available but 'spec_token_acceptance_counts' attribute is missing: {output.metrics}")
        print("Assuming all generated tokens are from the target model.")
        token_origins = ["Target"] * len(generated_token_ids)
        print("\nToken Provenance (assuming all Target):")
        for i, token_id in enumerate(generated_token_ids):
            token_text = hf_tokenizer.decode([token_id], skip_special_tokens=True)
            origin = token_origins[i]
            print(f"  Token {i + 1}: ID={token_id}, Text={token_text!r}, Origin={origin}")
        print("-" * 30)
    else:
        if hasattr(llm.llm_engine, 'speculative_config') and llm.llm_engine.speculative_config:
             print("Metrics not available in output, but speculative decoding is configured.")
             print("Assuming all generated tokens are from the target model by default.")
        else:
            print("Speculative decoding is not configured, or metrics are unavailable. All tokens are from the target model.")

        token_origins = ["Target"] * len(generated_token_ids)
        print("\nToken Provenance (assuming all Target):")
        for i, token_id in enumerate(generated_token_ids):
            token_text = hf_tokenizer.decode([token_id], skip_special_tokens=True)
            origin = token_origins[i]
            print(f"  Token {i + 1}: ID={token_id}, Text={token_text!r}, Origin={origin}")
        print("-" * 30)
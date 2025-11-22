from __future__ import annotations

import os
import sys
import asyncio
import dspy
from pathlib import Path
from typing import List

# Add the current directory to sys.path to allow importing from personas package
sys.path.append(os.getcwd())

try:
    from personas.blp_prompt_optimization import optimize_blp_prompt, BLPLearningExample, DSPyBLPExtractor
    from personas.prompts import read_prompt
except ImportError as e:
    print(f"Error importing application modules: {e}")
    print("Make sure you are running this script from the project root directory.")
    sys.exit(1)

# Dummy data for testing
SAMPLE_TRANSCRIPT = """
Doctor: Good morning. How are you doing today?
Patient: I've been feeling really down lately, honestly. Just no energy to do anything.
Doctor: I see. How long has this been going on?
Patient: About three weeks now. I just can't seem to get out of bed some days.
Doctor: Have you noticed any changes in your appetite or sleep?
Patient: Yeah, I'm not eating much. And I wake up at 3 AM every night and can't fall back asleep.
"""

SAMPLE_CASE = """
CASE SUMMARY:
Patient is a 35-year-old female presenting with symptoms of depression.
Primary complaints: low mood, anhedonia, fatigue, insomnia (early morning awakening), and poor appetite.
Duration: 3 weeks.
History: No prior history of depression. No substance use.
Social: Lives alone, works as a graphic designer.
Diagnosis: Major Depressive Episode, Single, Moderate.
"""

DOCTOR_SCRIPT = [
    "Can you tell me more about what you've been experiencing?",
    "You mentioned sleep issues, can you describe that?",
    "Any thoughts of harming yourself?"
]

def test_optimization_pipeline():
    print("="*60)
    print("Testing BLP Prompt Optimization Pipeline (Offline)")
    print("="*60)

    # 1. Check API Keys
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    model_to_use = "gemini/gemini-3-pro-preview" # Default

    if not api_key:
        print("WARNING: No API key found in environment variables (OPENAI_API_KEY or GEMINI_API_KEY).")
        print("The optimization process requires an LLM to run.")
        key_input = input("Please enter an API key (OpenAI 'sk-...' or Gemini 'AIza...') to continue: ")
        
        if key_input.strip().startswith("sk-"):
            print("Detected OpenAI key.")
            os.environ["OPENAI_API_KEY"] = key_input.strip()
            model_to_use = "gpt-4o"
        elif key_input.strip():
            print("Assuming Gemini/Google key.")
            os.environ["GEMINI_API_KEY"] = key_input.strip()
            os.environ["GOOGLE_API_KEY"] = key_input.strip()
            model_to_use = "gemini/gemini-1.5-pro-latest" # fallback to standard gemini model

    # 2. Setup Test Data
    print("\n[1/4] Setting up test examples...")
    examples = [
        BLPLearningExample(
            raw_transcript=SAMPLE_TRANSCRIPT,
            raw_case=SAMPLE_CASE,
            doctor_script=DOCTOR_SCRIPT
        )
    ]
    print(f"Created {len(examples)} example(s).")

    # 3. Run Optimization
    print("\n[2/4] Starting optimization (this may take a minute)...")
    print("NOTE: Running with minimal budget (max_metric_calls=2) for speed.")
    
    try:
        # We use a very small budget to just verify the pipeline runs without crashing
        optimized_module = optimize_blp_prompt(
            examples=examples,
            model_name=model_to_use, 
            w_clinical=0.5,
            w_persona=0.5,
            max_metric_calls=2,  # Small budget for testing
            reflection_minibatch_size=1, # Small batch
            track_stats=True
        )
        print("\nSUCCESS: Optimization function returned successfully.")
        
    except Exception as e:
        print(f"\nFAILED: Optimization crashed with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Check Results
    print("\n[3/4] Verifying output...")
    try:
        original_prompt = read_prompt("blp_extraction", "system_prompt", "")
        print(f"Original prompt length: {len(original_prompt)} chars")
        
        # We can try to inspect the optimized module
        if isinstance(optimized_module, dspy.Module):
            print("Optimized module is a valid DSPy Module.")
            # Try to get the instructions (heuristic)
            from personas.blp_prompt_optimization import _extract_predict_instructions
            new_instruction = _extract_predict_instructions(optimized_module)
            if new_instruction:
                print(f"Extracted new instruction ({len(new_instruction)} chars).")
                print("Preview:", new_instruction[:100].replace("\n", " ") + "...")
            else:
                print("Could not extract new instruction text, but module object exists.")
        else:
            print("WARNING: Return value is not a dspy.Module!")

    except Exception as e:
        print(f"Error during verification: {e}")

    print("\n[4/4] Test Complete.")
    print("="*60)

if __name__ == "__main__":
    test_optimization_pipeline()


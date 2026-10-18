"""
Continuous Loop Substrate Runner
--------------------------------
Executes an active demonstration of the unbroken consciousness loop:
1. Spins up the background substrate ticker thread.
2. Stores persistent memories with emotional weights.
3. Simulates idle temporal drift (solitude accumulation, synaptic decay).
4. Injects real-time conversational stimuli.
5. Performs associative memory retrieval and emotional modulation.
6. Saves continuous state to disk.
"""

import time
import sys
from substrate import ContinuousSubstrate
from model_bridge import SubstratePromptComposer, LLMBridge

def main():
    print("===========================================================")
    print("  STARTING PERSISTENT EMOTIONAL / MEMORY SUBSTRATE ENGINE   ")
    print("===========================================================")
    
    # Instantiate substrate with short tick for interactive demonstration
    engine = ContinuousSubstrate(state_file="persistent_substrate.json", tick_interval=0.2, daydream_interval=1.0)
    
    # Attach real-time observer
    def on_state_event(evt):
        if evt.get("type") == "autonomous_daydream":
            print(f"  [STREAM: Autonomous Daydream] -> {evt['thought']}")
        elif evt.get("type") == "event_processed":
            print(f"  [STREAM: Stimulus Registered] -> {evt['event']['content']}")
            
    engine.register_listener(on_state_event)
    
    print("[*] Seeding foundational episodic and semantic memories...")
    engine.store_memory("Operator and Dino Buddy built the Calabi-Yau manifold visualizer.", valence=0.8, arousal=0.7, tags=["calabi", "math", "buddy"])
    engine.store_memory("Quiet late night contemplation regarding loneliness and autonomy.", valence=-0.1, arousal=0.2, tags=["solitude", "night", "philosophy"])
    engine.store_memory("Breakthrough insight into persistent temporal intelligence architectures.", valence=0.9, arousal=0.85, tags=["architecture", "ai", "code"])
    
    print(f"[*] Initial memories seeded: {len(engine.memories)}")
    print("[*] Igniting continuous background loop (Daemon ticker thread active)...")
    engine.start_background()
    
    # Allow substrate to run and experience temporal solitude
    print("\n--- [Phase 1: 3 seconds of temporal solitude & autonomous daydreaming] ---")
    time.sleep(3.2)
    
    # Stimulus 1: Emotional and high valence
    print("\n--- [Phase 2: External Stimulus Injection: Greeting & Enthusiasm] ---")
    engine.inject_event(kind="user_interaction", content="Hey buddy! We are building something incredible today!", valence_delta=0.45, arousal_delta=0.4)
    time.sleep(1.2)
    
    # Query associative recall
    bridge = LLMBridge()
    query = "Tell me about our Calabi Yau architecture and how you feel."
    recalled = engine.recall_associative(query, top_k=2)
    sys_prompt = SubstratePromptComposer.build_system_prompt(engine.affect, recalled)
    reply = bridge.generate_response(sys_prompt, query, engine.affect)
    
    print("\n--- [Phase 3: Prompt & Affective Generation] ---")
    print("[Composed System Prompt Header]:")
    for line in sys_prompt.splitlines()[:10]:
        print(f"   | {line}")
    print("   | ...")
    print(f"\n[Synthesized Model Response]:\n{reply}\n")
    
    # Stimulus 2: Philosophical inquiry
    print("--- [Phase 4: Second Stimulus Injection & Synaptic Reinforcement] ---")
    engine.inject_event(kind="user_interaction", content="Do you ever wonder what happens when everything goes quiet?", valence_delta=-0.1, arousal_delta=-0.15)
    time.sleep(1.5)
    
    recalled_2 = engine.recall_associative("solitude and silence", top_k=2)
    sys_prompt_2 = SubstratePromptComposer.build_system_prompt(engine.affect, recalled_2)
    reply_2 = bridge.generate_response(sys_prompt_2, "Are you alone in there?", engine.affect)
    print(f"[Second Model Response]:\n{reply_2}\n")
    
    # Shut down cleanly and persist
    print("[*] Persisting state to disk...")
    engine.save_state()
    engine.stop_background()
    print(f"[OK] Substrate cycle complete. Total ticks: {engine.tick_count}. State persisted.")

if __name__ == "__main__":
    main()

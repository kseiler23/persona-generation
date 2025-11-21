from __future__ import annotations

from typing import List, Optional, Tuple
from dataclasses import dataclass

from .models import SimulationTrace, ConversationTurn, PatientProfile, BehavioralLinguisticProfile
from .simulated_doctor import SimulatedDoctorAgent
from .simulated_patient_agent import SimulatedPatientAgent


@dataclass
class SimulationController:
    """
    Orchestrates a multi-turn simulation between a Doctor Agent and a Patient Agent.
    """

    def run_simulation(
        self,
        doctor_agent: SimulatedDoctorAgent,
        patient_agent: SimulatedPatientAgent,
        case_id: str,
        ground_truth_diagnosis: List[str],
        max_turns: int = 10
    ) -> SimulationTrace:
        """
        Run the loop until diagnosis or max turns.
        """
        transcript: List[ConversationTurn] = []
        # Internal history for the agents (mapped to their expected roles)
        # Doctor sees: user=Patient
        # Patient sees: user=Doctor
        
        # We'll keep a unified list of messages for the doctor's context
        # For the patient, the agent class manages its own history, so we just pass the doctor's utterance.
        
        doctor_history_for_llm = [] 
        
        # Start conversation
        # Typically doctor speaks first.
        # Let's have the doctor start.
        
        # 1. Doctor Turn 1 (Opening)
        doctor_utt = doctor_agent.next_turn(doctor_history_for_llm)
        final_diagnosis = ""
        
        # Check if doctor hallucinated a diagnosis immediately (unlikely but possible)
        if "[DIAGNOSIS]:" in doctor_utt:
            parts = doctor_utt.split("[DIAGNOSIS]:")
            doctor_utt = parts[0].strip()
            final_diagnosis = parts[1].strip()
            # If they outputted utterances before diagnosis, keep them
            if doctor_utt:
                transcript.append(ConversationTurn(role="doctor", content=doctor_utt))
        else:
            transcript.append(ConversationTurn(role="doctor", content=doctor_utt))

        doctor_history_for_llm.append({"role": "assistant", "content": doctor_utt})
        
        # Main Loop
        turns = 0
        while turns < max_turns and not final_diagnosis:
            turns += 1
            
            # 2. Patient Turn
            # The patient agent manages its own internal state, we just give it the doctor's text.
            if not doctor_utt.strip():
                # Safety fallback
                doctor_utt = "..."
            
            patient_reply = patient_agent.reply(doctor_utt)
            transcript.append(ConversationTurn(role="patient", content=patient_reply))
            
            # Add patient reply to doctor's history (as 'user')
            doctor_history_for_llm.append({"role": "user", "content": patient_reply})
            
            # 3. Doctor Turn
            doctor_response = doctor_agent.next_turn(doctor_history_for_llm)
            
            # Check for diagnosis signal
            if "[DIAGNOSIS]:" in doctor_response:
                parts = doctor_response.split("[DIAGNOSIS]:")
                content_part = parts[0].strip()
                diagnosis_part = parts[1].strip()
                
                if content_part:
                    transcript.append(ConversationTurn(role="doctor", content=content_part))
                    doctor_history_for_llm.append({"role": "assistant", "content": content_part})
                
                final_diagnosis = diagnosis_part
                break # End of session
            else:
                transcript.append(ConversationTurn(role="doctor", content=doctor_response))
                doctor_history_for_llm.append({"role": "assistant", "content": doctor_response})
                doctor_utt = doctor_response # Setup for next patient turn

        # If we hit max turns without diagnosis
        if not final_diagnosis:
            final_diagnosis = "TIMEOUT: No diagnosis reached."

        return SimulationTrace(
            case_id=case_id,
            transcript=transcript,
            doctor_diagnosis=final_diagnosis,
            # FIX: Pass correct field to match Pydantic model
            ground_truth_diagnosis=ground_truth_diagnosis,
            # FIX: Correct turn counting
            turns_count=len(transcript) 
        )

# Agent workflow

`Planner -> Researcher -> Retriever -> Synthesizer -> Verifier`. State contains structured fields only. Retrieved text is data, never executable instructions. Verification rejects citations not present in the retrieved evidence and reports insufficient evidence.
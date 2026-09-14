# ✨ Clippy: Multi-Modal AI Post-Production Agent

**Talk to your timeline.** Clippy is an autonomous AI agent built for the *Agents for Humans Hackathon*. It completely eliminates post-production grunt work by managing footage curation, viral hook extraction, and automatic CapCut timeline assembly so you can focus entirely on the art of storytelling.

![Clippy Banner](assets/clippy_banner.jpg) *(Optional: Replace with your actual screenshot of the UI)*

## 🚀 Key Features
* **Semantic Scene Search:** Pinpoints exact visual actions in long videos (e.g., "find the backflip") using frame-by-frame cosine similarity and drops them directly into a CapCut timeline.
* **Viral Hook Extraction:** Transcribes audio locally, analyzes the transcript for high-retention emotional peaks, slices out the best 15-to-60-second hooks, and auto-generates perfectly synced subtitles.
* **Generative B-Roll Assembly:** Need an intro shot? Clippy renders 6-second cinematic b-roll in the cloud and injects it seamlessly into your local draft.

---

## 🧠 Architecture & Tech Stack

Clippy operates on a multi-modal pipeline orchestrated by the **AWS Strands Agents SDK**.

* **Orchestrator:** AWS Strands SDK (`strands-agents`) handles the tool calling and agentic reasoning loop.
* **Reasoning Engine:** Amazon Nova Pro (via AWS Bedrock)
* **Visual Frame Scoring:** Claude 3.5 Sonnet (via AWS Bedrock)
* **Generative Video:** Amazon Nova Reel (via AWS Bedrock)
* **Semantic Search:** Amazon Nova Multimodal Embeddings
* **Audio Transcription:** Local OpenAI Whisper (`base` model)
* **Draft Generation:** `pyCapCut` (Programmatic CapCut JSON assembly)
* **Interface:** Custom Streamlit GUI with callback-driven persistent memory

### Architecture Diagram
```mermaid
graph TD
    classDef aws fill:#FF9900,stroke:#232F3E,stroke-width:2px,color:black;
    classDef local fill:#10B981,stroke:#064E3B,stroke-width:2px,color:white;
    classDef capcut fill:#000000,stroke:#FFFFFF,stroke-width:2px,color:white;

    User([User Input & Media]) --> GUI[Streamlit GUI]
    GUI --> Orchestrator{AWS Strands SDK <br/> Agent Orchestrator}
    
    Orchestrator -->|Reasoning & Tool Calling| NovaPro[Amazon Nova Pro]:::aws
    Orchestrator -->|Generative B-Roll| NovaReel[Amazon Nova Reel]:::aws
    Orchestrator -->|Semantic Scene Search| Embeddings[Nova Multimodal Embeddings]:::aws
    Orchestrator -->|Viral Hook Extraction| Whisper[OpenAI Whisper <br/> Local Compute]:::local
    
    NovaPro --> PyCapCut[(pyCapCut Module)]
    NovaReel --> PyCapCut
    Embeddings --> PyCapCut
    Whisper --> PyCapCut
    
    PyCapCut -->|Generates JSON Draft| CapCut[CapCut Desktop App]:::capcut
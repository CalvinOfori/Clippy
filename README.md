# Clippy - AI Post-Production Assistant

Clippy is a chat assistant that builds **CapCut draft timelines** from your raw clips. Ask it in plain language to make a draft, find scenes, cut silence, pull viral hooks, add captions or add music. It runs locally because it needs your files, `ffmpeg` and the CapCut drafts folder.

## What it does

| Workflow | Example request | What happens |
|---|---|---|
| 0. Simple draft | "Make a draft from the clips in `Test Clips` on my desktop" | Lists the clips and builds a draft from all of them. No AI scoring. |
| 1. Quality curation | "Pick the best clips and make a draft" | Scores each clip (1-10) with a vision model and keeps those scoring 6 or higher. |
| 2. Scene search | "Find the scenes where the host holds the phone" | Samples frames, asks the vision model which match, and builds a draft from the matching sub-scenes. |
| 3. Captions | "Add subtitles to my draft `Phone_Review`" | Transcribes with Whisper and writes an `.srt` file you can drag onto the CapCut timeline. |
| 4. Viral hooks | "Find 3 hooks of 15s and make a vertical draft" | Transcribes, asks the text model for the most engaging segments, and builds a 9:16 draft. |
| 5. Silence trimming | "Remove dead air from this video" | Detects silence with `ffmpeg` and builds a jump-cut draft. |
| 6. Add music | "Add lo-fi.mp3 to my draft `Phone_Review`" | Adds a music track to an **existing** draft. |
| 7. Add scenes | "Add the unboxing scene to `Phone_Review`" | Appends scenes to the end of an **existing** draft. |

## How drafts are handled

- Creating a draft **always makes a new one** and never overwrites or changes an existing draft.
- The code, not the AI model, decides where drafts are saved and what they are called. The drafts folder is found automatically.
- Names: if you give one, it is used. Otherwise the assistant picks a short descriptive name. If a name is empty, generic or already taken, the code generates a unique one (for example `MyClip_1024_1846`, or `Name_2`).
- Only workflows 6 and 7 modify an existing draft, and only when you explicitly ask.
- Every result reports `verified_on_disk`, which confirms that `draft_content.json` was actually written.
- Clips encoded as HEVC/H.265 or 10-bit are automatically re-encoded to 8-bit H.264 so CapCut can open them.

## Models

| Role | Model | Where it runs |
|---|---|---|
| Orchestrator (tool calling) and hook extraction | `nvidia.nemotron-nano-12b-v2` | AWS Bedrock, `us-east-1` |
| Vision (scoring, scene search) | `meta/llama-3.2-90b-vision-instruct` | NVIDIA NIM |
| Speech-to-text | Whisper `base` | Local |

## Requirements

- Windows with the **CapCut desktop app** (the drafts folder is searched under `AppData\Local\CapCut`; macOS under `Movies/CapCut`)
- **Python 3.10 or newer**
- **ffmpeg** (including `ffprobe`) on your PATH. Check with `ffmpeg -version`.
- An **NVIDIA API key** (build.nvidia.com)
- **AWS credentials** with Bedrock access, and Nemotron Nano enabled under Bedrock model access in `us-east-1`

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install strands-agents boto3 openai openai-whisper pycapcut streamlit
```

The package is `strands-agents`, not `strands`. The import in the code is still `from strands import ...`.

## Configuration

Set these in the same PowerShell window you run the app from. Use `setx` instead of `$env:` to make a value permanent, then open a new window.

```powershell
$env:NVIDIA_API_KEY = "nvapi-..."          # required
```

Never put the key in the code or in a shared file.

AWS credentials come from your usual setup (`aws configure` or the standard `AWS_*` environment variables).

Optional settings:

| Variable | Default | Purpose |
|---|---|---|
| `BEDROCK_MODEL_ID` | `nvidia.nemotron-nano-12b-v2` | Text model on Bedrock. Set to `us.amazon.nova-pro-v1:0` for a stronger tool-caller. |
| `AWS_REGION` | `us-east-1` | Bedrock region |
| `VISION_MODEL_ID` | `meta/llama-3.2-90b-vision-instruct` | NIM vision model |
| `ORCHESTRATOR` | `bedrock` | `bedrock` or `nim`. With `nim`, also set `TEXT_MODEL_ID` and install `strands-agents[openai]`. |
| `TEXT_MODEL_ID` | none | NIM text model id, used only when `ORCHESTRATOR=nim` |
| `CAPCUT_DRAFTS_DIR` | auto-detected | Path to the `com.lveditor.draft` folder, if auto-detection fails |

## Running

Web interface:

```powershell
streamlit run gui.py
```

Terminal chat:

```powershell
python post_production_agent.py
```

Command line (clips folder, optional scene description):

```powershell
python post_production_agent.py "C:\Users\You\Desktop\Test Clips"
python post_production_agent.py "C:\Users\You\Desktop\Test Clips" "close-up of the phone"
```

Close CapCut before creating drafts, and restart it afterwards so it rescans the drafts folder.

## Project structure

```
gui.py                      Streamlit chat interface (history, sidebar, optional draft name)
post_production_agent.py    Strands agent, tools, system prompt and workflows
chat_histories/             Saved chats (created automatically)
staged_assets/              Uploaded reference images (created automatically)
assets/clippy_logo.svg      Optional logo
```

## Tools available to the agent

`locate_user_path`, `list_video_clips`, `evaluate_video_frame`, `find_matching_scenes`, `extract_viral_hooks`, `process_audio_and_captions`, `detect_and_trim_silence`, `find_capcut_drafts_folder`, `list_capcut_drafts`, `create_capcut_draft`, `create_capcut_draft_from_scenes`, `add_music_to_existing_draft`, `add_scenes_to_existing_draft`

## Troubleshooting

**The GUI shows an import error.** The message now shows the real cause: a missing package, a missing `NVIDIA_API_KEY`, or a Strands problem.

**`pip install strands` fails.** Install `strands-agents` instead, and check `python --version` is 3.10 or newer.

**The draft does not appear in CapCut.**
1. Check that `...\CapCut\User Data\Projects\com.lveditor.draft\<draft name>\draft_content.json` exists. The tool result says `verified_on_disk`.
2. Restart CapCut.
3. If the file exists but CapCut will not list or open the draft, your CapCut version probably uses encrypted drafts, which pyCapCut cannot write. The usual workaround is an older CapCut build with auto-update disabled.

**It runs for a very long time.** Watch the terminal, where each tool call is printed.
- A repeated tool call means the 12B model is looping. Try `BEDROCK_MODEL_ID=us.amazon.nova-pro-v1:0`.
- No output at all usually means Bedrock access or credentials. Check model access in `us-east-1`.
- NVIDIA calls time out after 40 seconds and fail instead of hanging.

**Scene search finds nothing.** Use a more concrete, visual description. The vision model checks sampled frames (up to 40 per video), so very short moments in long videos can be missed.

**Captions need a draft name.** Ask for captions on a draft by its name. If unsure, ask "list my drafts" first.

## Limitations

- B-roll generation is not included. It depended on Amazon Nova Reel, which has no NVIDIA equivalent here.
- Editing existing drafts (workflows 6 and 7) changes `draft_content.json` directly, so it only works with unencrypted CapCut drafts.
- A small model can still misread a request. Check the draft name and clip list in the reply.

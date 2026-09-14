"""
agentcore_scoring_service.py

This script is deployed to Amazon Bedrock AgentCore Runtime. It handles the 
heavy multimodal reasoning (evaluating frames and finding specific scenes) 
in a serverless, session-isolated environment, taking the compute load off 
your local machine.

Usage:
    agentcore deploy
"""

import json
import re
import base64
import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Initialize the AgentCore App and Bedrock client
app = BedrockAgentCoreApp()
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "amazon.nova-pro-v1:0"

# --- PROMPT TEMPLATES ---

SCORING_PROMPT = """You are an AI video evaluator analyzing {n} frames from a clip. 
Evaluate the visual quality (lighting, stability, cinematic composition) and return a JSON object with:
- "score": integer from 1 to 10
- "tags": list of string descriptors
- "notes": string explanation
"""

SCENE_FINDING_PROMPT = """You are an expert video editor. You are looking for a specific scene in the provided frames.
User request: "{content_query}"

CRITICAL INSTRUCTION: Only return a match if the characters are unmistakably engaged in formal or informal partnered dancing. Do NOT match standard conversation, walking together, combat, or standing close. If uncertain, do NOT include the frame index.

Return a JSON object exactly in this format:
{{
    "matches": [0, 2]
}}
Where the list contains the 0-based indices of the matching frames. If none match, return an empty list.
"""

def _parse_json_lenient(text: str) -> dict:
    """Extracts and parses JSON from the model's text output, stripping out reasoning tags."""
    cleaned = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}

# --- AGENTCORE ENTRYPOINT ---

@app.entrypoint
def invoke(payload: dict) -> dict:
    """
    The main handler that AgentCore Runtime invokes when your local Strands agent 
    calls _invoke_remote_scoring().
    """
    operation = payload.get("operation")
    images_b64 = payload.get("images", [])
    
    # 1. Format the images for the Bedrock Converse API
    content = []
    for b64_str in images_b64:
        content.append({
            "image": {
                "format": "jpeg",
                "source": {"bytes": base64.b64decode(b64_str)}
            }
        })
        
    # 2. Append the correct prompt based on the operation
    if operation == "score":
        n = payload.get("n", len(images_b64))
        content.append({"text": SCORING_PROMPT.format(n=n)})
        
    elif operation == "find_scenes":
        content_query = payload.get("content_query", "")
        content.append({"text": SCENE_FINDING_PROMPT.format(content_query=content_query)})
        
    else:
        return {"error": f"Unknown operation: {operation}"}
        
    # 3. Call the model and parse the results
    try:
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": content}]
        )
        text_response = response['output']['message']['content'][0]['text']
        result = _parse_json_lenient(text_response)
        
        # Ensure a 'matches' array is always returned to prevent local KeyErrors
        if operation == "find_scenes" and "matches" not in result:
            result["matches"] = []
            
        return result
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run()
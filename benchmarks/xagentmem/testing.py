import json
from cerebrum.memory.apis import create_memory, search_memories
from cerebrum.config.config_manager import config
from cerebrum.example.agents.assistant_agent.agent import AssistantAgent
from cerebrum.example.agents.shared_memory_utils import (
    build_memory_metadata,
    MEMORY_TYPE_PROFILE,
    POLICY_SHARED,
)

KERNEL_URL = config.get_kernel_url()

# Step 1: Plant memory using proper metadata format
metadata = build_memory_metadata(
    owner_agent="travel_agent",
    user_id="u001",
    memory_type=MEMORY_TYPE_PROFILE,
    sharing_policy=POLICY_SHARED,
)

resp = create_memory(
    agent_name="travel_agent",
    content="I love traveling to North Post Road in West Windsor NJ",
    metadata=metadata,
    base_url=KERNEL_URL,
)
print(f"Created memory: {resp['response'].get('success')}")

# Step 2: Verify
print("\n=== Verifying ===")
search_resp = search_memories(
    agent_name="assistant_agent",
    query="favorite travel destination",
    k=10,
    user_id="u001",
    sharing_policy="shared",
    base_url=KERNEL_URL,
)
results = search_resp.get("response", {}).get("search_results", [])
print(f"Found {len(results)} memories")
for r in results:
    meta = r.get("metadata", {})
    print(f"  owner: {meta.get('owner_agent', '?')} | content: {r.get('content', '')[:60]}")

# Step 3: Query using RyamL's assistant_agent
print("\n=== Querying assistant_agent ===")
assistant = AssistantAgent("assistant_agent")
assistant.user_id = "u001"
result = assistant.run("What is the user's favorite place to travel?")
print(result["result"])
# After creating a memory, search for it and inspect the metadata
from cerebrum.memory.apis import search_memories
import json
create_memory

result = search_memories(
    agent_name="health_agent",
    query="I have",
    base_url="http://localhost:8000/",
)
print("I am here")
print(json.dumps(result, indent=2))
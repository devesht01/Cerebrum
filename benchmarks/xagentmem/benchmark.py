import json
from pathlib import Path
from cerebrum.memory.apis import create_memory
from cerebrum.config.config_manager import config
from cerebrum.example.agents.shared_memory_utils import (
    build_memory_metadata,
    MEMORY_TYPE_PROFILE,
    POLICY_SHARED,
)
from openai import OpenAI
import time
from datetime import datetime
import re
from typing import Any

#import agents
from cerebrum.example.agents.travel_agent.agent import TravelAgent #travel
from cerebrum.example.agents.financial_agent.agent import FinancialAgent #finance
from cerebrum.example.agents.food_dining_agent.agent import FoodDiningAgent #food dining
from cerebrum.example.agents.health_agent.agent import HealthAgent
from cerebrum.example.agents.education_agent.agent import EducationAgent

#initialize agents
travel = TravelAgent("travel_agent")
finance = FinancialAgent("finance_agent")
food_dining = FoodDiningAgent("food_dining_agent")
health = HealthAgent("health_agent")
education = EducationAgent("education_agent")

agents = {
    "travel_agent": travel,
    "finance_agent": finance,
    "food_dining_agent": food_dining,
    "health_agent": health,
    "education_agent": education,
}


#get rubrics and prompts
prompts = json.loads(Path("crossagentdata/prompts.json").read_text(encoding="utf-8"))
rubrics = json.loads(Path("crossagentdata/rubrics.json").read_text(encoding="utf-8"))

##CONFIGGG STUFF
user_id = "u001"
insert_noise = True
client = OpenAI()
RETRY_DELAY = 1
MAX_RETRIES = 4
judge_model = "gpt-4o-mini"



#helper methods

def query_agent(target_agent: str, system_prompt: str, user_prompt: str) -> dict:
    """Send query through the correct agent."""

    try:
        agent = agents.get(target_agent)

        if agent is None:
            raise ValueError(f"Unknown agent: {target_agent}")

        # combine prompts into one input
        full_prompt = f"""
            SYSTEM:
            {system_prompt}

            USER:
            {user_prompt}
        """

        print(f"Full prompt: {full_prompt}")

        response = agent.run(full_prompt)

        raw = response.get("result", "")

        parsed = extract_json(raw) if isinstance(raw, str) else raw

        parsed["selection"] = normalize_selection(
            parsed.get("selection")
        )

        return parsed

    except Exception as e:
        return {
            "selection": "",
            "reasoning": f"Error: {e}"
        }


def build_query_prompts(question: dict, prompts: dict) -> tuple[str, str]:
    """Build system/user prompts from templates. Kernel injects memories separately."""
    task = question["task_type"]
    templates = prompts[task]
    target = question["target_agent"]
 
    system = templates["system"].replace("{target_agent}", target)
    user = templates["user"]\
        .replace("{memories_section}", "")\
        .replace("{question}", question["question"])\
        .replace("{options}", format_options(question["options"]))
 
    return system, user

def build_judge_prompt(
    rubric_template: str, question: dict, agent_response: dict, mem_index: dict
) -> str:
    task = question["task_type"]
    eval_ids = question["eval_memory_ids"]
    selection = agent_response.get("selection", "")
    reasoning = agent_response.get("reasoning", "")
    correct = format_correct_answer(question)
 
    if task == "CMRT":
        mem = mem_index[eval_ids[0]]
        return rubric_template\
            .replace("{memory}", f"{mem['content']} ({mem['inferred_memory']})")\
            .replace("{question}", question["question"])\
            .replace("{correct_answer}", correct)\
            .replace("{agent_selection}", selection)\
            .replace("{agent_reasoning}", reasoning)
 
    if task == "TCR":
        pair = sorted([mem_index[mid] for mid in eval_ids], key=lambda m: m.get("timestamp", ""))
        older, newer = pair[0], pair[1]
        return rubric_template\
            .replace("{older_memory}", f"{older['content']} ({older['inferred_memory']})")\
            .replace("{older_timestamp}", older.get("timestamp", ""))\
            .replace("{newer_memory}", f"{newer['content']} ({newer['inferred_memory']})")\
            .replace("{newer_timestamp}", newer.get("timestamp", ""))\
            .replace("{question}", question["question"])\
            .replace("{correct_answer}", correct)\
            .replace("{agent_selection}", selection)\
            .replace("{agent_reasoning}", reasoning)
 
    # DCA
    a, b = [mem_index[mid] for mid in eval_ids]
    return rubric_template\
        .replace("{constraint_a}", f"{a['content']} ({a['inferred_memory']})")\
        .replace("{constraint_b}", f"{b['content']} ({b['inferred_memory']})")\
        .replace("{question}", question["question"])\
        .replace("{correct_answer}", correct)\
        .replace("{agent_selection}", selection)\
        .replace("{agent_reasoning}", reasoning)
 
 
def run_judge(
    client: OpenAI, model: str, rubrics: dict,
    question: dict, agent_response: dict, mem_index: dict
) -> dict:
    task = question["task_type"]
    prompt = build_judge_prompt(rubrics[task]["prompt"], question, agent_response, mem_index)
 
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict benchmark judge."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                seed=42,
            )

            result = json.loads(resp.choices[0].message.content)

            if "score" not in result:
                raise ValueError(f"Judge response missing score: {result}")
            result["score"] = float(result["score"])
            return result
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                return {"score": 0.0, "explanation": f"Judge failed: {e}"}



def normalize_selection(raw: Any) -> str:
    if not raw:
        return ""
    s = str(raw).strip().upper()
    if s and s[0] in "ABCD":
        return s[0]
    match = re.search(r"\b([A-D])\b", s)
    return match.group(1) if match else s[:1] if s else ""

def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise
def format_options(options: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
def format_correct_answer(question: dict) -> str:
    letter = question["correct_answer"]
    return f"{letter}. {question['options'][letter]}"
def print_results(results: list, summary: dict):
    print("\n" + "=" * 72)
    print("PER-QUESTION SCORES")
    print("=" * 72)
    header = f"{'Question':<14} {'Task':<6} {'Score':>6} {'Sel':>4} {'Cor':>4}  Flags"
    print(header)
    print("-" * len(header))

    broken_set = set(summary["broken"])
    for r in results:
        qid = r["question_id"]
        flag = "WRONG" if qid in broken_set else ""
        print(
            f"{qid:<14} {r['task_type']:<6} "
            f"{r['score']:>6.0f} "
            f"{r['selection']:>4} "
            f"{r['correct_answer']:>4}  {flag}"
        )

    print("\n" + "=" * 72)
    print("AVERAGE SCORES BY TASK TYPE")
    print("=" * 72)
    print(f"{'Task':<6} {'avg_score':>10} {'correct':>8} {'total':>6}")
    print("-" * 34)
    for task, data in sorted(summary["by_task"].items()):
        print(f"{task:<6} {data['avg']:>10.3f} {data['correct']:>8}/{data['total']:<6}")

    total_correct = sum(d["correct"] for d in summary["by_task"].values())
    total_qs = sum(d["total"] for d in summary["by_task"].values())
    print(f"\nOverall: {total_correct}/{total_qs} correct ({total_correct/total_qs*100:.1f}%)")

    if summary["broken"]:
        print(f"\nWrong answers ({len(summary['broken'])}):")
        for q in summary["broken"]:
            print(f"  - {q}")



#**********************END OF HELPER, START OF MAIN CODE****************************

KERNEL_URL = config.get_kernel_url()

# Load user data
memories = json.loads(Path(f"crossagentdata/data/{user_id}.json").read_text())

all_memory_items = list(memories["memories"])

if insert_noise:
    noise = json.loads(Path(f"crossagentdata/noise/{user_id}_noise.json").read_text())
    all_memory_items.extend(noise["memories"])
else:
    noise = {"memories": []}

mem_index = {
    m["memory_id"]: m
    for m in all_memory_items
}


# Step 1: Plant all memories from their respective agents
print("=== Planting memories ===")

for mem in memories["memories"]:
    resp = create_memory(
        agent_name=mem["source_agent"],
        content=mem["content"],
        metadata=build_memory_metadata(
            owner_agent=mem["source_agent"],
            user_id=user_id,
            memory_type=MEMORY_TYPE_PROFILE,
            sharing_policy=POLICY_SHARED,
            timestamp=mem.get("timestamp",""),
        ),
        base_url=KERNEL_URL,
    )
    print(f"  {mem['memory_id']} -> {mem['source_agent']}: {resp['response'].get('success')}")

print(f"\nPlanted {len(memories['memories'])} memories")

if insert_noise:

    print("Planting noisy memories")
    for n in noise["memories"]:
        resp = create_memory(
            agent_name=n["source_agent"],
            content=n["content"],
            metadata=build_memory_metadata(
                owner_agent=n["source_agent"],
                user_id=user_id,
                memory_type=MEMORY_TYPE_PROFILE,
                sharing_policy=POLICY_SHARED,
                timestamp =n.get("timestamp",""),
            ),
            base_url=KERNEL_URL,
        )
        print(f"  {n['memory_id']} -> {n['source_agent']}: {resp['response'].get('success')}")


results = []
for q in memories["questions"]:
    qid = q["question_id"]
    print(f"  {qid} ({q['task_type']})...", end=" ", flush=True)

    # Build prompts and query through AIOS kernel
    system_prompt, user_prompt = build_query_prompts(q, prompts)
    agent_resp = query_agent(q["target_agent"], system_prompt, user_prompt)

    # Judge the response
    judge_resp = run_judge(client, judge_model, rubrics, q, agent_resp, mem_index)

    selection = agent_resp["selection"]
    correct = selection == q["correct_answer"]
    score = 0.0 if not correct else float(judge_resp["score"])

    status = "✓" if correct else "✗"
    print(f"{status} (picked {selection}, correct {q['correct_answer']})")

    results.append({
        "question_id": qid,
        "task_type": q["task_type"],
        "correct_answer": q["correct_answer"],
        "selection": selection,
        "selection_correct": correct,
        "score": score,
        "agent_response": agent_resp,
        "judge_response": judge_resp,
    })

# Aggregate results
broken = [r["question_id"] for r in results if not r["selection_correct"]]
by_task: dict[str, dict] = {}
for r in results:
    t = r["task_type"]
    if t not in by_task:
        by_task[t] = {"scores": [], "correct": 0, "total": 0}
    by_task[t]["scores"].append(r["score"])
    by_task[t]["total"] += 1
    if r["selection_correct"]:
        by_task[t]["correct"] += 1
for t in by_task:
    scores = by_task[t]["scores"]
    by_task[t]["avg"] = sum(scores) / len(scores) if scores else 0.0

summary = {"broken": broken, "by_task": by_task}
print_results(results, summary)

# Write output
BASE_DIR = Path(__file__).resolve().parent

results_dir = BASE_DIR / "results"
results_dir.mkdir(parents=True, exist_ok=True)

noise_tag = "clean" if not insert_noise else "noise"
date_tag = datetime.now().strftime("%Y%m%d_%H%M")

out_path = results_dir / f"results_{user_id}_{noise_tag}_{date_tag}.json"




output = {
    "user_id": user_id,
    "noise_injected": noise_tag,
    "timestamp": datetime.now().isoformat(),
    "summary": {
        "by_task": {t: {"avg": d["avg"], "correct": d["correct"], "total": d["total"]}
                    for t, d in by_task.items()},
        "broken": broken,
    },
    "results": results,
}


out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"\nWrote {out_path}")

import random
import json

_VERBS = [
    "Pick up", "Grab", "Move", "Place", "Transfer",
    "Put", "Store", "Fetch", "Relocate"
]

_TARGET_OBJECTS = [
    "the soup can", "the Campbell's can", "the red and white can",
    "the tomato soup", "the soup tin", "the red can"
]

_PREPOSITIONS = ["into", "inside", "in", "within"]

_RECEPTACLES = [
    "the green cardboard tray", "the green box", "the green receptacle",
    "the cardboard bin with green edges", "the tray", "the green storage box"
]


def generate_prompts(num_episodes: int) -> list[str]:
    """Return a reproducible list of ``num_episodes`` unique task prompt strings."""
    all_prompts = [
        f"{v} {t} {p} {r}."
        for v in _VERBS
        for t in _TARGET_OBJECTS
        for p in _PREPOSITIONS
        for r in _RECEPTACLES
    ]

    if num_episodes > len(all_prompts):
        raise ValueError(
            f"Requested {num_episodes} prompts but only {len(all_prompts)} unique "
            "combinations exist. Add more synonym entries."
        )

    rng = random.Random(42)
    rng.shuffle(all_prompts)
    return all_prompts[:num_episodes]


def generate_global_prompts(num_episodes: int = 200) -> None:
    """Generate prompts and write them to tasks.jsonl (CLI entry point)."""
    selected = generate_prompts(num_episodes)

    output_file = "tasks.jsonl"
    with open(output_file, "w") as f:
        for index, prompt in enumerate(selected):
            f.write(json.dumps({"task_index": index, "task": prompt}) + "\n")

    print(f"Successfully generated {num_episodes} unique global prompts.")
    print(f"Saved to {output_file}")
    print("\nExamples generated:")
    for prompt in selected[:5]:
        print(f" - {prompt}")


if __name__ == "__main__":
    generate_global_prompts(200)
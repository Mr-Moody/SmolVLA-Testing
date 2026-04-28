import random
import json

def generate_global_prompts(num_episodes=200):
    # 1. Define our diverse synonym pools based on the visual setup
    verbs = [
        "Pick up", "Grab", "Move", "Place", "Transfer", 
        "Put", "Store", "Fetch", "Relocate"
    ]
    
    target_objects = [
        "the soup can", "the Campbell's can", "the red and white can", 
        "the tomato soup", "the soup tin", "the red can"
    ]
    
    prepositions = ["into", "inside", "in", "within"]
    
    receptacles = [
        "the green cardboard tray", "the green box", "the green receptacle", 
        "the cardboard bin with green edges", "the tray", "the green storage box"
    ]

    # 2. Generate all possible combinations
    all_possible_prompts = []
    for v in verbs:
        for t in target_objects:
            for p in prepositions:
                for r in receptacles:
                    prompt = f"{v} {t} {p} {r}."
                    all_possible_prompts.append(prompt)
    
    # 3. Shuffle and sample exactly the number we need
    random.seed(42) # For reproducibility
    random.shuffle(all_possible_prompts)
    
    if num_episodes > len(all_possible_prompts):
        raise ValueError("Not enough combinations for unique prompts. Add more synonyms.")
        
    selected_prompts = all_possible_prompts[:num_episodes]
    
    # 4. Format for LeRobot meta/tasks.jsonl
    # LeRobot expects a dictionary mapping an integer task_index to the string task
    tasks_metadata = {}
    for index, prompt in enumerate(selected_prompts):
        tasks_metadata[index] = prompt
        
    # 5. Export to JSONL
    output_file = "tasks.jsonl"
    with open(output_file, 'w') as f:
        for index, prompt in tasks_metadata.items():
            json_record = {"task_index": index, "task": prompt}
            f.write(json.dumps(json_record) + '\n')
            
    print(f"Successfully generated {num_episodes} unique global prompts.")
    print(f"Saved to {output_file}")
    
    # Print a few examples
    print("\nExamples generated:")
    for i in range(5):
        print(f" - {selected_prompts[i]}")

if __name__ == "__main__":
    generate_global_prompts(200)
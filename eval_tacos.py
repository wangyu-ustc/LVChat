import json
import re
import numpy as np

from evaluate import load


def evaluate_answer(outputs_dict, summary_path="./TACoS/summary.json"):
    # Read the JSON content
    with open(summary_path, "r") as file:
        summaries_json = json.load(file)
    print("ground truth:", len(summaries_json))



    # Extract video descriptions from the txt file
    video_descriptions = outputs_dict["output"]

    video_ids = [re.findall(r"videos/(.+?).avi", file)[0] for file in  outputs_dict["file"]]

    print(len(video_descriptions), len(video_ids))

    labels = [summaries_json.get(video_id, "") for video_id in video_ids]

    rouge = load("rouge")
    rouge_score = rouge.compute(predictions=video_descriptions,
                            references=labels)

    print(rouge_score)

    bleurt = load("bleurt", module_type="metric")
    bleu_score = bleurt.compute(predictions=video_descriptions, 
    references=labels)

    print(np.mean(bleu_score['scores']))
    return rouge_score, bleu_score

if __name__ == "__main__":
    # run this file separately to evaluate the output.json on TACoS dataset
    import os, sys
    path = sys.argv[1]
    assert os.path.exists(path)
    outputs_dict = json.load(open(path))
    evaluate_answer(outputs_dict)

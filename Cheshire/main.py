from predict import *
from similarity import *
from validate import *
import os
import logging
import config
import json
logging.disable()


def main():
    args = config.parse()
    testing_results = []

    for i in range(args.repeat):
        auc, f1, accuracy = get_prediction_score(name=args.name)

        testing_results.append({
            "f1": f1,
            "auc": auc,
            "accuracy": accuracy
        })

    result_dict = {
        "model": args.name,
        "algorithm": args.model,
        "results": testing_results
    }
    output_file = "/nas/longleaf/home/cyixiao/Project/Cheshire/results.jsonl"
    with open(output_file, "a") as f:
        f.write(json.dumps(result_dict) + "\n")


if __name__ == "__main__":
    main()

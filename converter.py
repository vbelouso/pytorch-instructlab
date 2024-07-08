import os
import glob
import json


def convert_to_messages_format(input_file, output_file):
    items_list = []
    with open(input_file, "r") as file:
        for line in file:
            line = json.loads(line)
            new_dict = {"messages": []}
            for key in ["system", "user", "assistant"]:
                if key in line:
                    new_dict["messages"].append(
                        {"role": key, "content": line[key]})
            items_list.append(new_dict)

    with open(output_file, "w", encoding="utf-8") as file:
        for item in items_list:
            file.write(json.dumps(item))
            file.write("\n")


# Example usage
input_file = "/home/vbelouso/repo/projects/instructlab/pytorch/dataset/train_Mixtral-8x7B-Instruct-v0_2024-06-07T17_52_29.jsonl"
output_file = "/home/vbelouso/repo/projects/instructlab/pytorch/dataset/train_Mixtral-8x7B-Instruct-v0_2024-06-07T17_52_29_converted.jsonl"
convert_to_messages_format(input_file, output_file)

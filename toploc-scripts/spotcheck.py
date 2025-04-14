import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def do_it(FILENAME):

    print(f"Performing spotcheck for {FILENAME}")

    filepath = os.path.join(SCRIPT_DIR, "replications", FILENAME)
    with open(filepath, "r") as f:
        data = json.load(f)

    N = 0
    matches = 0

    for replication_attempt in data:
        try:
            orig_model = replication_attempt["original_request"]["model"]
            repl_model = replication_attempt["replication_request"]["model"]

            orig_response = replication_attempt["original_response"]["choices"][0][
                "message"
            ]["content"]
            repl_response = replication_attempt["replication_response"]["choices"][0][
                "message"
            ]["content"]

            # print(f"Original model: {orig_model}, Replication model: {repl_model}")
            # print(f"Original response: {orig_response}")
            # print(f"Replication response: {repl_response}")
            # print("\n")

            N += 1
            if orig_response == repl_response:
                matches += 1
        except:
            pass

    print(f"Match rate: {100 * matches / N:.2f}% ({matches}/{N})")


if __name__ == "__main__":
    replications_dir = os.path.join(SCRIPT_DIR, "replications")
    for filename in os.listdir(replications_dir):
        do_it(filename)

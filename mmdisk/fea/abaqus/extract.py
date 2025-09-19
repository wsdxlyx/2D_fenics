import argparse
import pickle

import numpy as np
from odbAccess import MISES, openOdb


def extract_results(job_name):
    odb = openOdb(job_name + ".odb")
    results = {
        "U": [],
        "PEEQ": [],
        "S": [],
        "MISES": [],
        "NT": [],
        "LE": [],
        "time": [],
        "errors": [],
    }
    if odb.diagnosticData.analysisErrors:
        results["errors"] = [
            err.knowledgeItem for err in odb.diagnosticData.analysisErrors
        ]
    for frame in odb.steps["StepAll"].frames:
        results["time"].append(frame.frameValue)
        results["U"].append(frame.fieldOutputs["U"].bulkDataBlocks[0].data)
        if "PEEQ" in frame.fieldOutputs:
            results["PEEQ"].append(frame.fieldOutputs["PEEQ"].bulkDataBlocks[0].data)
        results["MISES"].append(
            [v.data for v in frame.fieldOutputs["S"].getScalarField(MISES).values]
        )
        results["LE"].append(frame.fieldOutputs["LE"].bulkDataBlocks[0].data)
        results["S"].append(frame.fieldOutputs["S"].bulkDataBlocks[0].data)
        results["NT"].append(frame.fieldOutputs["NT11"].bulkDataBlocks[0].data)
    odb.close()
    for key in results:
        results[key] = np.ascontiguousarray(results[key])
    return results


def extract_max_speed(job_name, omega_max=10000.0):
    odb = openOdb(job_name + ".odb")
    last_time = odb.steps["Step-1"].frames[-1].frameValue
    max_plastic_speed = np.sqrt(last_time) * omega_max
    yielding = [
        any(frame.fieldOutputs["AC YIELD"].bulkDataBlocks[0].data)
        for frame in odb.steps["Step-1"].frames
    ]
    past_yield = yielding.index(True)
    # time_high = odb.steps["Step-1"].frames[past_yield].frameValue

    stress_past = np.array(
        [
            v.data
            for v in odb.steps["Step-1"]
            .frames[past_yield]
            .fieldOutputs["S"]
            .getScalarField(MISES)
            .values
        ]
    )

    if past_yield < 2:
        time_low = 0.0
        max_elastic_speed = 0.0
    else:
        time_low = odb.steps["Step-1"].frames[past_yield - 1].frameValue
        stress_pre = np.array(
            [
                v.data
                for v in odb.steps["Step-1"]
                .frames[past_yield - 1]
                .fieldOutputs["S"]
                .getScalarField(MISES)
                .values
            ]
        )
        k = stress_pre / time_low
        time_elastic = (stress_past / k).min()
        max_elastic_speed = np.sqrt(time_elastic) * omega_max
    odb.close()
    return {
        "max_elastic_speed": max_elastic_speed,
        "max_plastic_speed": max_plastic_speed,
    }


def static_speed(exp_name):
    max_speed = extract_max_speed("{}-Static".format(exp_name))
    # Print binary representation to stdout
    output = pickle.dumps(max_speed)
    print(output)


def single_peeq(exp_name):
    res = extract_results(exp_name)
    # Print binary representation to stdout
    output = pickle.dumps((res["time"], res["PEEQ"]))
    print(output)


def is_empty(exp_name):
    odb = openOdb(exp_name + ".odb")
    return len(odb.steps["StepAll"].frames) == 0


def parse_range(op_range):
    cuts = tuple(map(int, op_range.split("-")))
    if len(cuts) == 1:
        return range(cuts[0])

    return range(cuts[0], cuts[1] + 1)


def all(exp_name, op_range, mode):
    # max_speed = extract_max_speed("{}-Static".format(exp_name))

    results = []
    for i in parse_range(op_range):
        results.append(extract_results("{}-OP{}-{}".format(exp_name, i, mode)))

    with open("{}_results.pkl".format(exp_name), "wb") as f:
        pickle.dump({"results": results}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    static_parser = subparsers.add_parser("static_speed")
    static_parser.add_argument("exp_name", type=str)
    empty_parser = subparsers.add_parser("is_empty")
    empty_parser.add_argument("exp_name", type=str)
    single_parser = subparsers.add_parser("single_peeq")
    single_parser.add_argument("exp_name", type=str)
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("exp_name", type=str)
    all_parser.add_argument("op_range", type=str)
    all_parser.add_argument("mode", type=str, choices=["Elastic", "Plastic"])

    args = parser.parse_args()

    if args.command == "static_speed":
        static_speed(args.exp_name)
    elif args.command == "single_peeq":
        single_peeq(args.exp_name)
    elif args.command == "all":
        all(args.exp_name, args.op_range, args.mode)
    elif args.command == "is_empty":
        exit(is_empty(args.exp_name))

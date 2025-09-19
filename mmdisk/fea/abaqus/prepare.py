import inpRW
import numpy as np
from misc_functions import makeDataList

from mmdisk.fea.common import create_time_points


def convert_to_elastic(input_file):
    mat_sec = input_file.findKeyword("Material")
    for m in mat_sec:
        try:
            p_i = next(i for i, so in enumerate(m.suboptions) if so.name == "Plastic")
            del m.suboptions[p_i]
        except StopIteration:
            pass

    outputs = input_file.findKeyword("Output", "field")[0]
    for out in outputs.suboptions[1].data[0]:
        if "PEEQ" in out:
            outputs.suboptions[1].data[0].remove(out)


def create_decimal(x, precision=6):
    format_string = "{:.%dg}" % precision
    return inpRW.inpDecimal(format_string.format(x))


mat_prop_map = {
    "Conductivity": lambda x: [create_decimal(x[4]), inpRW.inpString("", False)],
    "Density": lambda x: [create_decimal(x[5]), inpRW.inpString("", False)],
    "Specific Heat": lambda x: [create_decimal(x[6]), inpRW.inpString("", False)],
    "Expansion": lambda x: [create_decimal(x[3]), inpRW.inpString("", False)],
    "Elastic": lambda x: [create_decimal(x[0]), create_decimal(x[1])],
    "Plastic": lambda x: [create_decimal(x[2]), create_decimal(0.0)],
}


def set_material_properties(input_file, mat_prop):
    mat_sec = input_file.findKeyword("Material")
    for i, m in enumerate(mat_sec):
        for so in m.suboptions:
            so.data[0] = mat_prop_map[so.name](mat_prop[i])


def set_centrifugal_load(input_file, omega):
    """Set the centrifugal load in the input file

    :param input_file: The input file object
    :param omega: The angular velocity in rad/s
    """
    load_sec = input_file.findKeyword("DLoad")[0]
    load_sec.data[0][2] = create_decimal(omega**2)


def set_temperature_cycles(
    input_file,
    n_cycles,
    one_cycle,
    dT,
    output_division=1,
    # points_per_rise=10,
    # dltmx_min=5,
):
    """Set the temperature cycles in the input file

    :param input_file: The input file object
    :param n_cycles: The number of cycles
    :param one_cycle: The time durations for the four steps of one cycle
    :param dT: The temperature difference between the hot and cold dwells
    :param output_division: How many points for each of the four steps in one cycle
                            1 means no division (4 points per cycle)
    """
    period, Totalcycletimepoint, OutputTimePoints = create_time_points(
        n_cycles, one_cycle, output_division
    )
    solver = input_file.findKeyword("Coupled Temperature-displacement")[0]
    # solver.parameter["deltmx"] = create_decimal(
    #     np.clip(np.floor(dT / points_per_rise), dltmx_min)
    # )
    solver.data[0][1] = create_decimal(Totalcycletimepoint[-1])
    solver.data[0][3] = create_decimal(period)

    input_file.findKeyword("Time Points")[0].data = makeDataList(
        [create_decimal(x) for x in OutputTimePoints], 8
    )

    temp_cycle_k = input_file.findKeyword(
        "Amplitude", parameters={"name": "Temp_Cycling"}
    )[0]

    temperatures = np.zeros(n_cycles * len(one_cycle) + 1)
    temperatures[1:] = np.tile(np.array([dT, dT, 0.0, 0.0]), n_cycles)
    temp_cycle = (
        np.stack([Totalcycletimepoint, temperatures], axis=1).flatten().tolist()
    )
    temp_cycle_k.data = makeDataList(
        [create_decimal(x) for x in temp_cycle], len(one_cycle) * 2
    )


def scale_mesh(input_file, r_o, r_i, thickness):
    r_L = create_decimal(r_o - r_i, 12)
    mesh = input_file.findKeyword("Node")[0].data
    r_max = max([node.data[1] for node in mesh.values()])
    t_max = max([node.data[2] for node in mesh.values()])
    for n_id, node in mesh.items():
        r = node.data[1]
        node.data[1] = create_decimal(r / r_max * r_L + create_decimal(r_i, 12), 12)
        t = node.data[2]
        node.data[2] = create_decimal(t / t_max * create_decimal(thickness, 12), 12)


def rename(input_file, exp_name):
    input_file.inpName = f"{exp_name}.inp"
    heading = input_file.findKeyword("Heading")[0].data
    heading[0] = heading[0].replace("Disk-Universal", exp_name)


def parse_mesh_from_input_file(input_file):
    inp = inpRW.inpRW(
        input_file,
        preserveSpacing=True,
        useDecimal=True,
        organize=True,
    )
    inp.parse()
    node_idx = np.arange(len(inp.findKeyword("Node")[0].data))
    elements = inp.findKeyword("Element")[0]
    n_int_point = 1 if elements.parameter["type"] == "CAX4RT" else 4
    cell_idx = np.arange(len(elements.data) * n_int_point)
    return node_idx, cell_idx

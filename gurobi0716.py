import random
from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple

uc_list = []
ur_list = []

def create_gamma_list(max_num: int, p_violate: float):
    Gamma_list = pd.read_csv("robust_1000.csv")
    Gamma_list = Gamma_list.sort_values(by=['n', 'gamma'])

    N_Gamma_map = [(0, 0)]
    for i in range(1, max_num + 1):
        frame = Gamma_list[Gamma_list['n'] == i]
        for j in range(len(frame)):
            if frame.iloc[j, 1] < p_violate:
                N_Gamma_map.append((i, j))
                break
        else:
            N_Gamma_map.append((i, len(frame)))

    return N_Gamma_map

def load_data_from_json(filepath: str) -> Dict:
    with open(filepath) as f:
        raw_data = json.load(f)
        raw_data = raw_data[:12]

    data = {
        "vm_num": sum(len(pm["vms"]) for pm in raw_data),
        "host_num": len(raw_data),
        "cpu_total": [pm["total_cpu"] for pm in raw_data],
        "mem_total": [pm["total_memory"] for pm in raw_data],
        "threshold": [pm["threshold"] for pm in raw_data],
        "uc": [],
        "ur": [],
        "mem": [],
        "initial_vm_placement": {},
    }

    vm_index = 0
    for pm_index, pm in enumerate(raw_data):
        for vm in pm["vms"]:
            vcpus = vm["vcpus"]
            cpu_usage = np.array(vm["cpu_usage"]) * vcpus / 100
            cpu_max = np.percentile(cpu_usage, 95)
            cpu_min = np.percentile(cpu_usage, 5)
            data["uc"].append((cpu_max + cpu_min) / 2)
            data["ur"].append((cpu_max - cpu_min) / 2)
            data["mem"].append(vm["memory"])
            data["initial_vm_placement"][vm_index] = pm_index
            vm_index += 1

    return data

def main():
    filepath = '15-2.json'
    data = load_data_from_json(filepath)

    global N_Gamma_map
    N_Gamma_map = create_gamma_list(1200, 0.05)

    model = Model("OfflineRobustScheduling2")

    x, x_hat, x_breve, y, H, W, S, Max_ur = {}, {}, {}, {}, {}, {}, {}, {}

    for i in range(data["vm_num"]):
        for j in range(data['host_num']):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name="x[%s,%s]" % (i, j))

    for i in range(data["vm_num"]):
        for j in range(data['host_num']):
            x_hat[i, j] = model.addVar(vtype=GRB.BINARY, name="x_hat[%s,%s]" % (i, j))

    for i in range(data["vm_num"]):
        for j in range(data['host_num']):
            x_breve[i, j] = model.addVar(vtype=GRB.BINARY, name="x_breve[%s,%s]" % (i, j))

    for j in range(data['host_num']):
        y[j] = model.addVar(vtype=GRB.BINARY, name="y[%s]" % j)

    for j in range(data['host_num']):
        for k in range(data["vm_num"] + 1):
            H[j, k] = model.addVar(vtype=GRB.BINARY, name="H[%s,%s]" % (j, k))

    for i in range(data["vm_num"]):
        W[i] = model.addVar(vtype=GRB.BINARY, name="W[%s]" % i)

    for j in range(data['host_num']):
        S[j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="S[%s]" % j)

    for j in range(data['host_num']):
        Max_ur[j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="Max_ur[%s]" % j)

    # Constraints
    for i in range(data["vm_num"]):
        model.addConstr(quicksum(x[i, j] for j in range(data["host_num"])) == 1)

    for i in range(data["vm_num"]):
        for j in range(data["host_num"]):
            model.addConstr((x_hat[i, j] + x_breve[i, j]) == x[i, j])

    for j in range(data["host_num"]):
        model.addConstr(
            quicksum(x[i, j] for i in range(data["vm_num"])) == quicksum(
                H[j, k] * k for k in range(data["vm_num"] + 1)))
        model.addConstr(quicksum(H[j, k] for k in range(data["vm_num"] + 1)) == 1)

    for j in range(data["host_num"]):
        model.addConstr(quicksum(x_hat[i, j] for i in range(data["vm_num"])) == quicksum(
            H[j, k] * N_Gamma_map[k][1] for k in range(data["vm_num"] + 1)))

    for i in range(data["vm_num"]):
        for j in range(data["host_num"]):
            model.addConstr(Max_ur[j] >= x_hat[i, j] * data['ur'][i])

    for j in range(data["host_num"]):
        for i in range(data["vm_num"]):
            model.addConstr(S[j] >= x_breve[i, j] * data["ur"][i])
            model.addConstr((Max_ur[j] - S[j]) >= x_hat[i, j] * (Max_ur[j] - data["ur"][i]))

    # CPU constraints
    for j in range(data["host_num"]):
        model.addConstr(quicksum(x[i, j] * data["uc"][i] for i in range(data['vm_num'])) + quicksum(
            x_hat[i, j] * data["ur"][i] for i in range(data["vm_num"])) <= y[j] * data["cpu_total"][j] * data["threshold"][j])

    # Memory constraints
    for j in range(data["host_num"]):
        model.addConstr(
            quicksum(x[i, j] * data["mem"][i] for i in range(data["vm_num"])) <= y[j] * data["mem_total"][j])

    # Define migration cost
    migration_cost = 1  # Set migration cost for each migration as 1
    total_migration_cost = quicksum((1 - x[i, data["initial_vm_placement"][i]]) for i in range(data["vm_num"]))

    alpha = 10 ** 3
    model.setObjective(alpha * quicksum(y[j] for j in range(data["host_num"])) +
                       total_migration_cost, GRB.MINIMIZE)

    model.setParam("MIPGap", 0.001)
    model.setParam("TimeLimit", 3600)

    model.optimize()

    migration_cost = model.ObjVal
    active_set = [j for j in range(data["host_num"]) if abs(y[j].X - 1) < 0.001]

    x_set, min_set, max_set = {}, {}, {}
    for j in range(data["host_num"]):
        x_set[j], min_set[j], max_set[j] = [], [], []
        for i in range(data["vm_num"]):
            if abs(x[i, j].X - 1) < 0.001:
                x_set[j].append(i)
            if abs(x_hat[i, j].X - 1) < 0.001:
                max_set[j].append(i)
            if abs(x_breve[i, j].X - 1) < 0.001:
                min_set[j].append(i)

    host_vm_num = []
    for j in range(data["host_num"]):
        for k in range(data["vm_num"] + 1):
            if abs(H[j, k].X - 1) < 0.001:
                host_vm_num.append((j, k))

    # Calculate the total number of migrations
    total_migrations = 0
    for i in range(data["vm_num"]):
        initial_host = data["initial_vm_placement"][i]
        for j in range(data["host_num"]):
            if abs(x[i, j].X - 1) < 0.001 and j != initial_host:
                total_migrations += 1

    print("Number on host", host_vm_num)
    print("Run")
    print("Objective function value:", model.ObjVal)
    print("Active set", active_set)
    print("Min set", min_set)
    print("Max set", max_set)
    print("Total number of migrations:", total_migrations)

if __name__ == "__main__":
    main()

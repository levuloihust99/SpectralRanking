import json
import argparse

from tqdm import tqdm
from typing import List, Dict, Any, Union
from collections import defaultdict, Counter

from lib.data_helpers.utils import create_bytedataset
from lib.data_helpers.bytedataset import ByteDataset
from lib.data_helpers.dataitem import DataItem


class CycleStats:
    def __init__(self):
        self.num_cycle_stats = defaultdict(int)
        self.cycle_length_stats = defaultdict(int)

    def update_num_cycles(self, num_cycles: int):
        self.num_cycle_stats[num_cycles] += 1

    def update_cycle_length(self, cycle_length: int):
        self.cycle_length_stats[cycle_length] += 1

    def dump_stats(self):
        with open("cycle_stats.json", "w") as writer:
            json.dump(
                {
                    "num_cycles": self.num_cycle_stats,
                    "cycle_length": self.cycle_length_stats
                },
                writer, indent=4, ensure_ascii=False
            )


def find_cycles(graph: Dict[Any, Union[List, set]]) -> List:
    """Use brute-force algorithm to loop through all paths to find all cycles."""

    nodes = set()
    cloned_graph = defaultdict(set)
    for k, v in graph.items():
        nodes.add(k)
        cloned_graph[k] = set(v)
        for n in v:
            nodes.add(n)

    all_cycles = []
    cycle = []
    visited = {}
    for n in nodes:
        visited[n] = 0

    def brute_force(node) -> bool:
        cycle.append(node)
        visited[node] = counter
        if node in cycle[:-1]: # anchor case: find a cycle
            all_cycles.append(cycle[::])
            cycle.pop()
        else:
            for n in cloned_graph[node]:
                brute_force(n)
            cycle.pop()

    counter = 0
    for n in nodes:
        counter += 1
        if not visited[n]:
            brute_force(n)

    return all_cycles


def do_ignore_cyclic(item: DataItem, do_stats: bool = False):
    comparisons = item.comparisons

    # build a graph where there are only direct edges, i.e. there are
    # no 2-hop edge that connects an ancestor to a descendant. 1-hop edges only.
    adj = defaultdict(set)
    parents = defaultdict(set)

    def get_descendants(node):
        """Return all descendants of this node, i.e. nodes that can reached from this node.
        Use Depth-First Search (DFS) algorithm to traverse graph."""

        visited = {}
        for v in nodes:
            visited[v] = False
        descendants = []
        def dfs(n):
            visited[n] = True
            descendants.append(n)
            for v in adj[n]:
                if not visited[v]:
                    dfs(v)
        dfs(node)
        return descendants

    nodes = set()
    for k, v in comparisons.items():
        g1, g2 = eval(k)
        nodes.add(g1)
        nodes.add(g2)
        preferred =  v.get("preferred")
        if preferred == 1:
            head = g1
            tail = g2
        else:
            head = g2
            tail = g1

        head_descendants = get_descendants(head)
        if tail not in head_descendants: # when `tail` is not descendant of `head`, try to
                                         # build necessary edges as well as cut some no-longer-needed edges.
                                         # otherwise, no need to do anything.
            distance_parents = []
            # check if any of parents of `tail` become distance, since 2-hop edges are not allowed
            for p in parents[tail]:
                tail_parent_descendants = get_descendants(p)
                if head in tail_parent_descendants: # if `head` is descendant of a parent of `tail`,
                                                    # that parent becomes distance, i.e. p -> head -> tail,
                                                    # edge p -> tail becomes 2-hop, making `p` a distance parent.
                    distance_parents.append(p)

            distance_children = []
            tail_descendants = get_descendants(tail)
            for c in adj[head]:
                if c in tail_descendants: # if a child `c` of `head` is descendant of `tail`,
                                          # that child becomes distance, i.e. head -> tail -> c,
                                          # edge head -> c becomes 2-hop, making `c` a distance child.
                    distance_children.append(c)

            # remove 2-hop edges and build direct edge head -> tail
            for p in distance_parents:
                parents[tail].remove(p)
                adj[p].remove(tail)
            for c in distance_children:
                parents[c].remove(head)
                adj[head].remove(c)
            parents[tail].add(head)
            adj[head].add(tail)

    paths = find_cycles(adj)
    cycle_sets = set()
    for path in paths:
        idx = next(i for i, v in enumerate(path) if v == path[-1])
        cycle = path[idx:-1]
        cycle_nodes = tuple(sorted(cycle))
        cycle_sets.add(cycle_nodes)

    if do_stats and len(cycle_sets) > 0:
        if "cycle_stats" not in globals():
            globals()["cycle_stats"] = CycleStats()
        cycle_stats: CycleStats = globals()["cycle_stats"]
        cycle_stats.update_num_cycles(len(cycle_sets))
        for cycle_nodes in cycle_sets:
            cycle_stats.update_cycle_length(len(cycle_nodes))

    out_comparisons = {}
    for k, v in comparisons.items():
        g1, g2 = eval(k)
        in_cycle = False
        for cycle_nodes in cycle_sets:
            if g1 in cycle_nodes and g2 in cycle_nodes:
                in_cycle = True
                break
        if in_cycle:
            continue
        out_comparisons[k] = v

    out_item = DataItem.model_validate({**item.model_dump(), "comparisons": out_comparisons})
    return out_item


def unify_items(items: List[DataItem]):
    if len(items) == 0:
        raise ValueError("The number of items must be > 0")
    comparisons = defaultdict(set)
    for item in items:
        for k, v in item.comparisons.items():
            preferred = v.get("preferred", "not available")
            comparisons[k].add(preferred)

    unified_comparisons = {}
    for k, v in comparisons.items():
        if len(v) == 1:
            comp = next(iter(v))
            if isinstance(comp, int):
                unified_comparisons[k] = {"preferred": comp}

    unified_item = DataItem(
        sample_id=items[0].sample_id,
        article=items[0].article,
        summaries=items[0].summaries,
        comparisons=unified_comparisons
    )
    return unified_item


def unify(dataset: ByteDataset, ignore_cyclic: bool = False, do_stats: bool = False):
    unified_data = []
    tracker = {} # map sample_id to a list of data items,
                 # each item corresponds to a LLM

    for idx, item in enumerate(dataset):
        if item["sample_id"] not in tracker:
            unified_data.append([DataItem.model_validate(item)])
            tracker[item["sample_id"]] = len(unified_data) - 1
        else:
            unified_data[tracker[item["sample_id"]]].append(DataItem.model_validate(item))

    final_data = []
    for idx, items in enumerate(tqdm(unified_data)):
        unified_item = unify_items(items)
        if ignore_cyclic:
            unified_item = do_ignore_cyclic(unified_item, do_stats)
        if not unified_item.comparisons:
            continue
        final_data.append(unified_item)

    return final_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", required=True)
    parser.add_argument("--output_path", "-o", default="data/pairwise_ranking/unify/byte")
    parser.add_argument("--ignore_cyclic", default=False, action="store_true")
    parser.add_argument("--do_stats", default=False, action="store_true")
    args = parser.parse_args()

    dataset = ByteDataset(args.input_path)
    unified_data = unify(dataset, args.ignore_cyclic, args.do_stats)
    if args.do_stats:
        cycle_stats: CycleStats = globals()["cycle_stats"]
        cycle_stats.dump_stats()
    create_bytedataset(args.output_path, unified_data)


if __name__ == "__main__":
    main()

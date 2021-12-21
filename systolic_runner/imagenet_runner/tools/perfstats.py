# Copyright (c) 2021 Simon Guo, Berkeley Architecture Research
 
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

"""
Modified Script for parsing profiling results from onnxruntime. Outputs to console or CSV.
This script focuses on analyzing one trace file.
Prints total durations grouped by node (-n), optype (-t), provider (-p), (optype, provider) (-po), or step (-s)

Ex:
    python perfstats.py -t -l 5 data/trace.json # List durations by optype

Result:
    op_type               duration    percent     count
    --------------------  ----------  ----------  ----------
    MemcpyFromHost        3388472     66.0656981  193
    Conv                  1205958     23.5127978  5
    Loop                  451989      8.81251751  5
    Add                   23115       0.45067765  3482
    Concat                14228       0.27740608  323
"""

import json
import argparse
import shutil
import csv
from collections import namedtuple, defaultdict

_HELP_TEXT = """
Usage Examples:
python perfstats.py [flags] [path to json trace file]
"""


raw_entry_headers = \
    ["name", "duration", "op_type", "provider", "graph_index", "parameter_size", "activation_size", "output_size"]
raw_entry_col_widths = [None, 10, 20, 20, 11, 14, 15, 11]
RawEntry = namedtuple("RawEntry", raw_entry_headers)
node_entry_headers = ["name", "duration", "op_type", "provider", "percent", "count"]
node_entry_col_widths = [None, 10, 20, 20, 10, 10]
NodeEntry = namedtuple("NodeEntry", node_entry_headers)
step_entry_headers = ["name", "duration", "op_type", "provider", "percent"]
step_entry_col_widths = [None, 10, 20, 20, 10]
StepEntry = namedtuple("StepEntry", step_entry_headers)
type_entry_headers = ["op_type", "duration", "percent_time", "count", "execution_provider"]
type_entry_col_widths = [20, 10, 15, 10, None]
OpTypeEntry = namedtuple("OpTypeEntry_time", type_entry_headers)

provider_entry_headers = ["execution_provider", "duration", "percent_time", "ops_count", "percent_ops"]
provider_entry_col_widths = [20, 10, 15, 10, 15]
ProviderEntry = namedtuple("ProviderEntry", provider_entry_headers)

provider_ops_entry_headers = ["execution_provider", "op_type", "op_duration", "op_dur_percent_exec","op_dur_percent_all", "ops_count"]
provider_ops_entry_col_widths = [20, 20, 15, 20, 20, 10]
ProviderOpsEntry = namedtuple("ProviderOpsEntry", provider_ops_entry_headers)


def compute_step_entries(raw_entries):
    total_duration = sum(entry.duration for entry in raw_entries)
    step_entries = []
    for entry in raw_entries:
        percent = entry.duration * 100 / total_duration
        step_entries.append(StepEntry(entry.name, entry.duration, entry.op_type, entry.provider, percent))
    step_entries.sort(key=lambda x: -x.duration)
    total = StepEntry("Total", total_duration, "", "", "")
    return step_entries, total


def compute_node_entries(raw_entries):
    name_to_data = defaultdict(list)
    total_ops = len(raw_entries)
    total_duration = sum(entry.duration for entry in raw_entries)
    for entry in raw_entries:
        name_to_data[entry.name].append(entry)
    node_entries = []
    for name, entries in name_to_data.items():
        duration = sum(entry.duration for entry in entries)
        percent = duration * 100 / total_duration
        op_type = entries[0].op_type
        provider = entries[0].provider
        node_entries.append(NodeEntry(name, duration, op_type, provider, percent, len(entries)))
    node_entries.sort(key=lambda x: -x.duration)
    total = NodeEntry("Total", total_duration, "", "", "", total_ops)
    return node_entries, total


def compute_op_type_entries(raw_entries):
    type_to_data = defaultdict(list)
    total_ops = len(raw_entries)
    total_duration = sum(entry.duration for entry in raw_entries)
    for entry in raw_entries:
        type_to_data[entry.op_type].append(entry)
    op_type_entries = []
    for op_type, entries in type_to_data.items():
        duration = sum(entry.duration for entry in entries)
        percent_time = duration * 100 / total_duration

        exec_provider = set()  

        for entry in entries:
            exec_provider.add(entry.provider)

        op_type_entries.append(OpTypeEntry(op_type, duration, "{:.3f}".format(percent_time), len(entries), list(exec_provider)))
    op_type_entries.sort(key=lambda x: -x.duration)

    total = OpTypeEntry("Total", total_duration, " ", total_ops, " ")
    return op_type_entries, total

def compute_provider_entries(raw_entries):
    provider_to_data = defaultdict(list)
    total_duration = sum(entry.duration for entry in raw_entries)
    total_ops = len(raw_entries)
    for entry in raw_entries:
        provider_to_data[entry.provider].append(entry)
    provider_entries = []
    for provider, entries in provider_to_data.items():
        duration = sum(entry.duration for entry in entries)
        percent_time = duration * 100 / total_duration
        percent_ops = len(entries) * 100 / total_ops
        provider_entries.append(ProviderEntry(provider, duration, "{:.3f}".format(percent_time), len(entries), "{:.3f}".format(percent_ops)))
    provider_entries.sort(key=lambda x: -x.duration)
    total = ProviderEntry("Total", total_duration, "", total_ops, "")
    return provider_entries, total


def compute_provider_ops_entries(raw_entries):
    provider_to_data = defaultdict(list)
    total_duration = sum(entry.duration for entry in raw_entries)
    total_ops = len(raw_entries)

    for entry in raw_entries:
        provider_to_data[entry.provider].append(entry)

    provider_ops_entries = []
    for provider, entries in provider_to_data.items():
        type_to_data = defaultdict(list)

        exec_provider_total_dur = sum(entry.duration for entry in entries)

        for entry in entries:
            type_to_data[entry.op_type].append(entry)

        for op_type, entries in type_to_data.items():
            duration = sum(entry.duration for entry in entries)

            op_dur_percent_exec =  (100*duration)/exec_provider_total_dur
            op_dur_percent_all = (100*duration)/total_duration


            provider_ops_entries.append(ProviderOpsEntry(provider, op_type, duration, 
                            "{:.3f}".format(op_dur_percent_exec), "{:.3f}".format(op_dur_percent_all),
                            len(entries)))

    provider_ops_entries.sort(key=lambda x: (x.execution_provider, -x.op_duration))
    total = ProviderOpsEntry("Total", "", total_duration, "", "", total_ops)
    return provider_ops_entries, total


def read_raw_entries(profile_path):
    with open(profile_path, "r") as f:
        data = json.load(f)
    if type(data) == dict:
        data = data['traceEvents']
    entries = []
    for item in data:
        cat = item.get("cat")
        if cat not in ["Node", "Op"]:
            continue
        arg = item.get('args')
        if not arg:
            continue
        provider = arg.get("provider")
        op = arg.get("op_name")
        if op:
            name = item['name']
            if not name.endswith("_kernel_time"): # we only focus on kernel time
                continue
            provider = provider.replace("ExecutionProvider", "") # make provider name shorter
            dur = item['dur']
            name = name.replace("_kernel_time", "")
            graph_index = arg.get('graph_index')
            parameter_size = arg.get('parameter_size')
            activation_size = arg.get('activation_size')
            output_size = arg.get('output_size')
        if not op:
            continue
        entries.append(RawEntry(name, dur, op, provider, graph_index, parameter_size, activation_size, output_size))
    return entries


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="Parses a json profiling file from onnx runtime",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_HELP_TEXT)
    parser.add_argument("input", help=".json file from onnx runtime")
    parser.add_argument("-t", "--type", action="store_true", help="total execution time per op type (sorted)")
    parser.add_argument("-n", "--node", action="store_true", help="total execution time per node (sorted)")
    parser.add_argument("-s", "--step", action="store_true", help="times for each execution step (sorted)")
    parser.add_argument("-r", "--raw", action="store_true", help="unsorted raw data")
    parser.add_argument("-p", "--provider", action="store_true", help="total execution time per execution provider (sorted)")
    parser.add_argument("-po", "--provider_ops", action="store_true", help="ratio of op time per execution provider (sorted)") 
    parser.add_argument("-d", "--data-only", action="store_true", help="don't include headers")
    parser.add_argument("-q", "--query", help="only include entries satisfying the provided query")
    parser.add_argument("-l", "--limit", type=int, default=-1, help="only show first n results")
    parser.add_argument("-o", "--output", help="output to csv file")
    args = parser.parse_args()
    if sum(bool(a) for a in [args.type, args.node, args.step, args.raw, args.provider, args.provider_ops]) != 1:
        print("exactly one of flags -t, -n, -s, -r, -p, -po must be provided")
        exit(1)
    try:
        if args.query:
            args.query = Query(args.query)
    except Exception:
        print("invalid query: %r" % args.query)
        exit(1)
    return args


class QueryClause:
    def __init__(self, clause_string):
        self.rule_type = 'inc'
        letter = None
        if '!=' in clause_string:
            self.rule_type = 'exc'
            letter, clause_string = clause_string.split('!=')
        elif '=' in clause_string:
            letter, clause_string = clause_string.split('=')
        self.match_name = letter in [None, 'n']
        self.match_type = letter in [None, 't']
        self.patterns = set(clause_string.split(','))

    def match(self, entry):
        if isinstance(entry, NodeEntry) and self.match_name and entry.name in self.patterns:
            return self.rule_type
        if self.match_type and entry.op_type in self.patterns:
            return self.rule_type
        return None


class Query:
    def __init__(self, query_string):
        self.clauses = [QueryClause(s) for s in query_string.split(";")]
        self.no_inc = not any(c.rule_type == 'inc' for c in self.clauses)

    def match(self, entry):
        matches = [c.match(entry) for c in self.clauses]
        return (self.no_inc or 'inc' in matches) and 'exc' not in matches


class TablePrinter:
    def __init__(self, col_widths, padding=2, min_width=5):
        self.col_widths = col_widths
        self.unknown_cnt = col_widths.count(None)
        self.padding = padding
        self.fixed_sum = sum(w for w in col_widths if w is not None) + self.padding * (len(col_widths) - 1)
        self.min_width = min_width

    def get_col_widths(self, total_width):
        remaining_width = total_width - self.fixed_sum
        computed_widths = []
        for i in range(self.unknown_cnt):
            w = remaining_width // (self.unknown_cnt - i)
            remaining_width -= w
            computed_widths.append(max(w, self.min_width))
        col_widths = []
        for w in self.col_widths:
            if w is None:
                col_widths.append(computed_widths.pop())
            else:
                col_widths.append(w)
        return col_widths

    def format(self, entry, width):
        if isinstance(entry, float):
            entry = ("%." + str(width) + "f") % entry
            return entry[:width]
        else:
            entry = str(entry)
        if len(entry) > width:
            x = (width - 3) // 2
            y = width - 3 - x
            entry = entry[:x] + "..." + entry[-y:]
        return entry + " " * (width - len(entry))

    def print_divider(self):
        total_width = shutil.get_terminal_size((80, 20)).columns
        col_widths = self.get_col_widths(total_width)
        line = (" " * self.padding).join("-" * w for w in col_widths)
        print(line)

    def print(self, entries):
        total_width = shutil.get_terminal_size((80, 20)).columns
        col_widths = self.get_col_widths(total_width)
        line = (" " * self.padding).join(self.format(e, w) for e, w in zip(entries, col_widths))
        print(line)


def main():
    args = get_args()
    entries = read_raw_entries(args.input)
    if args.type:
        entries, total = compute_op_type_entries(entries)
    elif args.node:
        entries, total = compute_node_entries(entries)
    elif args.step:
        entries, total = compute_step_entries(entries)
    elif args.provider:
        entries, total = compute_provider_entries(entries)
    elif args.provider_ops:
        entries, total = compute_provider_ops_entries(entries)

    exc_entries = []
    if args.query:
        exc_entries.extend(e for e in entries if not args.query.match(e))
        entries = [e for e in entries if args.query.match(e)]
    if args.limit >= 0:
        exc_entries.extend(entries[args.limit:])
        entries = entries[:args.limit]

    if args.type:
        col_widths = type_entry_col_widths
        headers = type_entry_headers
    elif args.node:
        col_widths = node_entry_col_widths
        headers = node_entry_headers
    elif args.step:
        col_widths = step_entry_col_widths
        headers = step_entry_headers
    elif args.provider:
        col_widths = provider_entry_col_widths
        headers = provider_entry_headers
    elif args.provider_ops:
        col_widths = provider_ops_entry_col_widths
        headers = provider_ops_entry_headers
    elif args.raw:
        col_widths = raw_entry_col_widths
        headers = raw_entry_headers

    if args.output:
        with open(args.output, mode='w', newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if not args.data_only:
                writer.writerow(headers)
            for entry in entries:
                writer.writerow(entry)
    else:
        printer = TablePrinter(col_widths)
        if not args.data_only:
            printer.print_divider()
            printer.print(headers)
            printer.print_divider()
        for entry in entries:
            printer.print(entry)
        printer.print_divider()
        printer.print(total)

if __name__ == '__main__':
    main()
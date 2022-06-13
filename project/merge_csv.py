import csv
import sys

results = dict()
for reader in [csv.reader(open(file, "r"), delimiter=";") for file in sys.argv[1:]]:
    for line in reader:
        if line[0] not in results:
            results[line[0]] = list()

        results[line[0]].extend(line[1:])


out = csv.writer(sys.stdout, delimiter=";")
out.writerow(["graph", *results["graph"]])
del results["graph"]
for name, rs in results.items():
    out.writerow([name, *rs])
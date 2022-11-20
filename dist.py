import csv

with open("train_master.tsv", mode="r") as fh:
    reader = csv.reader(fh, delimiter="\t")
    hoge = {}
    for row in reader:
        if not str(row[1]) in hoge:
            hoge[row[1]] = 1
        else:
            hoge[row[1]] += 1

    print(hoge)

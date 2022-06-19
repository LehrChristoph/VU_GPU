evaluate: evaluate.csv

sparse/evaluate.csv:
	make -C sparse evaluate.csv

dense/evaluate.csv:
	make -C dense evaluate.csv

thrust/evaluate.csv:
	make -C thrust evaluate.csv

evaluate.csv: sparse/evaluate.csv dense/evaluate.csv thrust/evaluate.csv
	python3 merge_csv.py sparse/evaluate.csv dense/evaluate.csv thrust/evaluate.csv > evaluate.csv

clean:
	make -C sparse clean
	make -C dense clean
	make -C thrust clean
	rm evaluate.csv

.PHONY: evaluate.csv

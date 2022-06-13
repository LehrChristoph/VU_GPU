evaluate: evaluate.csv

v1/evaluate.csv:
	make -C v1 evaluate.csv

v2/evaluate.csv:
	make -C v2 evaluate.csv

thrust/evaluate.csv:
	make -C thrust evaluate.csv

evaluate.csv: v1/evaluate.csv v2/evaluate.csv thrust/evaluate.csv
	python3 merge_csv.py v1/evaluate.csv v2/evaluate.csv thrust/evaluate.csv > evaluate.csv

clean:
	make -C v1 clean
	make -C v2 clean
	make -C thrust clean
	rm evaluate.csv

.PHONY: evaluate.csv

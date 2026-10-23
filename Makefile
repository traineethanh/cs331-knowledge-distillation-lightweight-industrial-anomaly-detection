.PHONY: env-check test validate-data smoke-test train-baseline benchmark

env-check:
	python scripts/check_environment.py

test:
	python -m unittest discover -s tests -v

validate-data:
	python scripts/validate_dataset.py --data-root ./datasets/mvtec --categories carpet bottle screw

smoke-test:
	python scripts/train.py --config configs/experiments/smoke_test.yaml

train-baseline:
	python scripts/train.py --config configs/baseline/cda_mvtec.yaml --categories carpet bottle screw

benchmark:
	python scripts/benchmark_latency.py --variants B0 B1_aux_csd B2_sep_adapter B3_cda_lite --img-size 256 --warmup 20 --iters 50

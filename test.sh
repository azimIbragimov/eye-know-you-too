torchrun --nnodes=1 --nproc_per_node=2 src/test.py --config=./config/lohr-22.yaml --fold=0
torchrun --nnodes=1 --nproc_per_node=2 src/test.py --config=./config/lohr-22.yaml --fold=1
torchrun --nnodes=1 --nproc_per_node=2 src/test.py --config=./config/lohr-22.yaml --fold=2
torchrun --nnodes=1 --nproc_per_node=2 src/test.py --config=./config/lohr-22.yaml --fold=3
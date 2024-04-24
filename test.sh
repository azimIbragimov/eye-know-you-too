# Testing EKYT model across every ds value
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=1 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=1 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=1 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=1 --fold=3

CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=2 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=2 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=2 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=2 --fold=3

CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=4 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=4 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=4 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=4 --fold=3

CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=8 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=8 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=8 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=8 --fold=3

CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=20 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=20 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=20 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=20 --fold=3

CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=32 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=32 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=32 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/test.py --ds=32 --fold=3

docker compose build openfhe-cu128

# 백그라운드에서 꺼지지 않게 무한 대기
docker compose run -d --name openfhe-dev openfhe-cu128 tail -f /dev/null

# 들어가기
docker exec -it openfhe-dev bash

# pytorch 동작 확인
python -c "import torch; print('PyTorch:', torch.__version__, ' CUDA:', torch.version.cuda)"

# cpp build
cmake --build . -j"$(nproc)"
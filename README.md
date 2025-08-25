# 백그라운드에서 꺼지지 않게 무한 대기
docker compose run -d --name openfhe-dev openfhe-cu128 tail -f /dev/null

# 들어가기
docker exec -it openfhe-dev bash

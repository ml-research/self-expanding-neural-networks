# IMAGE=$(docker build . --no-cache -q -t egwene)
IMAGE=$(docker build . -q -t egwene)
CONTAINER=$(docker run --rm -v $PWD:/water\
 -v $PWD/results:/water/results\
 -v $PWD/datasets:/datasets\
 -v $PWD/logs:/logs\
 -p 56006:6006\
 -itd ${IMAGE})
docker exec ${CONTAINER} sh -c "tensorboard --logdir /logs &"
docker attach ${CONTAINER}

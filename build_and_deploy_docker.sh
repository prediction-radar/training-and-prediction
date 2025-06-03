cd docker/

docker buildx build --platform linux/amd64 -t radar-training:latest .

docker tag radar-training:latest twiebe/radar-training:latest

docker push twiebe/radar-training:latest
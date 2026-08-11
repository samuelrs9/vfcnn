docker build -t tf_kdtree:test .
docker run --runtime=nvidia --rm -it tf_kdtree:test bash
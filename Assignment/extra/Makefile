run: build ex

ex:
	./script.sh

test: build go

go:
	./out imgs/dart0.jpg imgs/out0.jpg

build:
	g++ truth.cpp /usr/lib64/libopencv_core.so.2.4 \
	/usr/lib64/libopencv_highgui.so.2.4 \
	/usr/lib64/libopencv_imgproc.so.2.4 \
	/usr/lib64/libopencv_objdetect.so.2.4 \
	-o out

run: build ex feh

ex:
	./script.sh

test: build go

go:
	./out imgs/dart0.jpg

build:
	g++ task.cpp /usr/lib64/libopencv_core.so.2.4 \
	/usr/lib64/libopencv_highgui.so.2.4 \
	/usr/lib64/libopencv_imgproc.so.2.4 \
	/usr/lib64/libopencv_objdetect.so.2.4 \
	/usr/lib64/libopencv_features2d.so.2.4 \
	/usr/lib64/libopencv_nonfree.so.2.4 \
	/usr/lib64/libopencv_nonfree.so.2.4 \
	/usr/lib64/libopencv_flann.so.2.4.13 \
	/usr/lib64/libopencv_calib3d.so.2.4 \
	-o out

feh:
	feh imgs/out*

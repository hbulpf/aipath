# FAQ  

## 安装PP OCR

1. 安装ppocr错误
    error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
  --------------------------------------
  	ERROR: Failed building wheel for lanms-neo

  	解决方法: 从微软下载[vs_BuildTools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)按下图并安装

  ![image-20221120203257852](C:\Users\yx\AppData\Roaming\Typora\typora-user-images\image-20221120203257852.png)
  ```
  # lanms-neo Implements Locality-Aware NMS from [EAST](https://github.com/argman/EAST) that may be used across other frameworks like [mmocr](https://github.com/open-mmlab/mmocr). Install pip install lanms-neo
  python from lanms import merge_quadrangle_n9 as la_nms
  ```

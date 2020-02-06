#导入框架
import requests

# 确定url
url=

# 请求  content 二进制（mp3，mp4）  text(文本)
mp3=requests.get(url).comtent
#保存
with open('yinyue.mp3','wb') as file:
    file.write(mp3)
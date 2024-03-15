import joblib

shakes_en = joblib.load("shakespearetxt_en.lzma")
shakes_zh = joblib.load("shakespearetxt_zh.lzma")

print(len(shakes_en), len(shakes_zh), 'lines')
# 179918, 130039, lines
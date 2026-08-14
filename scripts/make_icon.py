# 生成 Tauri 应用图标源图（1024x1024）
from PIL import Image, ImageDraw, ImageFont

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

grad = Image.new("RGB", (S, S))
px = grad.load()
for y in range(S):
    t = y / S
    r = int(47 + (23 - 47) * t)
    g = int(107 + (195 - 107) * t)
    b = int(255 + (178 - 255) * t)
    for x in range(S):
        px[x, y] = (r, g, b)

mask = Image.new("L", (S, S), 0)
d = ImageDraw.Draw(mask)
d.rounded_rectangle([0, 0, S - 1, S - 1], radius=200, fill=255)
img.paste(grad, (0, 0), mask)

hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
dh = ImageDraw.Draw(hl)
dh.rounded_rectangle([0, 0, S - 1, int(S * 0.45)], radius=200, fill=(255, 255, 255, 26))
img = Image.alpha_composite(img, hl)

font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 420)
d2 = ImageDraw.Draw(img)
text = "LR"
bbox = d2.textbbox((0, 0), text, font=font)
w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
d2.text(((S - w) / 2 - bbox[0], (S - h) / 2 - bbox[1] + 20), text, font=font, fill=(255, 255, 255, 255))

img.save("desktop/app-icon.png")
print("icon saved: desktop/app-icon.png", img.size)

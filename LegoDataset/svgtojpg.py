import cairosvg

cairosvg.svg2png(url="lion_16strokes_seed0_best.svg", write_to="temp.png")

from PIL import Image
image = Image.open("temp.png")
image.convert("RGB").save("output.jpg", "JPEG")
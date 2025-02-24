"""
`pycairo` binary needs admin privileges to be installed on the cluster - so we generate the masks locally
"""

import math
import os
import random
from pathlib import Path
from typing import Optional

import cairo
import numpy as np
from PIL import Image


def get_circle_mask(
    width: int = 1000,
    height: int = 1000,
    row_count: int = 3,
    ring_count: int = 12,
    max_radius: Optional[int] = None,
) -> Image.Image:
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    context.set_source_rgba(0, 0, 0, 0)

    max_radius = width / 2 / row_count if max_radius is None else max_radius

    def draw_concentric_circles(x, y, ring_count):
        for i in range(ring_count):
            radius = max_radius * (i + 1) / ring_count
            context.arc(x, y, radius, 0, 2 * math.pi)

            color_ratio = i / (ring_count - 1)
            if color_ratio < 0.10:  # red
                rgb_color = (1, 0, 0)
            elif color_ratio < 0.5:  # gray
                rgb_color = (1, 1, 0)
            elif color_ratio < 0.75:  # yellow
                rgb_color = (0.5, 0.5, 0.5)
            else:  # blue
                rgb_color = (0, 0, 1)

            context.set_source_rgb(*rgb_color)
            context.set_line_width(1.5)
            context.stroke()

    for row in range(row_count):
        for col in range(row_count):
            x = (col + 0.5) * width / row_count
            y = (row + 0.5) * width / row_count
            draw_concentric_circles(x, y, ring_count)

    # surface.write_to_png(Path("circles.png"))
    return Image.frombuffer("RGBA", (width, height), surface.get_data(), "raw", "BGRA", 0, 1)


def get_square_mask(
    width: int = 1000,
    height: int = 1000,
    row_count: int = 3,
    square_count: int = 10,
    max_square_width: Optional[int] = None,
):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    context.set_source_rgba(0, 0, 0, 0)

    def draw_concentric_squares(x, y, size):
        step = size / square_count

        for i in range(square_count):
            if i == 0:
                context.set_source_rgb(0, 0, 1)
            elif i == square_count - 1:
                context.set_source_rgb(1, 0, 0)
            else:
                brown = 0.6 - (i / square_count) * 0.4
                context.set_source_rgb(brown, brown * 0.7, 0)

            width = size - i * step
            height = size - i * step
            context.rectangle(x + (size - width) / 2, y + (size - height) / 2, width, height)
            context.stroke()

    cell_size = min(width // row_count, height // row_count)
    if max_square_width:
        cell_size = min(cell_size, max_square_width)

    for row in range(row_count):
        for col in range(row_count):
            x = col * (width // row_count) + (width // row_count - cell_size) // 2
            y = row * (height // row_count) + (height // row_count - cell_size) // 2
            draw_concentric_squares(x, y, cell_size)

    return Image.frombuffer("RGBA", (width, height), surface.get_data(), "raw", "BGRA", 0, 1)


def get_word_mask(
    width: int = 1000,
    height: int = 1000,
    num_words: int = 15,
    font_range: tuple[int, int] = (20, 100),
    words: list[str] = ["cat", "guacamole", "hat", "penguin", "dog", "elephant"],
    avoid_center: bool = True,
):
    import matplotlib

    matplotlib.use("Agg")  # matplotlib can't render fonts

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    context.set_source_rgba(0, 0, 0, 0)
    context.paint()

    for _ in range(num_words):
        if avoid_center:
            x = random.choices([random.randint(0, width // 10), random.randint(width * 9 // 10, width), random.randint(0, width)], weights=[4, 4, 1])[0]
            y = random.choices([random.randint(0, height // 10), random.randint(height * 9 // 10, height), random.randint(0, height)], weights=[4, 4, 1])[0]
        else:
            x = random.randint(0, width)
            y = random.randint(0, height)

        context.set_font_size(random.randint(*font_range))

        word = random.choice(words)
        orientation = random.choice(["horizontal", "vertical", "flipped"])

        context.save()
        context.translate(x, y)

        if orientation == "vertical":
            context.rotate(-math.pi / 2)
        elif orientation == "flipped":
            context.rotate(random.uniform(0, 2 * math.pi))
            context.scale(-1 if random.random() > 0.5 else 1, -1 if random.random() > 0.5 else 1)

        grayshade = random.random()
        context.set_source_rgb(grayshade, grayshade, grayshade)

        context.move_to(0, 0)
        context.show_text(word)

        context.restore()

    return Image.fromarray(np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=surface.get_data()), "RGBA")


def get_knit_mask(
    width: int = 1000,
    height: int = 1000,
    step: int = 200,
):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    context.set_source_rgba(0, 0, 0, 0)

    def draw_knit(x, y, size, color):
        context.set_source_rgb(*[int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)])
        context.move_to(x, y - size)
        context.line_to(x + size, y)
        context.line_to(x, y + size)
        context.line_to(x - size, y)
        context.close_path()
        context.stroke()

    for x in range(0, width + step, step):
        for y in range(0, height + step, step):
            knit_colors = ["#0000FF", "#008000", "#804000", "#FF0000"]
            for i, color in enumerate(knit_colors):
                size = step - (i * step / 4)
                draw_knit(x, y, size, color)

    return Image.frombuffer("RGBA", (width, height), surface.get_data(), "raw", "BGRA", 0, 1)


def get_diamond_mask(
    width: int = 1000,
    height: int = 1000,
    diamond_count: int = 10,
    diamonds_per_row: int = 5,
):
    diamond_count = int(diamond_count)
    diamond_size = width // diamonds_per_row

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    context.set_source_rgba(0, 0, 0, 0)

    def draw_diamond(x, y, size, color):
        context.set_source_rgba(*color)
        context.move_to(x, y - size / 2)
        context.line_to(x + size / 2, y)
        context.line_to(x, y + size / 2)
        context.line_to(x - size / 2, y)
        context.close_path()
        context.stroke()

    def get_color(i, max_i):
        if i == max_i:
            return (0, 0, 1, 1)  # blue for outermost diamond
        elif i == 1:
            return (1, 0, 0, 1)  # red for innermost diamond
        else:
            t = (i - 1) / (max_i - 1)  # blue -> red gradient
            return (0, 0.5 * (1 - t), 0, 1)

    rows = int(height // (diamond_size // 2) + 2)
    cols = int(diamonds_per_row)

    for row in range(-1, rows):
        for col in range(-1, cols + 1):
            center_x = col * diamond_size
            center_y = row * (diamond_size // 2)

            # row offset
            if row % 2 == 1:
                center_x += diamond_size / 2

            for i in range(diamond_count, 0, -1):
                size = diamond_size * i / diamond_count
                color = get_color(i, diamond_count)
                draw_diamond(center_x, center_y, size, color)

    return Image.frombuffer("RGBA", (width, height), surface.get_data(), "raw", "BGRA", 0, 1)


def get_polygon_mask(
    width: int = 1000,
    height: int = 1000,
    num_sides: int = 6,
    num_polygons_per_row: int = 4,
    num_polygons_per_col: int = 4,
    num_polygons_concentric: int = 5,
    hcaptcha_colors: bool = True,
):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    context.set_source_rgba(0, 0, 0, 0)

    def draw_regular_polygon(x, y, size, sides):
        context.save()
        context.translate(x + size / 2, y + size / 2)
        angle = 2 * math.pi / sides
        radius = size / 2

        context.move_to(radius, 0)
        for i in range(1, sides):
            context.line_to(radius * math.cos(angle * i), radius * math.sin(angle * i))
        context.close_path()
        context.stroke()
        context.restore()

    def draw_concentric_polygons(x, y, size):
        step = size / num_polygons_concentric

        for i in range(num_polygons_concentric):
            if hcaptcha_colors:
                if i == 0:
                    context.set_source_rgb(0, 0, 1)  # blue for outermost
                elif i == num_polygons_concentric - 1:
                    context.set_source_rgb(1, 0, 0)  # red for innermost
                else:
                    brown = 0.6 - (i / num_polygons_concentric) * 0.4  # brown gradient for middle polygons
                    context.set_source_rgb(brown, brown * 0.7, 0)
            else:
                gray = i / num_polygons_concentric
                context.set_source_rgb(gray, gray, gray)

            current_size = size - i * step
            draw_regular_polygon(x + (size - current_size) / 2, y + (size - current_size) / 2, current_size, num_sides)

    cell_size = min(width // num_polygons_per_row, height // num_polygons_per_col)

    for row in range(num_polygons_per_col):
        for col in range(num_polygons_per_row):
            x = col * (width // num_polygons_per_row) + (width // num_polygons_per_row - cell_size) // 2
            y = row * (height // num_polygons_per_col) + (height // num_polygons_per_col - cell_size) // 2
            draw_concentric_polygons(x, y, cell_size)

    return Image.frombuffer("RGBA", (width, height), surface.get_data(), "raw", "BGRA", 0, 1)


def get_current_dir() -> Path:
    try:
        return Path(__file__).parent.absolute()
    except NameError:
        return Path(os.getcwd())


def add_overlay(background: Image.Image, overlay: Image.Image, opacity: int) -> Image.Image:
    # opacity: [0 (transparent); 255 (opaque)]
    overlay = overlay.resize(background.size)
    result = Image.new("RGBA", background.size)
    result.paste(background, (0, 0))
    mask = Image.new("L", overlay.size, opacity)
    result.paste(overlay, (0, 0), mask)
    return result


if __name__ == "__main__":
    mask_dir = get_current_dir() / "masks"
    mask_dir.mkdir(exist_ok=True)

    height = 224
    width = 224

    for sides in [3, 4, 6, 10]:
        for per_rowcol in [2, 4, 10]:
            for num_concentric in [2, 5, 10]:
                for colors in [True, False]:
                    img = get_polygon_mask(
                        width=height,
                        height=width,
                        # config
                        num_sides=sides,
                        num_polygons_per_row=per_rowcol,
                        num_polygons_per_col=per_rowcol,
                        num_polygons_concentric=num_concentric,
                        hcaptcha_colors=colors,
                    )

                    fp = mask_dir / f"{sides}_{per_rowcol}_{num_concentric}_{colors}.png"
                    img.save(fp)

    # from PIL import Image
    # import requests

    # url = "https://sueszli.github.io/datasets/cat_1966.jpeg"
    # img = Image.open(requests.get(url, stream=True).raw).convert("RGBA")
    # img = img.crop((0, img.height - img.width, img.width, img.height))
    # img = img.resize((width, height))
    # mask = get_polygon_mask(
    #     width=width,
    #     height=height,
    #     num_sides=10,
    #     num_polygons_per_row=10,
    #     num_polygons_per_col=10,
    #     num_polygons_concentric=2,
    #     hcaptcha_colors=True,
    # )
    # img = add_overlay(img, mask, 128)
    # img.show()

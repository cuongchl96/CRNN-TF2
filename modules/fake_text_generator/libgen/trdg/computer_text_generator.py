import random as rnd
import numpy as np 
import cv2
from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter


def generate(
    text, font, text_color, font_size, orientation, space_width, character_spacing, fit
):
    if orientation == 0:
        if rnd.random() < .0:
            return _generate_horizontal_text_underline(
                text, font, text_color, font_size, space_width, character_spacing, fit
            )
        else:
            return _generate_horizontal_text(
                text, font, text_color, font_size, space_width, character_spacing, fit
            )
    elif orientation == 1:
        return _generate_vertical_text(
            text, font, text_color, font_size, space_width, character_spacing, fit
        )
    else:
        raise ValueError("Unknown orientation " + str(orientation))


def _generate_horizontal_text(
    text, font, text_color, font_size, space_width, character_spacing, fit
):
    offset1 = 0
    offset2 = 0
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_width = int(image_font.getsize(" ")[0] * space_width)

    char_widths = [image_font.getsize(c)[0] if c != " " else space_width for c in text]
    text_width = sum(char_widths) + character_spacing * (len(text) - 1)
    text_height = max([image_font.getsize(c)[1] for c in text])

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
    txt_mask_draw.fontmode = "1"

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]
    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )

    fill = (
        rnd.randint(40, 70),
        rnd.randint(40, 70),
        rnd.randint(40, 70),
    )

    for i, c in enumerate(text):
        txt_img_draw.text(
            (sum(char_widths[0:i]) + i * character_spacing, 0),
            c,
            fill=fill,
            font=image_font,
        )

        txt_mask_draw.text(
            (sum(char_widths[0:i]) + i * character_spacing, 0),
            c,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
            font=image_font,
        )

    # if rnd.random() <.2:
    #     txt_img_draw.line((0, 0, 0, text_height), fill=fill, width=rnd.randint(0, 4))

    # if rnd.random() <.2:
    #     txt_img_draw.line((0, 0, text_width, 0), fill=fill, width=rnd.randint(0, 4))

    # if rnd.random() <.2:
    #     txt_img_draw.line((text_width, 0, text_width, text_height), fill=fill, width=rnd.randint(0, 4))
    

    if True:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask


def _generate_horizontal_text_underline(
    text, font, text_color, font_size, space_width, character_spacing, fit
):
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_width = int(image_font.getsize(" ")[0] * space_width)

    char_widths = [image_font.getsize(c)[0] if c != " " else space_width for c in text]
    text_width = sum(char_widths) + character_spacing * (len(text) - 1)

    list_size = [image_font.getsize(c)[1] for c in text]
    text_height = max(list_size) + 3
    underline_height = max(list_size, key=list_size.count)

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
    txt_mask_draw.fontmode = "1"

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )

    for i, c in enumerate(text):
        txt_img_draw.text(
            (sum(char_widths[0:i]) + i * character_spacing, 0),
            c,
            fill=fill,
            font=image_font,
        )

        txt_mask_draw.text(
            (sum(char_widths[0:i]) + i * character_spacing, 0),
            c,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
            font=image_font,
        )

    pos = (0, 0)
    ly = pos[1] + underline_height + rnd.randint(0, 2)

    if rnd.random() < .7:
        lx = 0
        len_x = text_width
    else:
        lx = rnd.randint(0, int(text_width / 2))
        len_x = rnd.randint(1, text_width - lx)

    txt_img_draw.line((lx, ly, lx + len_x, ly), fill=fill, width=rnd.randint(0, 4))

    if rnd.random() <.2:
        txt_img_draw.line((0, 0, 0, text_height), fill=fill, width=rnd.randint(0, 4))

    if rnd.random() <.2:
        txt_img_draw.line((0, 0, text_width, 0), fill=fill, width=rnd.randint(0, 4))

    if rnd.random() <.2:
        txt_img_draw.line((text_width, 0, text_width, text_height), fill=fill, width=rnd.randint(0, 4))
    
    if rnd.random() < .3:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask


def _generate_vertical_text(
    text, font, text_color, font_size, space_width, character_spacing, fit
):
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_height = int(image_font.getsize(" ")[1] * space_width)

    char_heights = [
        image_font.getsize(c)[1] if c != " " else space_height for c in text
    ]
    text_width = max([image_font.getsize(c)[0] for c in text])
    text_height = sum(char_heights) + character_spacing * len(text)

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_img)

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(c1[0], c2[0]),
        rnd.randint(c1[1], c2[1]),
        rnd.randint(c1[2], c2[2]),
    )

    for i, c in enumerate(text):
        txt_img_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=fill,
            font=image_font,
        )

        txt_mask_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=(i // (255 * 255), i // 255, i % 255),
            font=image_font,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask
